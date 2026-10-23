"""Blocked splits, training runs, metrics and the hotspot ranking for the budget-constrained network.

Run:  .venv/bin/python evaluate.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

import models as M

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"
MODELS_DIR = ROOT / "models"
MODEL_FILE = MODELS_DIR / "model_temporal_seed0.pt"

SEEDS = [0, 1, 2, 3, 4]
FIT_LAST, VAL_YEARS, TEST_YEARS = 2014, [2015, 2016], [2017, 2018]
N_FOLDS = 5


def load():
    df = pd.read_parquet(PROCESSED / "crop_budget_nuts2.parquet").reset_index(drop=True)
    df["log_uaar"] = np.log(df["UAAR_ha"])
    adjacency = pd.read_csv(INTERIM / "region_adjacency.csv")
    return df, adjacency


def make_splits(df, adjacency):
    splits = [("temporal", 0, df["year"] <= FIT_LAST, df["year"].isin(VAL_YEARS), df["year"].isin(TEST_YEARS))]
    pool = df["year"] <= max(VAL_YEARS)
    pairs = adjacency[["region_a", "region_b"]].to_numpy()
    folds = GroupKFold(N_FOLDS).split(df[pool], groups=df.loc[pool, "CNTR_CODE"])
    for k, (_, test_pos) in enumerate(folds, start=1):
        test_countries = set(df.loc[pool, "CNTR_CODE"].iloc[test_pos])
        in_test = df["CNTR_CODE"].isin(test_countries)
        test_regions = set(df.loc[in_test, "region"])
        touching = {b for a, b in pairs if a in test_regions} | {a for a, b in pairs if b in test_regions}
        usable = pool & ~in_test & ~df["region"].isin(touching)
        splits.append(("spatial", k, usable & (df["year"] <= FIT_LAST), usable & df["year"].isin(VAL_YEARS),
                       pool & in_test))
        train_regions = set(df.loc[usable, "region"])
        assert not train_regions & test_regions
        assert not any((a in train_regions and b in test_regions) or (b in train_regions and a in test_regions)
                       for a, b in pairs)
    return splits


def past_average_reference(df, splits):
    """Reference point, not a model: predict each test row's yield as its region-crop mean over the
    fitting years (crop mean where the region has no history)."""
    errors = []
    for name, _, fit, _, test in splits:
        past = df[fit].groupby(["region", "crop"])["yield_tha"].mean().rename("past")
        crop_mean = df[fit].groupby("crop")["yield_tha"].mean()
        t = df[test].join(past, on=["region", "crop"])
        errors.append(pd.DataFrame({"split": name, "err": t["past"].fillna(t["crop"].map(crop_mean)) - t["yield_tha"]}))
    err = pd.concat(errors).groupby("split")["err"]
    err.agg(yield_mae=lambda e: e.abs().mean(), yield_bias="mean").to_csv(RESULTS / "reference_past_average.csv")


def known_rho(df, fit):
    """Residue ratio available at prediction time: the region's fitting-year mean, else the overall median."""
    by_region = df[fit].groupby("region")["residue_ratio"].mean()
    return df["region"].map(by_region).fillna(df.loc[fit, "residue_ratio"].median()).to_numpy()


def leakage_check(df, fit):
    corr = df.loc[fit, M.FEATURES + M.TARGETS].corr().loc[M.FEATURES, M.TARGETS]
    corr.round(3).to_csv(RESULTS / "leakage_check.csv")
    return corr.abs().max().max()


def batches(df, split):
    _, _, fit, val, test = split
    rho = known_rho(df, fit)
    raw = {k: M.to_batch(df[m], rho[m.to_numpy()]) for k, m in [("fit", fit), ("val", val), ("test", test)]}
    scaler = M.Scaler(raw["fit"])
    return {k: scaler.apply(v) for k, v in raw.items()}, scaler


def tune(df, split):
    b, scaler = batches(df, split)
    rows = []
    for lr in (1e-3, 3e-3):
        for wd in (0.0, 1e-4):
            _, v, _ = M.fit(b["fit"], b["val"], scaler, lr, wd, seed=0)
            rows.append({"lr": lr, "weight_decay": wd, "val_mse": v})
    grid = pd.DataFrame(rows)
    grid.to_csv(RESULTS / "tuning.csv", index=False)
    best = grid.loc[grid["val_mse"].idxmin()]
    return best["lr"], best["weight_decay"]


def run_split(df, split, lr, wd):
    name, fold, _, _, test = split
    b, scaler = batches(df, split)
    rows = np.flatnonzero(test)
    preds = []
    for seed in SEEDS:
        t = time.time()
        predict, _, net = M.fit(b["fit"], b["val"], scaler, lr, wd, seed)
        y = predict(b["test"])
        preds.append(pd.DataFrame({"row": rows, "seed": seed, "split": name, "fold": fold,
                                   "pred_yield_tha": y[:, 0], "pred_removal": y[:, 1], "pred_surplus": y[:, 2],
                                   "residual": M.budget_residual(torch.tensor(y), b["test"]).numpy()}))
        if name == "temporal" and seed == 0:
            torch.save(net.state_dict(), MODEL_FILE)
        print(f"{name} fold {fold} seed {seed}: {time.time() - t:.0f}s", flush=True)
    return pd.concat(preds, ignore_index=True)


def error_metrics(g):
    out = {}
    for target, col in zip(M.TARGETS, ["pred_yield_tha", "pred_removal", "pred_surplus"]):
        err = g[col] - g[target]
        ss = ((g[target] - g[target].mean()) ** 2).sum()
        out.update({f"{target}_mae": err.abs().mean(), f"{target}_rmse": np.sqrt((err ** 2).mean()),
                    f"{target}_bias": err.mean(), f"{target}_r2": 1 - (err ** 2).sum() / ss})
    out["residual_max_abs"] = g["residual"].abs().max()
    out["violation_rate"] = (g["residual"].abs() > M.VIOLATION_KG).mean()
    out["n"] = g["row"].nunique()
    return pd.Series(out)


METRIC_COLS = M.TARGETS + ["pred_yield_tha", "pred_removal", "pred_surplus", "residual", "row"]


def summarise(df, preds):
    full = preds.join(df[M.TARGETS + ["crop", "CNTR_CODE", "year"]], on="row")
    per_seed = full.groupby(["split", "seed"])[METRIC_COLS].apply(error_metrics)
    per_seed.to_csv(RESULTS / "metrics_runs.csv")
    summary = per_seed.groupby("split").agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.to_csv(RESULTS / "metrics_summary.csv")
    for key in ["crop", "CNTR_CODE", "year"]:
        full.groupby(["split", key])[METRIC_COLS].apply(error_metrics).to_csv(RESULTS / f"metrics_by_{key.lower()}.csv")
    return summary


def hotspots(df, preds):
    p = preds[preds["split"] == "temporal"].groupby("row")[["pred_yield_tha", "pred_surplus"]].mean()
    t = df.loc[p.index, ["region", "CNTR_CODE", "CNTR_NAME", "year", "crop", "area_ha",
                         "yield_tha", "surplus"]].join(p)
    t["pred_surplus_per_t"] = t["pred_surplus"] / t["pred_yield_tha"]
    t["surplus_per_t"] = t["surplus"] / t["yield_tha"]
    med = t.groupby("crop")[["pred_yield_tha", "pred_surplus"]].transform("median")
    high_y, high_s = t["pred_yield_tha"] >= med["pred_yield_tha"], t["pred_surplus"] >= med["pred_surplus"]
    t["quadrant"] = np.select([high_y & ~high_s, high_y & high_s, ~high_y & ~high_s],
                              ["high yield, low surplus", "high yield, high surplus", "low yield, low surplus"],
                              "low yield, high surplus")
    t["rank_in_crop"] = t.groupby("crop")["pred_surplus_per_t"].rank(ascending=False)
    t.to_csv(RESULTS / "hotspots.csv", index=False)

    both_years = t.groupby(["region", "crop"])["quadrant"].agg(lambda q: q.nunique() == 1 and len(q) == 2)
    agreement = t.groupby("crop")[["pred_surplus_per_t", "surplus_per_t"]].apply(
        lambda g: g["pred_surplus_per_t"].rank().corr(g["surplus_per_t"].rank()))
    pd.DataFrame({"spearman_pred_vs_budget": agreement,
                  "regions_same_quadrant_both_years": both_years.groupby("crop").mean()}) \
        .to_csv(RESULTS / "hotspot_reliability.csv")
    return t


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    df, adjacency = load()
    splits = make_splits(df, adjacency)
    max_corr = leakage_check(df, splits[0][2])
    print(f"max |r| feature vs target: {max_corr:.2f}", flush=True)
    past_average_reference(df, splits)
    lr, wd = tune(df, splits[0])
    print(f"tuned lr={lr} weight_decay={wd}", flush=True)
    pd.DataFrame([{"split": s, "fold": f, "fit_rows": int(a.sum()), "val_rows": int(v.sum()),
                   "test_rows": int(t.sum()), "test_countries": " ".join(sorted(df.loc[t, "CNTR_CODE"].unique()))}
                  for s, f, a, v, t in splits]).to_csv(RESULTS / "splits.csv", index=False)
    preds = pd.concat([run_split(df, s, lr, wd) for s in splits], ignore_index=True)
    preds.to_parquet(RESULTS / "predictions.parquet", index=False)
    summarise(df, preds)
    hotspots(df, preds)


if __name__ == "__main__":
    main()
