"""Predict yield, harvested N removal and N surplus for one region, crop and year with the saved
budget-constrained network (temporal split, seed 0), and compare with the budget values.

Run:  .venv/bin/python predict.py --region DE11 --crop wheat --year 2018
      .venv/bin/python predict.py --list DE          (regions and crops available in a country)
"""
import argparse

import torch

import evaluate as E
import models as M

CROP_ALIASES = {"wheat": "SWHE", "soft wheat": "SWHE", "barley": "BARL", "maize": "MAIZ", "grain maize": "MAIZ",
                "rapeseed": "RAPE", "canola": "RAPE"}
CROP_NAMES = {"SWHE": "soft wheat", "BARL": "barley", "MAIZ": "grain maize", "RAPE": "rapeseed"}


def predict_rows(df, adjacency, rows):
    split = E.make_splits(df, adjacency)[0]
    _, scaler = E.batches(df, split)
    net = M.Network(len(M.FEATURES))
    net.load_state_dict(torch.load(E.MODEL_FILE))
    net.eval()
    rho = E.known_rho(df, split[2])
    b = scaler.apply(M.to_batch(df.loc[rows], rho[rows]))
    with torch.no_grad():
        y = M.outputs(net, b, scaler) * scaler.y_std + scaler.y_mean
    return y.numpy(), M.budget_residual(y, b).numpy()


def role(year):
    if year <= E.FIT_LAST:
        return "training year"
    return "validation year" if year in E.VAL_YEARS else "test year (not seen in training)"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", help="CAPRI region code or prefix, e.g. DE11 or FR24")
    p.add_argument("--crop", help="wheat, barley, maize or rapeseed")
    p.add_argument("--year", type=int, help="2000 to 2018")
    p.add_argument("--list", metavar="COUNTRY", help="list regions and crops for a two-letter country code")
    args = p.parse_args()

    df, adjacency = E.load()
    if args.list:
        sub = df[df["CNTR_CODE"] == args.list.upper()]
        if sub.empty:
            raise SystemExit(f"No rows for country '{args.list}'. Codes: {' '.join(sorted(df['CNTR_CODE'].unique()))}")
        print(sub.groupby("region")["crop"].unique().apply(lambda c: ", ".join(sorted(c))).to_string())
        return
    if not (args.region and args.crop and args.year):
        p.error("give --region, --crop and --year, or --list COUNTRY")

    crop = CROP_ALIASES.get(args.crop.lower(), args.crop.upper())
    match = df[df["region"].str.startswith(args.region.upper()) & (df["crop"] == crop) & (df["year"] == args.year)]
    if match.empty:
        raise SystemExit("No data for that region, crop and year (the crop may be under 1,000 ha there). "
                         "Try --list with the country code.")
    if match["region"].nunique() > 1:
        raise SystemExit("That code matches several regions: " + " ".join(sorted(match["region"].unique())))
    row = match.index[:1]
    r = df.loc[row[0]]
    y, residual = predict_rows(df, adjacency, row)

    print(f"\n{r['region']} ({r['CNTR_NAME']}), {CROP_NAMES[crop]}, {args.year}: {role(args.year)}")
    print(f"Inputs: mineral {r['mineral']:.0f}, manure {r['manure']:.0f}, deposition {r['deposition']:.0f}, "
          f"fixation {r['fixation']:.1f} kg N/ha; crop area {r['area_ha']:,.0f} ha\n")
    print(f"{'':28}{'Budget value':>14}{'Model':>14}{'Error':>14}")
    for i, (label, col) in enumerate([("Yield (t/ha)", "yield_tha"), ("Removal (kg N/ha)", "removal"),
                                      ("Surplus (kg N/ha)", "surplus")]):
        print(f"{label:28}{r[col]:>14.2f}{y[0, i]:>14.2f}{y[0, i] - r[col]:>+14.2f}")
    print(f"{'Budget residual (kg N/ha)':28}{'':>14}{residual[0]:>14.2f}")
    print("\nThe residual is how far predicted removal and surplus are from adding up to the inputs; the model "
          "computes surplus from the budget, so it is zero. Surplus is a potential-loss indicator, not a "
          "fertiliser recommendation.")


if __name__ == "__main__":
    main()
