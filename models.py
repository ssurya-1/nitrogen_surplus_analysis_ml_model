"""Budget-constrained multi-task network: one model for all four crops that predicts yield and
harvested N removal, and computes N surplus from the soil-surface budget so the outputs always add up."""
import copy

import numpy as np
import torch
from torch import nn

FEATURES = ["lon", "lat", "year", "mineral", "manure", "deposition", "area_share", "log_uaar", "LU"]
TARGETS = ["yield_tha", "removal", "surplus"]
CROPS = ["SWHE", "BARL", "MAIZ", "RAPE"]
VIOLATION_KG = 1.0


class Network(nn.Module):
    """Shared layers for all crops, a learned crop embedding, and a yield-and-removal head per crop."""

    def __init__(self, n_in, dropout=0.1, emb_dim=4):
        super().__init__()
        self.embed = nn.Embedding(len(CROPS), emb_dim)
        self.body = nn.Sequential(nn.Linear(n_in + emb_dim, 128), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout))
        self.heads = nn.Linear(64, 2 * len(CROPS))

    def forward(self, x, crop):
        h = self.body(torch.cat([x, self.embed(crop)], dim=1))
        return self.heads(h).view(-1, len(CROPS), 2)[torch.arange(len(crop)), crop]


class Scaler:
    def __init__(self, train):
        self.x_mean = train["x"].mean(0)
        self.x_std = train["x"].std(0).clamp_min(1e-6)
        self.y_mean = train["y"].mean(0)
        self.y_std = train["y"].std(0)

    def apply(self, batch):
        out = dict(batch)
        out["x"] = (batch["x"] - self.x_mean) / self.x_std
        out["z"] = (batch["y"] - self.y_mean) / self.y_std
        return out


def to_batch(df, rho):
    return {"x": torch.tensor(df[FEATURES].to_numpy(), dtype=torch.float32),
            "y": torch.tensor(df[TARGETS].to_numpy(), dtype=torch.float32),
            "crop": torch.tensor(df["crop"].map(CROPS.index).to_numpy()),
            "n_in": torch.tensor(df["n_inputs"].to_numpy(), dtype=torch.float32),
            "rho": torch.tensor(rho, dtype=torch.float32)}


def take(batch, idx):
    return {k: v[idx] for k, v in batch.items()}


def outputs(model, b, scaler):
    """Standardised yield, removal and surplus. Surplus is computed from the budget, not predicted:
    inputs plus residues (rho x removal) minus removal."""
    z = model(b["x"], b["crop"])
    removal = z[:, 1] * scaler.y_std[1] + scaler.y_mean[1]
    surplus = b["n_in"] + (b["rho"] - 1) * removal
    return torch.stack([z[:, 0], z[:, 1], (surplus - scaler.y_mean[2]) / scaler.y_std[2]], dim=1)


def budget_residual(y, b):
    """kg N/ha left over when predicted removal and surplus are put back into the budget."""
    return b["n_in"] + (b["rho"] - 1) * y[:, 1] - y[:, 2]


def loss_fn(scaler):
    # The surplus error is measured after the budget step, so it also trains the removal head.
    def loss(model, b):
        return ((outputs(model, b, scaler) - b["z"]) ** 2).mean()
    return loss


def train(model, loss, train_b, val_b, lr, wd, seed, epochs=400, patience=25, batch_size=256):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, wait = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(train_b["x"]), generator=gen).split(batch_size):
            opt.zero_grad()
            loss(model, take(train_b, idx)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss(model, val_b).item()
        if val < best - 1e-5:
            best, wait, best_state = val, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best


def fit(train_b, val_b, scaler, lr, wd, seed):
    torch.manual_seed(seed)
    model, val = train(Network(len(FEATURES)), loss_fn(scaler), train_b, val_b, lr, wd, seed)

    def predict(b):
        with torch.no_grad():
            return (outputs(model, b, scaler) * scaler.y_std + scaler.y_mean).numpy()
    return predict, val, model
