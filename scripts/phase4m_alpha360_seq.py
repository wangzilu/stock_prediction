"""Phase 4M.2: Sequence deep models on Alpha360 (60-day OHLCV).

This is where deep models should shine — temporal patterns in raw price data.
XGB treats 360 features as flat tabular; LSTM/Transformer see (60, 6) sequences.

Usage:
    python scripts/phase4m_alpha360_seq.py
"""
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
OUTPUT_DIR = DATA_DIR / "phase4m"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 6  # fewer splits — memory limited
TEST_DAYS = 20
VALID_DAYS = 40
TRAIN_DAYS = 250  # ~1 year training to fit in memory
LABEL_COL = "__label_5d"

# OHLCV field order in Alpha360
FIELDS = ["CLOSE", "OPEN", "HIGH", "LOW", "VWAP", "VOLUME"]
N_DAYS = 60
N_FIELDS = len(FIELDS)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reshape_to_sequence(X_flat: np.ndarray) -> np.ndarray:
    """Reshape (batch, 360) → (batch, 60, 6) for sequence models.

    Column order: CLOSE59..CLOSE0, OPEN59..OPEN0, ..., VOLUME59..VOLUME0
    → reshape to (batch, 60_days, 6_fields) with day 0 = most recent
    """
    batch = X_flat.shape[0]
    seq = np.zeros((batch, N_DAYS, N_FIELDS), dtype=np.float32)
    for i, field in enumerate(FIELDS):
        # Columns are FIELD59, FIELD58, ..., FIELD0 (59=oldest, 0=newest)
        # We want seq[:, 0, :] = oldest, seq[:, 59, :] = newest
        start = i * N_DAYS
        end = start + N_DAYS
        field_data = X_flat[:, start:end]  # (batch, 60) — 59=oldest first
        seq[:, :, i] = field_data  # already in oldest-first order
    return seq


class SeqLSTM(nn.Module):
    """LSTM on (batch, 60, 6) OHLCV sequences."""
    def __init__(self, n_fields=6, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(n_fields, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attn = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2),
                                  nn.Tanh(), nn.Linear(hidden_size // 2, 1))
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, 60, 6)
        out, _ = self.rnn(x)  # (batch, 60, hidden)
        attn_w = torch.softmax(self.attn(out), dim=1)  # (batch, 60, 1)
        ctx = (out * attn_w).sum(dim=1)  # (batch, hidden)
        return self.fc(ctx).squeeze(-1)


class SeqTransformer(nn.Module):
    """Transformer on (batch, 60, 6) OHLCV sequences."""
    def __init__(self, n_fields=6, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_fields, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, N_DAYS, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.proj(x) + self.pos_enc
        x = self.encoder(x)
        x = x[:, -1, :]  # last timestep
        return self.fc(x).squeeze(-1)


def train_seq_model(model, X_tr, y_tr, X_va, y_va, n_epochs=20, lr=1e-3,
                    batch_size=2048, early_stop=5):
    """Train a sequence model."""
    device = get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience = 0

    for epoch in range(n_epochs):
        model.train()
        indices = np.random.permutation(len(X_tr))
        total_loss = 0
        n_batches = 0

        for start in range(0, len(X_tr), batch_size):
            idx = indices[start:start + batch_size]
            xb = torch.from_numpy(X_tr[idx]).to(device)
            yb = torch.from_numpy(y_tr[idx]).to(device)

            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = []
            for start in range(0, len(X_va), batch_size):
                xb = torch.from_numpy(X_va[start:start + batch_size]).to(device)
                val_preds.append(model(xb).cpu().numpy())
            val_pred = np.concatenate(val_preds)
            val_loss = float(np.mean((val_pred - y_va) ** 2))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if (epoch + 1) % 5 == 0:
            logger.info(f"    Epoch {epoch+1}: train={total_loss/n_batches:.6f} val={val_loss:.6f}")

        if patience >= early_stop:
            logger.info(f"    Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model


def predict_seq(model, X, batch_size=4096):
    device = get_device()
    model = model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def evaluate_split(pred, label, index):
    mask = np.isfinite(pred) & np.isfinite(label)
    if mask.sum() < 200:
        return None
    pred_s = pd.Series(pred[mask], index=index[mask])
    label_s = pd.Series(label[mask], index=index[mask])
    ric_list, spreads = [], []
    for dt, g in pred_s.groupby(level=0):
        gl = label_s.reindex(g.index).dropna()
        common = g.index.intersection(gl.index)
        if len(common) < 40:
            continue
        p, l = g.loc[common].values, gl.loc[common].values
        ric = stats.spearmanr(p, l).statistic
        if np.isfinite(ric):
            ric_list.append(ric)
        tmp = pd.DataFrame({"p": p, "l": l}).sort_values("p", ascending=False)
        spreads.append(tmp.head(20)["l"].mean() - tmp.tail(20)["l"].mean())
    if not ric_list:
        return None
    return {
        "rank_ic": float(np.mean(ric_list)),
        "rank_ic_pos": float(np.mean([r > 0 for r in ric_list])),
        "spread_top20": float(np.mean(spreads)) if spreads else 0,
    }


def main():
    t_start = time.time()

    import gc
    logger.info("Loading Alpha360 cache (last 1 year, float32 to save memory)...")
    cache = pd.read_parquet(DATA_DIR / "feature_cache_alpha360.parquet")
    all_dates = cache.index.get_level_values(0)
    cutoff = all_dates.max() - pd.Timedelta(days=400)
    cache = cache.loc[all_dates >= cutoff]
    # Convert to float32 immediately to halve memory
    feature_cols = [c for c in cache.columns if not c.startswith("__")]
    for c in feature_cols:
        cache[c] = cache[c].astype(np.float32)
    gc.collect()
    logger.info(f"Cache: {cache.shape}, {len(feature_cols)} features")

    trade_dates = sorted(cache.index.get_level_values(0).unique())
    dates_level = cache.index.get_level_values(0)

    # Skip loading 174 cache to save memory (Alpha360 alone is 5.9GB)
    # XGB baseline runs on Alpha360 flat features instead

    MODELS = [
        {"name": "xgb_360_flat", "type": "xgb_360"},
        {"name": "lstm_360_seq", "type": "seq", "model_fn": lambda: SeqLSTM(N_FIELDS, 64, 2)},
        {"name": "transformer_360_seq", "type": "seq", "model_fn": lambda: SeqTransformer(N_FIELDS, 64, 4, 2)},
    ]

    all_results = {m["name"]: [] for m in MODELS}

    for split_idx in range(N_SPLITS):
        test_end_idx = len(trade_dates) - 1 - split_idx * TEST_DAYS
        test_start_idx = test_end_idx - TEST_DAYS
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - VALID_DAYS
        train_end_idx = valid_start_idx - 1
        train_start_idx = max(0, train_end_idx - TRAIN_DAYS)
        if train_start_idx >= train_end_idx:
            break

        test_start = trade_dates[test_start_idx]
        test_end = trade_dates[test_end_idx]
        logger.info(f"\nSplit {split_idx+1}/{N_SPLITS}: test {str(test_start)[:10]}~{str(test_end)[:10]}")

        train_mask = (dates_level >= trade_dates[train_start_idx]) & (dates_level <= trade_dates[train_end_idx])
        valid_mask = (dates_level >= trade_dates[valid_start_idx]) & (dates_level <= trade_dates[valid_end_idx])
        test_mask = (dates_level >= test_start) & (dates_level <= test_end)

        # Alpha360 data — load per segment to save memory
        import gc

        def load_segment(mask):
            X = cache.loc[mask, feature_cols].values.astype(np.float32)
            y = cache.loc[mask, LABEL_COL].values.astype(np.float32)
            m = np.isfinite(y)
            X, y = X[m], y[m]
            X = np.nan_to_num(X, nan=0.0)
            idx = cache.index[mask][m]
            return X, y, idx

        X360_tr, y_tr_c, _ = load_segment(train_mask)
        X360_va, y_va_c, _ = load_segment(valid_mask)
        X360_te_c, y_te_c, test_idx_c = load_segment(test_mask)

        # Reshape for sequence models (done lazily per model to save peak memory)
        X_seq_tr = reshape_to_sequence(X360_tr)
        X_seq_va = reshape_to_sequence(X360_va)
        X_seq_te = reshape_to_sequence(X360_te_c)
        gc.collect()

        for cfg in MODELS:
            name = cfg["name"]
            t1 = time.time()
            try:
                if cfg["type"] == "xgb_360":
                    import xgboost as xgb
                    XGB_PARAMS = {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
                                  "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
                                  "objective": "reg:squarederror", "nthread": 12, "verbosity": 0, "seed": 42}
                    dt = xgb.DMatrix(X360_tr, label=y_tr_c)
                    dv = xgb.DMatrix(X360_va, label=y_va_c)
                    model = xgb.train(XGB_PARAMS, dt, num_boost_round=400,
                                      evals=[(dv, "valid")], early_stopping_rounds=50, verbose_eval=0)
                    pred = model.predict(xgb.DMatrix(X360_te_c))

                elif cfg["type"] == "seq":
                    model = cfg["model_fn"]()
                    logger.info(f"  Training {name} on {get_device()} ({len(X_seq_tr)} samples)")
                    model = train_seq_model(model, X_seq_tr, y_tr_c, X_seq_va, y_va_c,
                                            n_epochs=20, batch_size=2048, early_stop=5)
                    pred = predict_seq(model, X_seq_te)

                elapsed = time.time() - t1
                metrics = evaluate_split(pred, y_te_c, test_idx_c)
                if metrics:
                    metrics["split"] = split_idx + 1
                    metrics["time"] = round(elapsed, 1)
                    all_results[name].append(metrics)
                    logger.info(f"  {name:<24} RankIC={metrics['rank_ic']:+.4f} "
                                f"Spread={metrics['spread_top20']*100:+.3f}% ({elapsed:.1f}s)")
            except Exception as e:
                logger.error(f"  {name}: FAILED: {e}")
                import traceback; traceback.print_exc()

    # Summary
    logger.info(f"\n{'='*90}")
    logger.info(f"PHASE 4M.2 ALPHA360 SEQUENCE MODEL COMPARISON: {N_SPLITS} splits")
    logger.info(f"{'='*90}")
    logger.info(f"{'Model':<26} {'AvgRIC':>8} {'MedRIC':>8} {'RICIR':>7} {'Spr20':>8} {'RIC>0':>6} {'Time':>7} {'#Spl':>5}")
    logger.info("-" * 90)

    summary = {}
    for cfg in MODELS:
        name = cfg["name"]
        splits = all_results[name]
        if not splits:
            continue
        rics = [s["rank_ic"] for s in splits]
        spreads = [s["spread_top20"] for s in splits]
        times = [s.get("time", 0) for s in splits]
        ric_pos = [s.get("rank_ic_pos", 0) for s in splits]
        ricir = float(np.mean(rics) / (np.std(rics) + 1e-8))
        summary[name] = {
            "avg_rank_ic": round(float(np.mean(rics)), 6),
            "med_rank_ic": round(float(np.median(rics)), 6),
            "rank_icir": round(ricir, 4),
            "avg_spread": round(float(np.mean(spreads)), 6),
            "avg_ric_pos": round(float(np.mean(ric_pos)), 4),
            "avg_time": round(float(np.mean(times)), 1),
            "n_splits": len(splits),
            "per_split": splits,
        }
        logger.info(f"{name:<26} {np.mean(rics):>+8.4f} {np.median(rics):>+8.4f} {ricir:>+7.3f} "
                    f"{np.mean(spreads)*100:>+7.3f}% {np.mean(ric_pos)*100:>5.0f}% "
                    f"{np.mean(times):>6.1f}s {len(splits):>5}")

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    output = {"evaluated_at": datetime.now().isoformat(timespec="seconds"),
              "n_splits": N_SPLITS, "summary": summary}
    out_path = OUTPUT_DIR / "alpha360_seq_comparison.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2,
                                   default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else str(o)))
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
