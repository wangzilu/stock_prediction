"""FeatureMerger: combine Alpha158 + fundamental + capital flow + macro + shareholder.

Merges multi-frequency features into a single matrix for XGB/LGB training.
- Alpha158: daily, from Qlib DatasetH
- Fundamental: weekly (forward-filled to daily)
- Capital flow: daily
- Macro: daily (broadcast to all stocks)
- Shareholder: quarterly (forward-filled to daily)

Usage:
    merger = FeatureMerger()
    X_merged, y = merger.merge(dataset)
    # X_merged has 158 + ~30 = ~188 columns
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "storage"


class FeatureMerger:

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

    def merge_for_training(self, dataset, segment: str = "train") -> tuple:
        """Merge Alpha158 features with supplementary factors.

        Args:
            dataset: Qlib DatasetH object
            segment: "train", "valid", or "test"

        Returns:
            (X_merged: np.ndarray, y: np.ndarray, index: pd.MultiIndex)
        """
        # 1. Alpha158 features from Qlib
        X_alpha = dataset.prepare(segment, col_set="feature")
        y_label = dataset.prepare(segment, col_set="label")
        if isinstance(y_label, pd.DataFrame):
            y_label = y_label.iloc[:, 0]

        logger.info(f"Alpha158: {X_alpha.shape}")

        # 2. Load supplementary features
        supp = self._load_supplementary(X_alpha.index)
        if supp is not None and not supp.empty:
            logger.info(f"Supplementary features: {supp.shape[1]} columns")
            X_merged = X_alpha.join(supp, how="left")
        else:
            logger.info("No supplementary features available")
            X_merged = X_alpha

        # 3. Convert to numpy (XGB handles NaN natively)
        X_np = X_merged.values.astype(np.float32)
        y_np = y_label.values.astype(np.float32)

        logger.info(f"Merged: {X_np.shape[1]} features, {len(X_np)} samples")
        return X_np, y_np, X_merged.index

    def _load_supplementary(self, index: pd.MultiIndex) -> pd.DataFrame:
        """Load and align all supplementary features to the Qlib index."""
        frames = []

        # Fundamental features
        fund = self._load_fundamental(index)
        if fund is not None:
            frames.append(fund)

        # Capital flow features
        flow = self._load_capital_flow(index)
        if flow is not None:
            frames.append(flow)

        # Macro features (broadcast)
        macro = self._load_macro(index)
        if macro is not None:
            frames.append(macro)

        # Shareholder features
        holder = self._load_shareholder(index)
        if holder is not None:
            frames.append(holder)

        if not frames:
            return None

        result = pd.concat(frames, axis=1)
        return result

    def _load_fundamental(self, index: pd.MultiIndex) -> pd.DataFrame:
        """Load fundamental features and align to Qlib MultiIndex."""
        path = self.data_dir / "fundamental_features.parquet"
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
            if df.empty or "qlib_code" not in df.columns:
                return None

            # Select numeric factor columns
            factor_cols = [c for c in ["pe_ttm", "pb", "ep", "bp", "log_mv", "log_circ_mv",
                                        "roe", "roa", "gross_margin", "net_margin",
                                        "debt_ratio", "revenue_growth", "profit_growth"]
                          if c in df.columns]
            if not factor_cols:
                return None

            df = df[["qlib_code"] + factor_cols].drop_duplicates("qlib_code")
            df = df.set_index("qlib_code")

            # Map to Qlib MultiIndex: (datetime, instrument)
            return self._align_stock_factors(df, index, factor_cols, prefix="fund")
        except Exception as e:
            logger.warning(f"Fundamental load failed: {e}")
            return None

    def _load_capital_flow(self, index: pd.MultiIndex) -> pd.DataFrame:
        """Load capital flow features."""
        path = self.data_dir / "capital_flow_features.parquet"
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
            if df.empty or "qlib_code" not in df.columns:
                return None

            factor_cols = [c for c in ["main_net_inflow", "main_net_pct", "super_large_net",
                                        "large_net", "medium_net", "small_net"]
                          if c in df.columns]
            if not factor_cols:
                return None

            df = df[["qlib_code"] + factor_cols].drop_duplicates("qlib_code")
            df = df.set_index("qlib_code")

            return self._align_stock_factors(df, index, factor_cols, prefix="flow")
        except Exception as e:
            logger.warning(f"Capital flow load failed: {e}")
            return None

    def _load_macro(self, index: pd.MultiIndex) -> pd.DataFrame:
        """Load macro features and broadcast to all stocks."""
        path = self.data_dir / "macro_features.parquet"
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
            if df.empty:
                return None

            factor_cols = [c for c in df.columns if c not in ("date", "collected_at")]
            if not factor_cols:
                return None

            # Get latest macro values
            latest = df.iloc[-1]
            macro_values = {f"macro_{c}": _safe_float(latest.get(c)) for c in factor_cols}

            # Broadcast: same values for all (datetime, instrument) pairs
            result = pd.DataFrame(
                {k: [v] * len(index) for k, v in macro_values.items()},
                index=index,
            )
            return result
        except Exception as e:
            logger.warning(f"Macro load failed: {e}")
            return None

    def _load_shareholder(self, index: pd.MultiIndex) -> pd.DataFrame:
        """Load shareholder features."""
        path = self.data_dir / "shareholder_features.parquet"
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
            if df.empty or "qlib_code" not in df.columns:
                return None

            factor_cols = [c for c in ["holder_count", "holder_count_change", "pledge_ratio"]
                          if c in df.columns]
            if not factor_cols:
                return None

            df = df[["qlib_code"] + factor_cols].drop_duplicates("qlib_code")
            df = df.set_index("qlib_code")

            return self._align_stock_factors(df, index, factor_cols, prefix="holder")
        except Exception as e:
            logger.warning(f"Shareholder load failed: {e}")
            return None

    def _align_stock_factors(
        self, factor_df: pd.DataFrame, index: pd.MultiIndex,
        factor_cols: list, prefix: str
    ) -> pd.DataFrame:
        """Align stock-level factors to Qlib's (datetime, instrument) MultiIndex.

        factor_df is indexed by qlib_code (e.g. SH600519).
        We broadcast the same values across all dates for each stock.
        """
        # Determine which level is instrument
        inst_level = 1 if index.nlevels > 1 else 0
        instruments = index.get_level_values(inst_level)

        # Map each instrument to its factor values
        result_data = {}
        for col in factor_cols:
            values = []
            for inst in instruments:
                # Normalize code format
                inst_str = str(inst).upper()
                if inst_str in factor_df.index:
                    values.append(factor_df.loc[inst_str, col])
                else:
                    values.append(np.nan)
            result_data[f"{prefix}_{col}"] = values

        return pd.DataFrame(result_data, index=index)


def _safe_float(val) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan
