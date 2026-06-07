"""FeatureMerger: combine Alpha158 + fundamental + capital flow + macro + shareholder.

Merges multi-frequency features into a single matrix for XGB/LGB training.
Includes cross-sectional preprocessing (winsorize, zscore, rank, missing flags).

Usage:
    merger = FeatureMerger()
    X_merged, y = merger.merge_for_training(dataset, "train")
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config.production_features import (
    PRODUCTION_SUPPLEMENTARY_GROUPS,
    RESEARCH_ALL_LOADERS,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "storage"


class FeatureMerger:

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

    def inject_qlib_custom_factors_into_handler(
        self,
        handler,
        factor_specs: tuple[tuple[str, str], ...] | None = None,
    ) -> int:
        """Inject Qlib expression-language factors into the handler.

        Used by the xgb_174 profile (and any future profile that needs
        compute-on-the-fly factors derived from $-prefixed Qlib fields).
        Pre-fix xgb_174 was inaccessible because FeatureMerger only
        knew about parquet-backed loaders; the expression factors lived
        in scripts/train_pit_baseline.py only.

        Args:
            handler: a Qlib DataHandlerLP with _data / _learn / _infer
                attributes (or any subset).
            factor_specs: tuple of ``(name, qlib_expr)`` pairs. When
                None, looks up the current profile's factors via
                ``config.production_features.current_profile_qlib_custom_factors``.

        Returns:
            Number of columns injected. Zero for profiles without
            qlib_custom factors (e.g. xgb_242).
        """
        if factor_specs is None:
            from config.production_features import current_profile_qlib_custom_factors
            factor_specs = current_profile_qlib_custom_factors()
        if not factor_specs:
            return 0
        try:
            from qlib.data import D
        except Exception as e:
            logger.error(
                "qlib not importable — cannot inject custom factors: %s", e,
            )
            raise

        # 2026-06-04 cx round 16 P1-1 fix: compute the union of all
        # handler frames' indexes so D.features covers every frame's
        # instrument × date span. Pre-fix the helper only used the
        # first non-empty frame as the base, then assigned ``.values``
        # to every frame — silently misaligning when _data / _learn /
        # _infer had different lengths or orders. Now we collect the
        # union, query D.features once, then reindex PER-FRAME using
        # ``.loc`` so each frame gets the correct values for its own
        # index (matches the supplementary path's correctness).
        all_indexes = []
        for attr in ("_data", "_learn", "_infer"):
            df = getattr(handler, attr, None)
            if df is not None and len(df) > 0:
                all_indexes.append(df.index)
        if not all_indexes:
            logger.warning(
                "inject_qlib_custom_factors: no _data/_learn/_infer found"
            )
            return 0
        union_index = all_indexes[0]
        for idx in all_indexes[1:]:
            union_index = union_index.union(idx)

        instruments = list(set(str(c) for c in union_index.get_level_values(1)))
        dates = sorted(union_index.get_level_values(0).unique())
        if not dates:
            return 0
        custom = D.features(
            instruments,
            [expr for _name, expr in factor_specs],
            start_time=str(min(dates))[:10],
            end_time=str(max(dates))[:10],
        )
        if custom is None or custom.empty:
            logger.warning("inject_qlib_custom_factors: D.features returned empty")
            return 0
        custom.columns = [name for name, _expr in factor_specs]
        # Swap levels so (datetime, instrument) order matches the handler's
        # MultiIndex. Replace inf with NaN so XGB does not crash.
        custom = custom.swaplevel().sort_index()
        custom = custom.replace([np.inf, -np.inf], np.nan)

        # Per-frame reindex + single concat (2026-06-05 fix). The old
        # per-column ``df.loc[attr_common, (...)] = ...`` loop ran
        # 13 cols × 3 frames = 39 ops on a wide MultiIndex frame,
        # which forces pandas BlockManager defragmentation. Same root
        # cause as inject_supplementary_into_handler; see that
        # function's comment for the wider context. This loop is
        # currently a no-op for xgb_242 (zero custom factors) but
        # xgb_174 needs the fast path too.
        custom_cols = list(custom.columns)
        for attr in ("_data", "_learn", "_infer"):
            df = getattr(handler, attr, None)
            if df is None or len(df) == 0:
                continue
            block = custom.reindex(df.index)
            block.columns = pd.MultiIndex.from_tuples(
                [("feature", c) for c in custom_cols]
            )
            new_df = pd.concat([df, block], axis=1)
            setattr(handler, attr, new_df)

        logger.info(
            "inject_qlib_custom_factors: %d custom expression cols injected "
            "(per-frame aligned)", len(factor_specs),
        )
        return len(factor_specs)

    def inject_supplementary_into_handler(
        self,
        handler,
        preprocess: bool = False,
        groups: tuple[str, ...] | str = PRODUCTION_SUPPLEMENTARY_GROUPS,
    ) -> int:
        """Inject supplementary features into a Qlib handler's internal
        _data / _learn / _infer frames.

        This is the SAME injection that `scripts/train_lgb.py` does at
        train time. The trained model expects to find these supp
        columns at inference, so `ShortTermModel.load_from_pickle`
        must call this method on the freshly-rebuilt inference dataset
        to avoid the 158 vs 242 column mismatch that caused the
        2026-06-03 22:00 all-negative-prediction incident.

        Args:
            handler: a Qlib DataHandlerLP with `_data` / `_learn` /
                `_infer` attributes (or any subset).
            preprocess: If True, apply `_preprocess_supplementary`
                (mode="rank") before injection. The CURRENT production
                model was trained with `preprocess=False` (raw values),
                so inference must also use False to match training
                distribution. When the train_lgb branch
                (`fix/train-lgb-use-feature-merger`) lands, both will
                switch to True simultaneously.
            groups: Explicit supplementary feature groups to inject. The
                default is the production contract; pass None only from
                research/shadow code that intentionally wants all loaders.

        Returns:
            Number of supplementary columns injected.
        """
        # The handler should have at least one of these frames
        for attr in ("_data", "_learn", "_infer"):
            df = getattr(handler, attr, None)
            if df is not None and len(df) > 0:
                base_index = df.index
                break
        else:
            logger.warning(
                "inject_supplementary_into_handler: no _data/_learn/_infer "
                "found on handler — nothing to inject"
            )
            return 0

        supp = self._load_supplementary(base_index, groups=groups)
        if supp is None or supp.empty:
            logger.warning(
                "inject_supplementary_into_handler: supplementary frame "
                "empty — handler unchanged (this WILL cause train/serve "
                "feature mismatch on a model trained with supp)"
            )
            return 0
        # Drop inf so XGB doesn't crash
        supp = supp.replace([np.inf, -np.inf], np.nan)

        if preprocess:
            supp = self._preprocess_supplementary(
                supp, base_index, mode="rank",
            )

        n_supp = supp.shape[1]
        # 2026-06-05: replace per-column ``df[(...)] = ...`` +
        # ``df.loc[idx, (...)] = ...`` loop with a single per-frame
        # ``concat``. The old loop ran 84 cols × 3 frames = 252
        # column-add operations against a wide MultiIndex DataFrame,
        # which forces pandas' BlockManager to defragment / consolidate
        # on nearly every iteration. Real-world cost on the production
        # 5195-stock × 19-day inference frame was **~890s** (see
        # ``scripts/probe_load_model_timing.py``), which was the root
        # cause of the morning_recommendation / sell_check /
        # brinson_attribution 25-min silent hangs. The single concat
        # path stays in vectorised pandas and finishes in seconds.
        supp_cols = list(supp.columns)
        for attr in ("_data", "_learn", "_infer"):
            df = getattr(handler, attr, None)
            if df is None or len(df) == 0:
                continue
            # Reindex the supp frame to the handler frame's exact index
            # so the per-frame .loc[attr_common, ...] semantics are
            # preserved: rows the handler has but supp doesn't get NaN.
            block = supp.reindex(df.index)
            block.columns = pd.MultiIndex.from_tuples(
                [("feature", c) for c in supp_cols]
            )
            new_df = pd.concat([df, block], axis=1)
            setattr(handler, attr, new_df)

        logger.info(
            "inject_supplementary_into_handler: %d supp cols injected "
            "into handler (preprocess=%s, groups=%s)",
            n_supp, preprocess, groups,
        )
        return n_supp

    def merge_for_training(self, dataset, segment: str = "train",
                           preprocess: str = "rank") -> tuple:
        """Merge Alpha158 features with supplementary factors.

        Args:
            dataset: Qlib DatasetH object
            segment: "train", "valid", or "test"
            preprocess: preprocessing mode for supplementary factors
                "raw"   - no preprocessing (original values)
                "zscore" - cross-sectional winsorize + zscore
                "rank"   - cross-sectional rank percentile (default, most robust)
                "both"   - append both zscore and rank versions

        Returns:
            (X_merged: np.ndarray, y: np.ndarray, index: pd.MultiIndex)
        """
        # 1. Alpha158 features from Qlib (already self-normalized)
        X_alpha = dataset.prepare(segment, col_set="feature")
        y_label = dataset.prepare(segment, col_set="label")
        if isinstance(y_label, pd.DataFrame):
            y_label = y_label.iloc[:, 0]

        logger.info(f"Alpha158: {X_alpha.shape}")

        # 2. Load supplementary features. ``merge_for_training`` is the
        # historical research API; preserve the "load everything" behavior
        # behind an EXPLICIT sentinel so production code stays gated.
        supp = self._load_supplementary(X_alpha.index,
                                        groups=RESEARCH_ALL_LOADERS)
        if supp is not None and not supp.empty:
            n_raw = supp.shape[1]

            # 3. Cross-sectional preprocessing
            supp_processed = self._preprocess_supplementary(
                supp, X_alpha.index, mode=preprocess)

            logger.info(f"Supplementary: {n_raw} raw → {supp_processed.shape[1]} after preprocess ({preprocess})")
            X_merged = X_alpha.join(supp_processed, how="left")
        else:
            logger.info("No supplementary features available")
            X_merged = X_alpha

        # 4. Convert to numpy (XGB handles NaN natively)
        X_np = X_merged.values.astype(np.float32)
        y_np = y_label.values.astype(np.float32)

        logger.info(f"Merged: {X_np.shape[1]} features, {len(X_np)} samples")
        return X_np, y_np, X_merged.index, list(X_merged.columns)

    def _preprocess_supplementary(self, supp: pd.DataFrame,
                                   index: pd.MultiIndex,
                                   mode: str = "rank") -> pd.DataFrame:
        """Cross-sectional preprocessing for supplementary factors.

        All operations are per-date (cross-sectional), not global.
        This prevents look-ahead bias from global normalization.

        Args:
            supp: DataFrame with supplementary factors, indexed by (datetime, instrument)
            index: Qlib MultiIndex
            mode: "raw", "zscore", "rank", "both", or "enhanced"
                  "enhanced" = time-series derivatives + neutralization + rank

        Returns:
            Preprocessed DataFrame
        """
        if mode == "raw":
            return supp

        date_level = 0

        # Replace inf with NaN first
        supp = supp.replace([np.inf, -np.inf], np.nan)

        # Enhanced mode: time-series derivatives + neutralization + rank
        if mode == "enhanced":
            return self._preprocess_enhanced(supp, index)

        result_frames = []

        if mode in ("zscore", "both"):
            zscore_df = supp.copy()
            for col in zscore_df.columns:
                grouped = zscore_df[col].groupby(level=date_level)
                # Winsorize: clip to 1-99 percentile per day
                low = grouped.transform(lambda x: x.quantile(0.01))
                high = grouped.transform(lambda x: x.quantile(0.99))
                zscore_df[col] = zscore_df[col].clip(lower=low, upper=high)
                # Zscore per day
                mean = grouped.transform("mean")
                std = grouped.transform("std")
                zscore_df[col] = (zscore_df[col] - mean) / (std + 1e-8)
                # Final clip to +-3
                zscore_df[col] = zscore_df[col].clip(-3, 3)

            if mode == "zscore":
                return zscore_df
            # For "both", rename and keep
            zscore_df = zscore_df.rename(columns={c: f"{c}_zs" for c in zscore_df.columns})
            result_frames.append(zscore_df)

        if mode in ("rank", "both"):
            rank_df = supp.copy()
            for col in rank_df.columns:
                # Rank percentile per day: 0~1
                rank_df[col] = rank_df[col].groupby(level=date_level).rank(pct=True)

            if mode == "rank":
                return rank_df
            rank_df = rank_df.rename(columns={c: f"{c}_rk" for c in rank_df.columns})
            result_frames.append(rank_df)

        if mode == "both":
            # Also add missing flags for low-coverage factors
            miss_df = supp.isna().astype(np.float32)
            # Only keep flags for columns with >5% missing
            miss_rate = miss_df.mean()
            miss_cols = miss_rate[miss_rate > 0.05].index.tolist()
            if miss_cols:
                miss_df = miss_df[miss_cols].rename(
                    columns={c: f"{c}_na" for c in miss_cols})
                result_frames.append(miss_df)
                logger.info(f"  Missing flags: {len(miss_cols)} columns (>5% NaN)")

            return pd.concat(result_frames, axis=1)

        return supp

    # ------ enhanced preprocessing: ts-derivatives + neutralize + rank ------

    def _preprocess_enhanced(self, supp: pd.DataFrame,
                             index: pd.MultiIndex) -> pd.DataFrame:
        """Enhanced preprocessing pipeline:
        1. Time-series derivatives per stock (5/20/60 day change, volatility)
        2. Market-cap neutralization (cross-sectional regression residual)
        3. Cross-sectional rank normalization

        Only derives time-series features for columns with sufficient non-NaN
        coverage (>30% of samples).
        """
        date_level = 0
        inst_level = 1 if index.nlevels > 1 else 0
        frames = []

        # --- 1. Raw rank (baseline) ---
        rank_df = supp.copy()
        for col in rank_df.columns:
            rank_df[col] = rank_df[col].groupby(level=date_level).rank(pct=True)
        rank_df = rank_df.rename(columns={c: f"{c}_rk" for c in rank_df.columns})
        frames.append(rank_df)

        # --- 2. Time-series derivatives ---
        ts_frames = self._compute_ts_derivatives(supp, index)
        if ts_frames is not None and not ts_frames.empty:
            # Rank-normalize the derivatives cross-sectionally
            for col in ts_frames.columns:
                ts_frames[col] = ts_frames[col].groupby(level=date_level).rank(pct=True)
            frames.append(ts_frames)
            logger.info(f"  TS derivatives: {ts_frames.shape[1]} new features")

        # --- 3. Market-cap neutralized rank ---
        neutral_df = self._neutralize_by_mcap(supp, index)
        if neutral_df is not None and not neutral_df.empty:
            frames.append(neutral_df)
            logger.info(f"  Neutralized: {neutral_df.shape[1]} features")

        result = pd.concat(frames, axis=1)
        logger.info(f"  Enhanced total: {result.shape[1]} features "
                    f"(from {supp.shape[1]} raw)")
        return result

    def _compute_ts_derivatives(self, supp: pd.DataFrame,
                                index: pd.MultiIndex) -> pd.DataFrame | None:
        """Compute per-stock time-series features: change rate and volatility.

        For each raw factor, compute:
          - {col}_chg5:  pct_change over 5 days
          - {col}_chg20: pct_change over 20 days
          - {col}_vol20: rolling 20-day std / rolling 20-day mean (CV)
        """
        inst_level = 1 if index.nlevels > 1 else 0

        # Only process columns with enough data
        coverage = supp.notna().mean()
        eligible = coverage[coverage > 0.30].index.tolist()
        if not eligible:
            return None

        pieces = []
        for col in eligible:
            s = supp[col]
            grouped = s.groupby(level=inst_level)

            # 5-day change rate
            chg5 = grouped.transform(lambda x: x.pct_change(5))
            chg5 = chg5.replace([np.inf, -np.inf], np.nan).clip(-5, 5)
            pieces.append(chg5.rename(f"{col}_chg5"))

            # 20-day change rate
            chg20 = grouped.transform(lambda x: x.pct_change(20))
            chg20 = chg20.replace([np.inf, -np.inf], np.nan).clip(-5, 5)
            pieces.append(chg20.rename(f"{col}_chg20"))

            # 20-day coefficient of variation (volatility proxy)
            roll_std = grouped.transform(lambda x: x.rolling(20, min_periods=5).std())
            roll_mean = grouped.transform(
                lambda x: x.rolling(20, min_periods=5).mean().abs() + 1e-8)
            vol20 = (roll_std / roll_mean).clip(0, 10)
            pieces.append(vol20.rename(f"{col}_vol20"))

        if not pieces:
            return None
        return pd.concat(pieces, axis=1)

    def _neutralize_by_mcap(self, supp: pd.DataFrame,
                            index: pd.MultiIndex) -> pd.DataFrame | None:
        """Cross-sectional market-cap + industry neutralization via OLS residuals.

        For each date, regress each factor on [log(market_cap), industry_dummies]
        and keep the residual.  This removes size and industry effects.

        Falls back to mcap-only if industry mapping is unavailable.
        """
        mcap = self._load_mcap_for_neutralize(index)
        if mcap is None:
            return None

        date_level = 0
        inst_level = 1 if index.nlevels > 1 else 0
        log_mcap = np.log1p(mcap.clip(lower=1))

        # Try to load industry dummies
        ind_dummies = self._load_industry_dummies(index)

        # Only neutralize columns with >30% coverage
        coverage = supp.notna().mean()
        eligible = coverage[coverage > 0.30].index.tolist()
        if not eligible:
            return None

        residuals = {}
        for col in eligible:
            vals = supp[col].copy()
            resid = vals.copy()
            resid[:] = np.nan

            for date, group_idx in vals.groupby(level=date_level).groups.items():
                y = vals.loc[group_idx]
                x_mcap = log_mcap.loc[group_idx]
                mask = y.notna() & x_mcap.notna()
                if mask.sum() < 30:
                    continue
                yv = y[mask].values
                # Build design matrix: intercept + log_mcap + industry dummies
                xv = x_mcap[mask].values.reshape(-1, 1)
                xm = np.column_stack([np.ones(len(xv)), xv])

                if ind_dummies is not None:
                    ind_sub = ind_dummies.loc[y[mask].index]
                    # Drop columns with zero variance in this date
                    ind_arr = ind_sub.values
                    nonzero = ind_arr.sum(axis=0) > 0
                    if nonzero.sum() > 1:
                        # Drop one dummy to avoid multicollinearity
                        xm = np.column_stack([xm, ind_arr[:, nonzero][:, 1:]])

                try:
                    beta, _, _, _ = np.linalg.lstsq(xm, yv, rcond=None)
                    r = yv - xm @ beta
                    resid.loc[y[mask].index] = r
                except np.linalg.LinAlgError:
                    continue

            resid = resid.groupby(level=date_level).rank(pct=True)
            residuals[f"{col}_neu"] = resid

        if not residuals:
            return None
        return pd.DataFrame(residuals, index=index)

    def _load_industry_dummies(self, index: pd.MultiIndex) -> pd.DataFrame | None:
        """Load industry mapping and return one-hot dummies aligned to index."""
        path = self.data_dir / "industry_mapping.parquet"
        if not path.exists():
            logger.info("  Industry mapping not found, skipping industry neutralization")
            return None

        try:
            df = pd.read_parquet(path)
            if df.empty or "qlib_code" not in df.columns or "industry" not in df.columns:
                return None

            # Build stock -> industry mapping
            ind_map = df.drop_duplicates("qlib_code").set_index("qlib_code")["industry"]

            inst_level = 1 if index.nlevels > 1 else 0
            instruments = index.get_level_values(inst_level).astype(str).str.lower()
            matched = instruments.map(ind_map).fillna("unknown")

            # One-hot encode
            dummies = pd.get_dummies(matched, prefix="ind", dtype=np.float32)
            dummies.index = index

            coverage = (matched != "unknown").mean()
            logger.info(f"  Industry dummies: {dummies.shape[1]} industries, "
                        f"coverage={coverage:.1%}")
            return dummies
        except Exception as e:
            logger.debug(f"Industry dummies load failed: {e}")
            return None

    def _load_mcap_for_neutralize(self, index: pd.MultiIndex) -> pd.Series | None:
        """Load market capitalization aligned to index for neutralization."""
        # Try ST daily_basic (has total_mv / circ_mv)
        path = self.data_dir / "st_daily_basic.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                mcap_col = None
                for c in ["st_circ_mv", "st_total_mv", "circ_mv", "total_mv"]:
                    if c in df.columns:
                        mcap_col = c
                        break
                if mcap_col and "qlib_code" in df.columns and "date" in df.columns:
                    df[mcap_col] = pd.to_numeric(df[mcap_col], errors="coerce")
                    ts = df[["qlib_code", "date", mcap_col]].copy()
                    merged = self._asof_merge_timeseries(ts, index, "date", [mcap_col])
                    if merged is not None:
                        s = merged[mcap_col]
                        if s.notna().mean() > 0.3:
                            logger.info(f"  MCap for neutralize: {mcap_col}, "
                                        f"coverage={s.notna().mean():.1%}")
                            return s
            except Exception as e:
                logger.debug(f"MCap load from st_daily_basic failed: {e}")

        # Fallback: try fundamental_features log_mv
        path2 = self.data_dir / "fundamental_features.parquet"
        if path2.exists():
            try:
                df = pd.read_parquet(path2)
                if "log_circ_mv" in df.columns and "qlib_code" in df.columns:
                    # Already log-transformed, exponentiate for consistency
                    df["_mv"] = np.exp(pd.to_numeric(df["log_circ_mv"], errors="coerce"))
                    result = self._align_pit_stock_factors(
                        df, index, ["_mv"], prefix="_mcap", label="MCap")
                    if result is not None:
                        s = result.iloc[:, 0]
                        if s.notna().mean() > 0.3:
                            return s
            except Exception as e:
                logger.debug(f"MCap load from fundamental failed: {e}")

        logger.info("  MCap data unavailable, skipping neutralization")
        return None

    def _load_supplementary(
        self,
        index: pd.MultiIndex,
        groups: tuple[str, ...] | str = PRODUCTION_SUPPLEMENTARY_GROUPS,
    ) -> pd.DataFrame:
        """Load and align supplementary features to the Qlib index.

        2026-06-04 P0-e: ``groups`` MUST be explicit. The historical
        ``groups=None → load every loader`` default was the暗道 that let
        every new FeatureMerger loader silently enter the production
        champion at the next retrain (see commit 95cd256, 2026-05-12,
        and the 6-3 22:00 incident). This default now points at the
        production contract; research code that legitimately wants the
        full loader fan-out must pass
        ``groups=config.production_features.RESEARCH_ALL_LOADERS``
        as an explicit opt-in.

        Args:
            index: Qlib MultiIndex.
            groups: Either a tuple of group names (e.g.
                ``PRODUCTION_SUPPLEMENTARY_GROUPS`` for the live
                champion, ``SHADOW_SUPPLEMENTARY_GROUPS`` for
                pre-promotion experiments) or the sentinel
                ``RESEARCH_ALL_LOADERS`` to bypass the contract.
                Passing ``None`` is rejected so silent fan-out cannot
                regress in.

        Raises:
            ValueError: when ``groups`` is None or an unknown type.
        """
        if groups is None:
            raise ValueError(
                "_load_supplementary: groups=None is rejected. "
                "Production code must pass "
                "config.production_features.PRODUCTION_SUPPLEMENTARY_GROUPS; "
                "research/shadow code must pass either "
                "SHADOW_SUPPLEMENTARY_GROUPS or the explicit "
                "RESEARCH_ALL_LOADERS sentinel. (P0-e contract gate.)"
            )
        if groups == RESEARCH_ALL_LOADERS:
            allowed = None  # legacy "load every loader" semantics
        elif isinstance(groups, (tuple, list, set)):
            allowed = set(groups)
        else:
            raise ValueError(
                f"_load_supplementary: groups must be a tuple of group "
                f"names or RESEARCH_ALL_LOADERS sentinel, got "
                f"{type(groups).__name__}={groups!r}"
            )
        frames = []

        # 2026-06-05: per-loader timing so we can see which loader
        # dominates _load_supplementary's wall-clock cost. Pre-fix the
        # 14.8-min hang in morning_recommendation was being chalked up
        # to "supplementary injection is slow" without a way to point at
        # which of the 11 loaders was actually heavy. Each loader gets a
        # one-line ``[supp-timing] name=NNNs cols=NN`` log.
        import time as _supp_time
        _t0_supp = _supp_time.time()

        def _run(name: str, fn):
            t = _supp_time.time()
            try:
                df = fn(index)
            except Exception:
                logger.exception("[supp-timing] %s RAISED", name)
                raise
            dt = _supp_time.time() - t
            ncols = 0 if df is None else int(df.shape[1])
            logger.info("[supp-timing] %s=%.2fs cols=%d", name, dt, ncols)
            return df

        # Fundamental features
        if allowed is None or "fundamental" in allowed:
            fund = _run("fundamental", self._load_fundamental)
            if fund is not None:
                frames.append(fund)

        # Capital flow features
        if allowed is None or "capital_flow" in allowed:
            flow = _run("capital_flow", self._load_capital_flow)
            if flow is not None:
                frames.append(flow)

        # Macro features. Production uses the current zero-baseline contract;
        # real macro values require a separate PIT-safe re-enable.
        if allowed is None or "macro_zero_baseline" in allowed:
            macro = _run("macro", self._load_macro)
            if macro is not None:
                frames.append(macro)

        # Shareholder features
        if allowed is None or "shareholder" in allowed:
            holder = _run("shareholder", self._load_shareholder)
            if holder is not None:
                frames.append(holder)

        # Valuation features (PE/PB/PS)
        if allowed is None or "valuation" in allowed:
            val = _run("valuation", self._load_valuation)
            if val is not None:
                frames.append(val)

        # Northbound holding features
        if allowed is None or "northbound" in allowed:
            nb = _run("northbound", self._load_northbound)
            if nb is not None:
                frames.append(nb)

        # Quality features (ROE/margins/growth)
        if allowed is None or "quality" in allowed:
            quality = _run("quality", self._load_quality)
            if quality is not None:
                frames.append(quality)

        # ST_CLIENT daily_basic (PE/PB/PS/turnover/mv - PIT-safe daily)
        if allowed is None or "st_daily_basic" in allowed:
            st_basic = _run("st_daily_basic", self._load_st_daily_basic)
            if st_basic is not None:
                frames.append(st_basic)

        # ST_CLIENT moneyflow (资金流 - PIT-safe daily)
        if allowed is None or "st_moneyflow" in allowed:
            st_mf = _run("st_moneyflow", self._load_st_moneyflow)
            if st_mf is not None:
                frames.append(st_mf)

        # ST_CLIENT holder number (股东户数 - PIT-safe quarterly)
        if allowed is None or "st_holder_number" in allowed:
            st_holder = _run("st_holder_number", self._load_st_holder_number)
            if st_holder is not None:
                frames.append(st_holder)

        # LLM event factors (shadow path before promotion via ablation)
        if allowed is None or "llm_event" in allowed:
            llm = _run("llm_event", self._load_llm_event_factors)
            if llm is not None:
                frames.append(llm)

        # Guba popularity factors (shadow path before promotion via ablation)
        if allowed is None or "guba" in allowed:
            guba = _run("guba", self._load_guba)
            if guba is not None:
                frames.append(guba)

        # Global supply chain alpha (rule-based) — shadow group
        if allowed is None or "global_chain" in allowed:
            gc = _run("global_chain", self._load_global_chain)
            if gc is not None:
                frames.append(gc)

        # Global supply chain alpha (LLM source) — Phase B.7 candidate
        if allowed is None or "global_chain_llm" in allowed:
            gcl = _run("global_chain_llm", self._load_global_chain_llm)
            if gcl is not None:
                frames.append(gcl)

        # PBC monetary policy liquidity (PE-1 step 3 output)
        if allowed is None or "pbc_liquidity" in allowed:
            pbc = _run("pbc_liquidity", self._load_pbc_liquidity)
            if pbc is not None:
                frames.append(pbc)

        # Cross-market regime signals (恒生/纳指 - broadcast to all stocks per date)
        if allowed is None or "cross_market_regime" in allowed:
            cross_mkt = _run("cross_market_regime", self._load_cross_market_regime)
            if cross_mkt is not None:
                frames.append(cross_mkt)

        logger.info(
            "[supp-timing] TOTAL _load_supplementary=%.2fs across %d frames",
            _supp_time.time() - _t0_supp, len(frames),
        )

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

            return self._align_pit_stock_factors(
                df, index, factor_cols, prefix="fund", label="Fundamental"
            )
        except Exception as e:
            logger.warning(f"Fundamental load failed: {e}")
            return None

    def _load_capital_flow(self, index: pd.MultiIndex) -> pd.DataFrame:
        """Load capital flow features from fund_flow_history or legacy parquet.

        Supports both:
        - fund_flow_history.parquet (ST_CLIENT format: net_mf_amount, buy_*/sell_*)
        - capital_flow_features.parquet (legacy AKShare format: main_net_inflow, etc.)
        """
        # Try new fund_flow_history first (daily time-series, richer data)
        hist_path = self.data_dir / "fund_flow_history.parquet"
        if hist_path.exists():
            try:
                df = pd.read_parquet(hist_path)
                if not df.empty and "qlib_code" in df.columns:
                    return self._load_capital_flow_from_history(df, index)
            except Exception as e:
                logger.warning(f"fund_flow_history load failed, trying legacy: {e}")

        # Fallback to legacy capital_flow_features.parquet
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

    def _load_capital_flow_from_history(self, df: pd.DataFrame, index: pd.MultiIndex) -> pd.DataFrame:
        """Load daily capital flow features from fund_flow_history.parquet.

        PIT-safe: computes rolling aggregates per (stock, date) so that each
        training sample only uses capital flow data available on or before that date.
        """
        # Normalize columns — handle both ST and AK formats
        if "net_mf_amount" not in df.columns and "主力净流入-净额" in df.columns:
            df["net_mf_amount"] = pd.to_numeric(df["主力净流入-净额"], errors="coerce")

        if "trade_date" not in df.columns and "日期" in df.columns:
            df["trade_date"] = pd.to_datetime(df["日期"], errors="coerce")
        else:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")

        needed = ["qlib_code", "trade_date", "net_mf_amount"]
        if not all(c in df.columns for c in needed):
            return None

        df = df[needed].dropna()
        df = df.sort_values(["qlib_code", "trade_date"]).reset_index(drop=True)

        # 2026-06-05 fix: replace ``groupby.transform(lambda x: ...)``
        # with vectorised ``groupby.rolling`` + the level-drop pattern.
        # Pre-fix this method took ~9.2 min on the production 4-month
        # 5195-stock fund_flow_history frame (probe_load_model_timing.py
        # 2026-06-05); ``transform(lambda)`` forces a per-stock Python
        # callback so the cost scales with stock-count × Python overhead
        # instead of stock-count × numpy. The first ``lambda x: x`` was
        # also literally a no-op — wasted entirely. Same column outputs,
        # just computed via the C-level path.
        df["flow_net_mf_latest"] = df["net_mf_amount"]
        rolling_grp = df.groupby("qlib_code", sort=False)["net_mf_amount"]
        df["flow_net_mf_5d"] = (
            rolling_grp.rolling(5, min_periods=1).sum()
            .reset_index(level=0, drop=True)
        )
        df["flow_net_mf_20d_avg"] = (
            rolling_grp.rolling(20, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )

        factor_cols = ["flow_net_mf_latest", "flow_net_mf_5d", "flow_net_mf_20d_avg"]
        flow_daily = df[["qlib_code", "trade_date"] + factor_cols].copy()

        # PIT safety: capital flow data is published after market close on day T,
        # so it should only be available for predictions on T+1.
        # Shifting trade_date forward by 1 business day enforces lag1 alignment.
        # (PIT audit confirmed flow_lag1 RankIC +0.043 > flow_lag0 +0.038)
        flow_daily["trade_date"] = flow_daily["trade_date"] + pd.tseries.offsets.BDay(1)

        # Asof merge with training index
        result = self._asof_merge_timeseries(flow_daily, index, "trade_date", factor_cols)
        if result is not None:
            logger.info(f"Capital flow features (PIT-safe): {result.notna().any(axis=1).sum()} "
                        f"non-null rows, {len(factor_cols)} factors")
        return result

    def _load_macro(self, index: pd.MultiIndex) -> pd.DataFrame:
        """Load macro features.

        DISABLED 2026-06-03 — macro features are dropped from training
        until daily as-of macro data is available.

        Reason (cx code review round 3 P1): the previous implementation
        loaded macro_features.parquet (currently a single-row snapshot of
        the LATEST values), took df.iloc[-1], and broadcast that snapshot
        to every historical (date, stock) row in the training index. Every
        training row therefore saw the LATEST macro values, not the macro
        values that were actually known at that row's prediction time.
        That is look-ahead bias — the training data leaked future state.

        config/data_availability.py already tagged this source as
        pit_safe_level="unsafe". The previous "impact is small because
        macro changes slowly" rationalisation is not acceptable for a
        training input: any consistent broadcast of future values can be
        learned by the model as a spurious shortcut.

        To re-enable safely a future PR must:
          1. Produce macro_features.parquet as a daily time series with
             an `available_date` column (T+1 publication conservatism).
          2. Replace the broadcast with an asof merge on available_date
             vs the training index trade_date.
          3. Re-add a PIT audit test that asserts each training row's
             macro_* values are drawn from on-or-before `available_date`.

        Until then this method returns None, the call site in
        merge_for_training treats None as "no macro frame", and no
        macro_* columns are present in the training/cache features.
        """
        # 2026-06-04 final: DIMENSION-PRESERVING zero baseline.
        # The trained XGBModel expects 242 features = Alpha158(158) +
        # supp(84) where supp includes 10 macro_* columns. Returning
        # None drops those 10 cols, so inference matrix is only 232
        # cols → XGB walks default-direction leaves → predictions
        # cluster around a single value → 22:00 evening_outlook
        # produces 0 stock recommendations.
        #
        # Returning a ZEROS frame with the trained column names is
        # PIT-safe (no future information leaks via zeros) AND
        # dimension-preserving (model sees the 10 columns it expects,
        # just always as 0). Re-enable contract for real macro values
        # documented below remains in force.
        path = self.data_dir / "macro_features.parquet"
        if not path.exists():
            # No parquet → no column-name template → cannot synthesise.
            return None
        try:
            df_template = pd.read_parquet(path)
        except Exception as e:  # noqa: BLE001
            logger.warning("macro template parquet read failed: %s", e)
            return None
        factor_cols = [c for c in df_template.columns
                       if c not in ("date", "collected_at")]
        if not factor_cols:
            return None

        # Single warn-once per session
        if not getattr(FeatureMerger, "_macro_drop_warned", False):
            logger.warning(
                "macro features DIMENSION-PRESERVED as zero baseline (PIT "
                "look-ahead drop, but column shape kept so inference does "
                "not silently bias). See _load_macro contract for re-enable."
            )
            FeatureMerger._macro_drop_warned = True

        macro_cols = [f"macro_{c}" for c in factor_cols]
        return pd.DataFrame(
            np.zeros((len(index), len(macro_cols)), dtype=np.float32),
            index=index,
            columns=macro_cols,
        )

    def _load_shareholder(self, index: pd.MultiIndex) -> pd.DataFrame:
        """Load shareholder features."""
        path = self.data_dir / "shareholder_features.parquet"
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
            if df.empty or "qlib_code" not in df.columns:
                return None

            factor_cols = [c for c in ["holder_count", "holder_count_change", "pledge_ratio",
                                        "total_share", "liquid_share", "liquid_ratio"]
                          if c in df.columns]
            if not factor_cols:
                return None

            return self._align_pit_stock_factors(
                df, index, factor_cols, prefix="holder", label="Shareholder"
            )
        except Exception as e:
            logger.warning(f"Shareholder load failed: {e}")
            return None

    def _load_northbound(self, index: pd.MultiIndex) -> pd.DataFrame:
        """Load northbound holding features from northbound_history.parquet.

        PIT-safe: computes rolling features per (stock, date) so that each
        training sample only uses northbound data available on or before that date.
        """
        path = self.data_dir / "northbound_history.parquet"
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
            if df.empty or "qlib_code" not in df.columns:
                return None

            # Detect holding-amount column
            hold_col = None
            for c in ["vol", "持股数量"]:
                if c in df.columns and df[c].notna().sum() > 1000:
                    hold_col = c
                    break
            ratio_col = None
            for c in ["ratio", "持股数量占A股百分比"]:
                if c in df.columns and df[c].notna().sum() > 1000:
                    ratio_col = c
                    break

            if hold_col is None and ratio_col is None:
                return None

            # Parse dates
            if "trade_date" not in df.columns and "持股日期" in df.columns:
                df["trade_date"] = pd.to_datetime(df["持股日期"], errors="coerce")
            else:
                df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
            df = df.dropna(subset=["trade_date"])
            df = df.sort_values(["qlib_code", "trade_date"])

            # Compute rolling features per stock per date (PIT-safe)
            factor_cols = []
            if hold_col:
                df["_hold"] = pd.to_numeric(df[hold_col], errors="coerce")
                grouped = df.groupby("qlib_code")["_hold"]
                df["nb_hold_change_5d"] = grouped.transform(lambda x: x.diff(5))
                df["nb_hold_change_20d"] = grouped.transform(lambda x: x.diff(20))
                factor_cols += ["nb_hold_change_5d", "nb_hold_change_20d"]

            if ratio_col:
                df["_ratio"] = pd.to_numeric(df[ratio_col], errors="coerce")
                df["nb_hold_ratio"] = df["_ratio"]
                grouped_r = df.groupby("qlib_code")["_ratio"]
                df["nb_ratio_change_5d"] = grouped_r.transform(lambda x: x.diff(5))
                factor_cols += ["nb_hold_ratio", "nb_ratio_change_5d"]

            if not factor_cols:
                return None

            nb_daily = df[["qlib_code", "trade_date"] + factor_cols].copy()
            # PIT safety: northbound data is published after market close on day T,
            # so it should only be available for predictions on T+1.
            nb_daily["trade_date"] = pd.to_datetime(nb_daily["trade_date"])
            nb_daily["trade_date"] = nb_daily["trade_date"] + pd.tseries.offsets.BDay(1)

            result = self._asof_merge_timeseries(nb_daily, index, "trade_date", factor_cols)
            if result is not None:
                logger.info(f"Northbound features (PIT-safe, T+1 lag): {len(factor_cols)} factors")
            return result
        except Exception as e:
            logger.warning(f"Northbound load failed: {e}")
            return None

    def _load_quality(self, index: pd.MultiIndex) -> pd.DataFrame:
        """Load quality features (ROE/margins/growth) from fundamental_quality.parquet."""
        path = self.data_dir / "fundamental_quality.parquet"
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
            if df.empty or "qlib_code" not in df.columns:
                return None

            factor_cols = [c for c in ["roe", "net_margin", "gross_margin", "eps_ttm",
                                        "asset_turnover", "equity_multiplier",
                                        "yoy_net_profit", "yoy_revenue"]
                          if c in df.columns]
            if not factor_cols:
                return None

            return self._align_pit_stock_factors(
                df, index, factor_cols, prefix="qual", label="Quality"
            )
        except Exception as e:
            logger.warning(f"Quality load failed: {e}")
            return None

    def _load_valuation(self, index: pd.MultiIndex) -> pd.DataFrame:
        """Load PE/PB/PS valuation features from fundamental_valuation.parquet.

        PIT-safe: uses asof merge so each training date only sees valuation
        data available on or before that date.
        """
        path = self.data_dir / "fundamental_valuation.parquet"
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
            if df.empty or "qlib_code" not in df.columns:
                return None

            factor_cols = [c for c in ["pe_ttm", "pb_mrq", "ps_ttm", "ep", "bp", "sp",
                                        "pcf_ncf_ttm"]
                          if c in df.columns]
            if not factor_cols:
                return None

            # Rename columns with prefix for output
            out_cols = [f"val_{c}" for c in factor_cols]
            df_renamed = df[["qlib_code", "date"] + factor_cols].copy()
            df_renamed.columns = ["qlib_code", "date"] + out_cols

            logger.info(f"Valuation features (PIT-safe): {df['qlib_code'].nunique()} stocks, "
                        f"{len(factor_cols)} factors")
            return self._asof_merge_timeseries(df_renamed, index, "date", out_cols)
        except Exception as e:
            logger.warning(f"Valuation load failed: {e}")
            return None

    def _asof_merge_timeseries(
        self, ts_df: pd.DataFrame, index: pd.MultiIndex,
        date_col: str, factor_cols: list
    ) -> pd.DataFrame:
        """PIT-safe asof merge: for each (date, instrument) in the training index,
        look up the most recent row in ts_df where ts_df.date <= training_date.

        ts_df must have columns: qlib_code, date_col, and factor_cols.
        Vectorized: uses np.searchsorted per stock instead of pd.merge_asof.
        """
        if ts_df.empty:
            return None

        inst_level = 1 if index.nlevels > 1 else 0
        date_level = 0

        ts_df = ts_df.copy()
        ts_df["qlib_code"] = ts_df["qlib_code"].astype(str).str.upper()
        ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
        ts_df = ts_df.dropna(subset=[date_col])

        # Ensure numeric
        for col in factor_cols:
            ts_df[col] = pd.to_numeric(ts_df[col], errors="coerce")

        # Sort and deduplicate: keep last per (stock, date)
        ts_df = ts_df.sort_values(["qlib_code", date_col])
        ts_df = ts_df.drop_duplicates(["qlib_code", date_col], keep="last")

        train_dates = pd.to_datetime(index.get_level_values(date_level))
        train_insts = index.get_level_values(inst_level).astype(str).str.upper()

        n = len(index)
        result_arrays = {col: np.full(n, np.nan, dtype=np.float64) for col in factor_cols}

        # Group training index by stock
        train_inst_arr = train_insts.values if hasattr(train_insts, 'values') else np.array(train_insts)
        train_date_arr = train_dates.values

        # 2026-06-05 fix: pre-build a {stock: (sorted_dates, values)}
        # dict via ONE groupby pass. Pre-fix the inner loop did
        # ``ts_df["qlib_code"] == stock`` for every stock — a full O(N)
        # boolean scan over the whole timeseries frame per stock,
        # i.e. O(stocks × ts_rows). On st_moneyflow's 4-month frame
        # (18 cols × ~500K rows × 5197 stocks) that produced the 208s
        # cost flagged by ``probe_load_model_timing.py`` 2026-06-05.
        # One groupby + dict lookup turns it into O(stocks) + O(N).
        ts_groups = {
            stock: (g[date_col].values, g[factor_cols].values)
            for stock, g in ts_df.groupby("qlib_code", sort=False)
        }

        # Build stock→positions mapping (vectorized with pandas groupby)
        inst_series = pd.Series(np.arange(n), index=train_inst_arr)
        inst_groups = inst_series.groupby(inst_series.index)

        processed = 0
        for stock, pos_idx in inst_groups:
            stock_block = ts_groups.get(stock)
            if stock_block is None:
                continue
            ts_dates, ts_values = stock_block
            if len(ts_dates) == 0:
                continue

            positions = pos_idx.values  # numpy array of row positions
            query_dates = train_date_arr[positions]

            # searchsorted: find insertion points (rightmost position where date <= query)
            insert_idx = np.searchsorted(ts_dates, query_dates, side="right") - 1

            # Valid: insert_idx >= 0 means we found a date <= query
            valid = insert_idx >= 0

            if valid.any():
                valid_positions = positions[valid]
                valid_idx = insert_idx[valid]

                for j, col in enumerate(factor_cols):
                    vals = ts_values[valid_idx, j]
                    result_arrays[col][valid_positions] = vals

            processed += 1

        logger.info(f"  asof_merge: {processed} stocks matched")
        return pd.DataFrame(result_arrays, index=index)

    def _align_pit_stock_factors(
        self,
        df: pd.DataFrame,
        index: pd.MultiIndex,
        factor_cols: list,
        prefix: str,
        label: str,
    ) -> pd.DataFrame | None:
        """Align stock-level factors using point-in-time effective dates.

        Snapshot financial data must not be broadcast backward through history.
        If a source does not provide any usable date, skip it rather than leak the
        latest snapshot into old training samples.
        """
        if df.empty or "qlib_code" not in df.columns:
            return None

        effective_date = self._effective_date_from_frame(df)
        if effective_date is None or effective_date.notna().sum() == 0:
            logger.warning(
                f"{label} skipped: no PIT date column found; refusing latest-snapshot broadcast"
            )
            return None

        out_cols = [f"{prefix}_{c}" for c in factor_cols]
        pit = df[["qlib_code"] + factor_cols].copy()
        pit["_effective_date"] = effective_date
        pit = pit.dropna(subset=["_effective_date"])
        if pit.empty:
            return None

        for col in factor_cols:
            pit[col] = pd.to_numeric(pit[col], errors="coerce")

        pit = pit.sort_values(["qlib_code", "_effective_date"])
        pit = pit.drop_duplicates(["qlib_code", "_effective_date"], keep="last")
        pit = pit.rename(columns={c: f"{prefix}_{c}" for c in factor_cols})

        logger.info(
            f"{label} features (PIT-safe): {pit['qlib_code'].nunique()} stocks, "
            f"{len(factor_cols)} factors"
        )
        return self._asof_merge_timeseries(
            pit[["qlib_code", "_effective_date"] + out_cols],
            index,
            "_effective_date",
            out_cols,
        )

    def _effective_date_from_frame(self, df: pd.DataFrame) -> pd.Series | None:
        """Return the first usable PIT date for a stock-level factor frame.

        Priority:
        1. ann_date / f_ann_date / publish_date: actual announcement date + 1 BDay
           (data becomes available the trading day after announcement)
        2. effective_date / trade_date / date: already represents availability
        3. report period end_date: conservative statutory delay fallback
        """
        # Tier 1: announcement dates — add 1 BDay for availability lag
        announce_cols = ["ann_date", "f_ann_date", "publish_date", "disclosure_date"]
        for col in announce_cols:
            if col in df.columns:
                parsed = _parse_date_series(df[col])
                if parsed.notna().any():
                    # Available next trading day after announcement
                    return parsed + pd.tseries.offsets.BDay(1)

        # Tier 2: already-available dates (no lag needed)
        avail_cols = ["effective_date", "pub_date", "trade_date", "date", "collected_at"]
        for col in avail_cols:
            if col in df.columns:
                parsed = _parse_date_series(df[col])
                if parsed.notna().any():
                    return parsed

        # Tier 3: report period fallback — conservative statutory deadlines
        # Only used when no announcement date exists at all
        report_cols = ["stat_date", "period", "report_period", "end_date"]
        for col in report_cols:
            if col in df.columns:
                parsed = _parse_date_series(df[col])
                if parsed.notna().any():
                    return _report_effective_date(parsed)
        return None

    def _align_stock_factors(
        self, factor_df: pd.DataFrame, index: pd.MultiIndex,
        factor_cols: list, prefix: str
    ) -> pd.DataFrame:
        """Align stock-level factors to Qlib's (datetime, instrument) MultiIndex.

        Only use this for truly static metadata. Time-varying stock factors
        should use _align_pit_stock_factors or _asof_merge_timeseries.
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


    def _load_st_daily_basic(self, index: pd.MultiIndex) -> pd.DataFrame:
        """Load ST_CLIENT daily_basic (PE/PB/PS/turnover/mv) - PIT-safe daily time-series."""
        path = self.data_dir / "st_daily_basic.parquet"
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
            if df.empty or "qlib_code" not in df.columns or "date" not in df.columns:
                return None

            factor_cols = [c for c in df.columns
                          if c.startswith("st_") and c not in ("st_code",)]
            if not factor_cols:
                return None

            for c in factor_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            daily = df[["qlib_code", "date"] + factor_cols].copy()
            # PIT safety: daily_basic is published after market close on day T,
            # so it should only be available for predictions on T+1.
            daily["date"] = pd.to_datetime(daily["date"])
            daily["date"] = daily["date"] + pd.tseries.offsets.BDay(1)

            result = self._asof_merge_timeseries(daily, index, "date", factor_cols)
            if result is not None:
                logger.info(f"ST daily_basic (PIT-safe, T+1 lag): {len(factor_cols)} factors")
            return result
        except Exception as e:
            logger.warning(f"ST daily_basic load failed: {e}")
            return None

    def _load_st_moneyflow(self, index: pd.MultiIndex) -> pd.DataFrame:
        """Load ST_CLIENT moneyflow (资金流) - PIT-safe daily time-series."""
        path = self.data_dir / "st_moneyflow.parquet"
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
            if df.empty or "qlib_code" not in df.columns or "date" not in df.columns:
                return None

            factor_cols = [c for c in df.columns if c.startswith("st_")]
            if not factor_cols:
                return None

            for c in factor_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            mf_daily = df[["qlib_code", "date"] + factor_cols].copy()

            # PIT safety: moneyflow data is published after market close on day T,
            # so it should only be available for predictions on T+1.
            mf_daily["date"] = pd.to_datetime(mf_daily["date"])
            mf_daily["date"] = mf_daily["date"] + pd.tseries.offsets.BDay(1)

            result = self._asof_merge_timeseries(
                mf_daily, index, "date", factor_cols)
            if result is not None:
                logger.info(f"ST moneyflow (PIT-safe): {len(factor_cols)} factors")
            return result
        except Exception as e:
            logger.warning(f"ST moneyflow load failed: {e}")
            return None

    def _load_st_holder_number(self, index: pd.MultiIndex) -> pd.DataFrame:
        """Load ST_CLIENT holder number (股东户数) - PIT-safe quarterly via ann_date."""
        path = self.data_dir / "st_holder_number.parquet"
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
            if df.empty or "qlib_code" not in df.columns:
                return None

            df["holder_num"] = pd.to_numeric(df.get("holder_num"), errors="coerce")
            df["ann_date"] = pd.to_datetime(df["ann_date"], format="%Y%m%d", errors="coerce")
            df = df.dropna(subset=["ann_date", "holder_num"])

            ts = df[["qlib_code", "ann_date", "holder_num"]].copy()
            ts = ts.sort_values(["qlib_code", "ann_date"]).drop_duplicates(
                ["qlib_code", "ann_date"], keep="last")

            # PIT safety: ann_date may be post-market; data available next BDay
            ts["ann_date"] = ts["ann_date"] + pd.tseries.offsets.BDay(1)

            result = self._asof_merge_timeseries(ts, index, "ann_date", ["holder_num"])
            if result is not None:
                logger.info(f"ST holder_number (PIT-safe, ann_date+1BDay): 1 factor, "
                            f"{result['holder_num'].notna().sum()} non-null")
            return result
        except Exception as e:
            logger.warning(f"ST holder_number load failed: {e}")
            return None

    def _load_global_chain(self, index: pd.MultiIndex,
                            llm: bool = False) -> pd.DataFrame | None:
        """Load global supply chain alpha factors.

        2026-06-07 (Phase B.7): added so xgb_209_chain / xgb_209_chain_llm
        candidate profiles can ablate via the standard --drop-group path.
        ``llm=False`` reads the rule-based parquet (production cron).
        ``llm=True`` reads the LLM-extracted parquet (B.7 ablation).
        Both share the same downstream loader so only the source events
        differ — the propagation, decay and shrink are identical.

        Drops the non-numeric ``level`` column before reindex.
        """
        name = "global_chain_factors_llm.parquet" if llm else "global_chain_factors.parquet"
        path = self.data_dir / name
        if not path.exists():
            logger.warning("Chain factors parquet not found: %s", path)
            return None
        try:
            df = pd.read_parquet(path)
            if df.empty:
                return None
            if not isinstance(df.index, pd.MultiIndex):
                logger.warning("Chain parquet has unexpected index: %s",
                               df.index.names)
                return None
            factor_cols = [
                c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
            ]
            if not factor_cols:
                return None
            result = df[factor_cols].reindex(index)
            non_null = result.notna().any(axis=1).sum()
            logger.info("Chain factors (%s): %d cols, %d rows non-null",
                        "llm" if llm else "rule", len(factor_cols), non_null)
            return result
        except Exception as e:
            logger.warning(f"Chain factors load failed: {e}")
            return None

    def _load_global_chain_llm(self, index: pd.MultiIndex) -> pd.DataFrame | None:
        """LLM source thin wrapper around _load_global_chain."""
        return self._load_global_chain(index, llm=True)

    def _load_pbc_liquidity(self, index: pd.MultiIndex) -> pd.DataFrame | None:
        """Load PBC monetary policy liquidity factors (market-level).

        2026-06-07 (Phase E.1 wiring): PE-1 step 3 produces
        ``pbc_liquidity_factors.parquet`` keyed by (datetime, instrument)
        with instrument always == "MARKET" (the synthetic broadcast key
        chain used by cross_market_regime). Cols:
        ``pbc_liquidity_zscore_20d`` / ``pbc_easing_dummy`` /
        ``pbc_tightening_dummy`` / ``short_rate_pressure``.

        Broadcast logic: each MARKET row is replicated across every
        instrument on the same date (same approach as
        ``_load_cross_market_regime``). The original index is preserved
        so unobserved dates get NaN → zero-fill downstream.
        """
        path = self.data_dir / "pbc_liquidity_factors.parquet"
        if not path.exists():
            logger.warning("PBC liquidity parquet not found: %s", path)
            return None
        try:
            df = pd.read_parquet(path)
            if df.empty:
                return None
            # Normalize to (datetime, instrument) MultiIndex regardless
            # of how the source wrote it.
            if isinstance(df.index, pd.MultiIndex):
                base = df.copy()
            elif {"datetime", "instrument"}.issubset(df.columns):
                base = df.set_index(["datetime", "instrument"])
            else:
                logger.warning("PBC parquet has unexpected shape: %s",
                               df.columns.tolist())
                return None
            factor_cols = [c for c in base.columns
                           if pd.api.types.is_numeric_dtype(base[c])]
            if not factor_cols:
                return None
            # Reduce to date-keyed frame (MARKET instrument only).
            base = base.reset_index()
            base["datetime"] = pd.to_datetime(base["datetime"])
            # Use the first MARKET row per date (build script guarantees one).
            market_df = (base[base["instrument"] == "MARKET"]
                         .drop_duplicates("datetime")
                         .set_index("datetime")[factor_cols])
            # Broadcast: align by caller's datetime level, regardless of stock.
            caller_dates = index.get_level_values("datetime")
            aligned = market_df.reindex(caller_dates)
            aligned.index = index
            non_null = aligned.notna().any(axis=1).sum()
            logger.info("PBC liquidity: %d cols, %d rows non-null",
                        len(factor_cols), non_null)
            return aligned
        except Exception as e:
            logger.warning(f"PBC liquidity load failed: {e}")
            return None

    def _load_guba(self, index: pd.MultiIndex) -> pd.DataFrame | None:
        """Load Eastmoney Guba popularity factors.

        2026-06-06 (P1 #4): collect_guba_sentiment.py produces a clean
        (datetime, instrument) parquet with 3 cols
        (popularity_rank / rank_change / popularity_score) but no
        FeatureMerger loader existed, so the data sat on disk and never
        reached production. This loader wires it. Inclusion in any
        profile is gated by ``config.production_features`` — the
        candidate profile xgb_209_guba uses it; the default xgb_209
        does NOT, pending ablation evidence.
        """
        path = self.data_dir / "guba_factors.parquet"
        if not path.exists():
            logger.warning("Guba factors parquet not found: %s", path)
            return None
        try:
            df = pd.read_parquet(path)
            if df.empty:
                logger.warning("Guba factors empty")
                return None
            # Source already has the right MultiIndex. Reindex onto the
            # caller's index; missing rows get NaN (downstream NaN
            # handler zeros them out).
            if not isinstance(df.index, pd.MultiIndex):
                logger.warning("Guba parquet has unexpected index: %s",
                               df.index.names)
                return None
            factor_cols = [
                c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
            ]
            if not factor_cols:
                return None
            result = df[factor_cols].reindex(index)
            non_null = result.notna().any(axis=1).sum()
            logger.info("Guba factors: %d cols, %d rows non-null",
                        len(factor_cols), non_null)
            return result
        except Exception as e:
            logger.warning(f"Guba factors load failed: {e}")
            return None

    def _load_llm_event_factors(self, index: pd.MultiIndex) -> pd.DataFrame | None:
        """Load LLM event factors from ``llm_event_factors.parquet``.

        2026-06-06 — added to close the loop the project lead flagged:
        scripts/build_llm_event_factors.py was writing the parquet but
        no FeatureMerger loader existed, so the LLM factors stayed as
        shadow signal that never reached the production model. This
        loader is the wiring; whether the group is included in a given
        profile is decided in ``config.production_features``
        (``xgb_209_llm`` profile, not the default ``xgb_209``).

        The current schema is the 5-column legacy:
            llm_impact_1d_decayed, llm_impact_5d_decayed,
            llm_sentiment_score, llm_event_count_5d, llm_avg_confidence
        After the L1 fact-count factors land in a fresh rebuild
        (llm_positive_event_count_3d / llm_negative_event_count_3d /
        llm_price_sensitive_count_3d / llm_official_event_count_3d /
        llm_event_count_3d / llm_repeated_ratio_3d /
        llm_event_intensity), those will be picked up automatically
        because this loader takes every non-key float column.
        """
        path = self.data_dir / "llm_event_factors.parquet"
        if not path.exists():
            logger.warning("LLM event factors parquet not found: %s", path)
            return None
        try:
            df = pd.read_parquet(path)
            if df.empty or "qlib_code" not in df.columns \
                    or "signal_date" not in df.columns:
                logger.warning("LLM event factors empty or missing keys")
                return None

            df["signal_date"] = pd.to_datetime(df["signal_date"])
            # Take every numeric factor column, drop keys
            factor_cols = [
                c for c in df.columns
                if c not in ("qlib_code", "signal_date")
                and pd.api.types.is_numeric_dtype(df[c])
            ]
            if not factor_cols:
                logger.warning("LLM event factors: no numeric factor columns")
                return None

            df = df.rename(columns={
                "qlib_code": "instrument",
                "signal_date": "datetime",
            })
            df = df.set_index(["datetime", "instrument"])
            df = df[factor_cols]
            # Reindex to caller's MultiIndex; rows that don't match get NaN.
            # The model's NaN-handling layer downstream zeros them out.
            result = df.reindex(index)
            non_null = result.notna().any(axis=1).sum()
            logger.info(
                "LLM event factors: %d cols, %d rows non-null",
                len(factor_cols), non_null,
            )
            return result
        except Exception as e:
            logger.warning(f"LLM event factors load failed: {e}")
            return None

    def _load_cross_market_regime(self, index: pd.MultiIndex) -> pd.DataFrame | None:
        """Load cross-market regime signals (恒生/纳指) and broadcast to all stocks.

        These are market-level signals (same value for all stocks on a given date).
        恒生/恒生科技 are leading indicators for A-share (faster price discovery).
        纳斯达克 tech themes propagate to A-share with 1-3 day delay.
        """
        path = self.data_dir / "cross_market_indices.parquet"
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
            if df.empty or "date" not in df.columns:
                return None

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
            # Normalize to ns precision to match Qlib index
            df["date"] = df["date"].dt.as_unit("ns")

            factor_cols = [c for c in df.columns if c != "date" and df[c].notna().sum() > 10]
            if not factor_cols:
                return None

            for c in factor_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            # Broadcast to all stocks: merge by date only (vectorized)
            date_level = 0
            train_dates = pd.to_datetime(index.get_level_values(date_level)).as_unit("ns")

            # Build date-only lookup: unique dates → regime values
            right = df[["date"] + factor_cols].drop_duplicates("date").sort_values("date")
            unique_dates = pd.DataFrame({"date": train_dates.unique()}).sort_values("date")
            date_map = pd.merge_asof(unique_dates, right, on="date", direction="backward")
            date_map = date_map.set_index("date")

            # Vectorized broadcast: map each training date to its regime values
            result_arrays = {}
            for col in factor_cols:
                mapped = date_map[col].reindex(train_dates.values).values
                result_arrays[col] = mapped.astype(np.float64)

            result = pd.DataFrame(result_arrays, index=index)
            n_nonnull = result.notna().any(axis=1).sum()
            logger.info(f"Cross-market regime: {len(factor_cols)} factors, "
                        f"{n_nonnull} non-null rows")
            return result
        except Exception as e:
            logger.warning(f"Cross-market regime load failed: {e}")
            return None


def _safe_float(val) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _parse_date_series(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    text = text.mask(text.isin(["", "None", "nan", "NaT"]))
    digits = text.str.replace(r"\D", "", regex=True)
    yyyymmdd = pd.to_datetime(digits.where(digits.str.len() == 8), format="%Y%m%d", errors="coerce")
    parsed = pd.to_datetime(text, errors="coerce")
    return parsed.fillna(yyyymmdd)


def _report_effective_date(report_dates: pd.Series) -> pd.Series:
    """Conservative statutory deadline for quarterly reports WITHOUT ann_date.

    Only used as fallback when no actual announcement date is available.
    Deadlines are based on CSRC disclosure rules (upper bound, not average):
      Q1 (0331): must disclose by Apr 30 → +45 days (was +30, too optimistic)
      H1 (0630): must disclose by Aug 31 → +75 days (was +60)
      Q3 (0930): must disclose by Oct 31 → +45 days (was +30)
      FY (1231): must disclose by Apr 30 → +120 days (unchanged)
    """
    result = report_dates.copy()
    month_day = result.dt.strftime("%m%d")
    delays = pd.Series(90, index=result.index)  # default fallback
    delays = delays.mask(month_day == "0331", 45)
    delays = delays.mask(month_day == "0630", 75)
    delays = delays.mask(month_day == "0930", 45)
    delays = delays.mask(month_day == "1231", 120)
    return result + pd.to_timedelta(delays, unit="D")
