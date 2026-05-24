"""Portfolio risk model — ShrinkCov + Barra exposure integration.

Does NOT replace optimizer_v2. Provides three risk overlays:
  1. Portfolio predicted volatility (daily output for RiskGuard L2)
  2. High-correlation holding penalty (for reranker diversification)
  3. Marginal contribution to risk (MCTR) per stock

Uses Qlib's ShrinkCovEstimator (Ledoit-Wolf) for covariance estimation.
Integrates barra_simple.py style exposures for monitoring.

Usage:
    from backtest.risk_model import PortfolioRiskModel

    rm = PortfolioRiskModel()
    report = rm.compute(
        holdings={"sh600519": 0.10, "sz000858": 0.08, ...},
        date="2026-05-22",
    )
    print(report["portfolio_vol"])       # annualized vol prediction
    print(report["high_corr_pairs"])     # correlated pairs > 0.8
    print(report["mctr"])                # {stock: marginal risk contribution}
    print(report["style_exposure"])      # from barra_simple
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "storage"

# Lookback for covariance estimation
COV_LOOKBACK = 60  # trading days
ANNUALIZE_FACTOR = np.sqrt(252)
CORR_THRESHOLD = 0.8  # flag pairs above this


class PortfolioRiskModel:

    def __init__(self, lookback: int = COV_LOOKBACK):
        self.lookback = lookback
        self._return_cache = None
        self._cache_date = None

    def compute(self, holdings: dict, date: str = None) -> dict:
        """Compute full risk report for a portfolio.

        Args:
            holdings: {instrument: weight} where weights sum to ~1.0
            date: as-of date (default: latest available)

        Returns:
            dict with portfolio_vol, high_corr_pairs, mctr, style_exposure, etc.
        """
        if not holdings:
            return {"portfolio_vol": 0.0, "error": "empty holdings"}

        instruments = list(holdings.keys())
        weights = np.array([holdings[s] for s in instruments])

        # Get return matrix
        returns = self._get_returns(instruments, date)
        if returns is None or returns.shape[0] < 20:
            return {
                "portfolio_vol": 0.0,
                "error": f"insufficient return data: {returns.shape[0] if returns is not None else 0} days",
            }

        # Estimate covariance with Ledoit-Wolf shrinkage
        cov_matrix = self._estimate_covariance(returns)

        # Align weights with available instruments
        available = returns.columns.tolist()
        w = np.array([holdings.get(s, 0) for s in available])
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum  # normalize

        report = {}

        # 1. Portfolio predicted volatility
        port_var = w @ cov_matrix @ w
        port_vol_daily = np.sqrt(max(port_var, 0))
        report["portfolio_vol_daily"] = round(float(port_vol_daily), 6)
        report["portfolio_vol_annual"] = round(float(port_vol_daily * ANNUALIZE_FACTOR), 4)

        # 2. High-correlation pairs
        corr_matrix = self._cov_to_corr(cov_matrix)
        high_corr = self._find_high_corr_pairs(corr_matrix, available, w)
        report["high_corr_pairs"] = high_corr
        report["n_high_corr_pairs"] = len(high_corr)

        # 3. MCTR (Marginal Contribution to Total Risk)
        mctr = self._compute_mctr(cov_matrix, w, available)
        report["mctr"] = mctr

        # 4. Risk concentration
        if mctr:
            mctr_vals = list(mctr.values())
            report["risk_concentration_top3"] = round(
                sum(sorted(mctr_vals, reverse=True)[:3]), 4
            )
            report["risk_concentration_hhi"] = round(
                sum(v**2 for v in mctr_vals), 6
            )

        # 5. Style exposure from barra_simple
        try:
            from backtest.barra_simple import compute_style_exposures, compute_portfolio_exposure
            exposures = compute_style_exposures(date=date)
            style_exp = compute_portfolio_exposure(holdings, exposures)
            report["style_exposure"] = style_exp
        except Exception as e:
            report["style_exposure"] = {"error": str(e)}

        report["date"] = date
        report["n_holdings"] = len(available)
        report["lookback_days"] = returns.shape[0]

        return report

    def compute_diversification_penalty(
        self, holdings: dict, candidates: list, date: str = None
    ) -> dict:
        """Compute correlation-based penalty for each candidate stock.

        For reranker: penalize candidates highly correlated with existing holdings.

        Args:
            holdings: {instrument: weight} current portfolio
            candidates: list of candidate instruments to evaluate
            date: as-of date

        Returns:
            {instrument: penalty} where penalty in [0, 1].
            0 = fully diversifying, 1 = fully redundant.
        """
        all_instruments = list(set(list(holdings.keys()) + candidates))
        returns = self._get_returns(all_instruments, date)
        if returns is None or returns.shape[0] < 20:
            return {c: 0.0 for c in candidates}

        cov_matrix = self._estimate_covariance(returns)
        corr_matrix = self._cov_to_corr(cov_matrix)
        available = returns.columns.tolist()

        penalties = {}
        for cand in candidates:
            if cand not in available:
                penalties[cand] = 0.0
                continue

            cand_idx = available.index(cand)
            max_corr = 0.0
            weighted_corr = 0.0
            total_w = 0.0

            for inst, w in holdings.items():
                if inst in available and inst != cand:
                    inst_idx = available.index(inst)
                    corr = corr_matrix[cand_idx, inst_idx]
                    if np.isfinite(corr):
                        max_corr = max(max_corr, abs(corr))
                        weighted_corr += w * abs(corr)
                        total_w += w

            avg_corr = weighted_corr / max(total_w, 1e-8)
            # Penalty = weighted average of max and avg correlation
            penalty = 0.6 * max_corr + 0.4 * avg_corr
            penalties[cand] = round(float(min(max(penalty, 0), 1)), 4)

        return penalties

    def _get_returns(self, instruments: list, date: str = None) -> pd.DataFrame:
        """Load daily returns for instruments from feature cache."""
        try:
            cache = pd.read_parquet(
                DATA_DIR / "feature_cache_174_holder_regime_ma.parquet",
                columns=["__pnl_return_1d"],
            )
        except Exception as e:
            logger.warning(f"Failed to load return data: {e}")
            return None

        dates = sorted(cache.index.get_level_values(0).unique())
        if date:
            target = pd.Timestamp(date)
            dates = [d for d in dates if d <= target]

        if len(dates) < self.lookback:
            use_dates = dates
        else:
            use_dates = dates[-self.lookback:]

        # Pivot to (date × instrument) matrix
        subset = cache.loc[cache.index.get_level_values(0).isin(use_dates)]
        ret_series = subset["__pnl_return_1d"]

        pivot = ret_series.unstack(level=1)
        # Filter to requested instruments
        common = [s for s in instruments if s in pivot.columns]
        if not common:
            return None

        returns = pivot[common].dropna(axis=1, thresh=int(len(use_dates) * 0.5))
        returns = returns.fillna(0)
        return returns

    def _estimate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate covariance using Ledoit-Wolf shrinkage."""
        try:
            from qlib.model.riskmodel import ShrinkCovEstimator
            estimator = ShrinkCovEstimator()
            # is_price=False: input is already returns, not prices
            cov = estimator.predict(returns.values, is_price=False)
            return cov
        except Exception:
            # Fallback: simple sample covariance with shrinkage
            return self._simple_shrink_cov(returns.values)

    def _simple_shrink_cov(self, X: np.ndarray) -> np.ndarray:
        """Simple Ledoit-Wolf shrinkage fallback."""
        n, p = X.shape
        sample_cov = np.cov(X, rowvar=False)
        # Shrink toward diagonal
        target = np.diag(np.diag(sample_cov))
        # Oracle approximation shrinkage
        shrinkage = min(1.0, max(0.0, (p / n) * 0.5))
        return (1 - shrinkage) * sample_cov + shrinkage * target

    def _cov_to_corr(self, cov: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        d = np.sqrt(np.diag(cov))
        d[d < 1e-10] = 1e-10
        return cov / np.outer(d, d)

    def _find_high_corr_pairs(
        self, corr: np.ndarray, names: list, weights: np.ndarray
    ) -> list:
        """Find pairs with correlation above threshold, weighted by portfolio weight."""
        pairs = []
        n = len(names)
        for i in range(n):
            if weights[i] < 0.01:  # skip tiny positions
                continue
            for j in range(i + 1, n):
                if weights[j] < 0.01:
                    continue
                c = corr[i, j]
                if np.isfinite(c) and abs(c) > CORR_THRESHOLD:
                    pairs.append({
                        "stock_a": names[i],
                        "stock_b": names[j],
                        "correlation": round(float(c), 3),
                        "combined_weight": round(float(weights[i] + weights[j]), 3),
                    })
        return sorted(pairs, key=lambda x: -abs(x["correlation"]))

    def _compute_mctr(
        self, cov: np.ndarray, weights: np.ndarray, names: list
    ) -> dict:
        """Marginal Contribution to Total Risk.

        MCTR_i = w_i * (Sigma @ w)_i / sigma_p
        Sums to 1.0 (each stock's risk contribution as fraction of total).
        """
        sigma_w = cov @ weights
        port_var = weights @ sigma_w
        port_vol = np.sqrt(max(port_var, 1e-12))

        mctr = {}
        for i, name in enumerate(names):
            contribution = weights[i] * sigma_w[i] / port_vol
            mctr[name] = round(float(contribution), 6)

        return mctr


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    logging.basicConfig(level=logging.INFO)

    rm = PortfolioRiskModel()

    # Demo: simulate top 10 equal weight
    from backtest.barra_simple import compute_style_exposures
    exp = compute_style_exposures()
    top10 = list(exp.index[:10])
    holdings = {s: 0.10 for s in top10}

    report = rm.compute(holdings)
    print(f"\n=== Portfolio Risk Report ===")
    print(f"  Holdings: {report.get('n_holdings')}")
    print(f"  Lookback: {report.get('lookback_days')} days")
    print(f"  Daily vol: {report.get('portfolio_vol_daily', 0):.4f}")
    print(f"  Annual vol: {report.get('portfolio_vol_annual', 0):.2%}")
    print(f"  High-corr pairs: {report.get('n_high_corr_pairs', 0)}")
    if report.get("high_corr_pairs"):
        for p in report["high_corr_pairs"][:5]:
            print(f"    {p['stock_a']} ↔ {p['stock_b']}: {p['correlation']}")
    print(f"  Risk top3 concentration: {report.get('risk_concentration_top3', 0):.2%}")

    if report.get("style_exposure") and "error" not in report["style_exposure"]:
        print(f"\n  Style exposure:")
        for k, v in report["style_exposure"].items():
            print(f"    {k}: {v}")

    print(f"\n  MCTR (top 5):")
    mctr = report.get("mctr", {})
    for name, val in sorted(mctr.items(), key=lambda x: -x[1])[:5]:
        print(f"    {name}: {val:.4f}")
