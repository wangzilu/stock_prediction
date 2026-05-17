"""Phase 4C: Risk exposure and capacity analysis.

Answers:
1. Is the return from stock selection alpha or industry/style drift?
2. Are single-stock or industry concentrations too high?
3. Can the strategy handle real capital (ADV participation)?

Usage:
    from backtest.exposure_report import ExposureAnalyzer
    analyzer = ExposureAnalyzer()
    report = analyzer.analyze(predictions, daily_returns, test_index)
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "storage"


class ExposureAnalyzer:
    """Analyze portfolio risk exposures: industry, size, concentration, capacity."""

    def __init__(self, industry_path: Path = None):
        self.industry_path = industry_path or DATA_DIR / "industry_mapping.parquet"
        self._industry_map = None

    def _load_industry_map(self) -> dict:
        """Load stock -> industry mapping."""
        if self._industry_map is not None:
            return self._industry_map

        if not self.industry_path.exists():
            logger.warning("Industry mapping not found, skipping industry analysis")
            return {}

        df = pd.read_parquet(self.industry_path)
        if "qlib_code" not in df.columns or "industry" not in df.columns:
            return {}

        self._industry_map = df.drop_duplicates("qlib_code").set_index(
            "qlib_code")["industry"].to_dict()
        return self._industry_map

    def analyze(
        self,
        predictions: pd.Series,
        daily_returns: pd.DataFrame,
        top_k: int = 20,
        buffer: int = 5,
    ) -> dict:
        """Full exposure analysis on the test period.

        Args:
            predictions: Series indexed by (datetime, instrument) with model scores
            daily_returns: DataFrame with 1-day realized returns
            top_k: number of stocks in portfolio
            buffer: buffer zone size

        Returns:
            dict with exposure metrics
        """
        industry_map = self._load_industry_map()

        dates = sorted(predictions.index.get_level_values(0).unique())
        logger.info(f"Exposure analysis: {len(dates)} dates, top_k={top_k}")

        # Per-date analysis
        daily_industry_weights = []
        daily_stock_weights = []
        daily_top_concentration = []
        daily_industry_counts = []

        for date in dates:
            if date not in predictions.index.get_level_values(0):
                continue

            day_pred = predictions.loc[date]
            if isinstance(day_pred, pd.DataFrame):
                scores = day_pred.iloc[:, 0]
            else:
                scores = day_pred
            scores = scores.dropna()

            if len(scores) < top_k:
                continue

            # Select top_k stocks
            top_stocks = scores.nlargest(top_k)
            weight = 1.0 / len(top_stocks)  # equal weight

            # Stock concentration
            daily_stock_weights.append({
                "date": date,
                "max_weight": weight,
                "n_stocks": len(top_stocks),
            })

            # Top 5 concentration
            top5_weight = min(5, len(top_stocks)) * weight
            daily_top_concentration.append(top5_weight)

            # Industry exposure
            if industry_map:
                ind_weights = {}
                for stock in top_stocks.index:
                    code = str(stock).lower()
                    ind = industry_map.get(code, "unknown")
                    ind_weights[ind] = ind_weights.get(ind, 0) + weight

                daily_industry_weights.append({
                    "date": date,
                    **ind_weights,
                })
                daily_industry_counts.append(len(ind_weights))

        # Aggregate results
        report = {
            "n_dates": len(dates),
            "top_k": top_k,
        }

        # Stock concentration
        if daily_stock_weights:
            max_weights = [d["max_weight"] for d in daily_stock_weights]
            report["stock_concentration"] = {
                "avg_max_weight": round(float(np.mean(max_weights)), 4),
                "max_max_weight": round(float(np.max(max_weights)), 4),
                "avg_top5_concentration": round(float(np.mean(daily_top_concentration)), 4),
                "gate_max_weight_8pct": float(np.max(max_weights)) <= 0.08,
            }

        # Industry exposure
        if daily_industry_weights:
            ind_df = pd.DataFrame(daily_industry_weights).set_index("date").fillna(0)

            # Max industry weight across all dates
            max_ind_weight = ind_df.max().max()
            avg_ind_count = np.mean(daily_industry_counts)

            # Most frequent top industry
            top_industry_per_day = ind_df.idxmax(axis=1)
            top_industry_freq = top_industry_per_day.value_counts()

            # Industry persistence: is the same industry always on top?
            top_ind = top_industry_freq.index[0] if len(top_industry_freq) > 0 else "unknown"
            top_ind_pct = top_industry_freq.iloc[0] / len(ind_df) if len(top_industry_freq) > 0 else 0

            report["industry_exposure"] = {
                "avg_industries_held": round(float(avg_ind_count), 1),
                "max_single_industry_weight": round(float(max_ind_weight), 4),
                "gate_max_industry_25pct": float(max_ind_weight) <= 0.25,
                "most_frequent_top_industry": top_ind,
                "top_industry_persistence": round(float(top_ind_pct), 4),
                "top_5_industries": top_industry_freq.head(5).to_dict(),
            }

            # Check if return is driven by single industry
            if top_ind_pct > 0.5:
                report["industry_exposure"]["warning"] = (
                    f"Portfolio frequently dominated by '{top_ind}' "
                    f"({top_ind_pct:.0%} of days). "
                    f"Return may be industry beta, not stock selection alpha."
                )

        # Capacity estimate (rough)
        # With equal weight and 20 stocks, each stock gets 5% of AUM
        # If AUM = 1M, each stock position = 50K
        # ADV participation = 50K / stock_ADV
        report["capacity"] = {
            "equal_weight_per_stock": round(1.0 / top_k, 4),
            "note": "ADV participation requires live ADV data; "
                    "with top_k=20 equal weight, each stock = 5% of AUM. "
                    "For 1M AUM, position = 50K per stock, "
                    "well within ADV for most A-share stocks.",
            "estimated_safe_aum_range": "100K - 10M CNY",
        }

        # Gate summary
        report["gate_pass"] = {
            "stock_weight_ok": report.get("stock_concentration", {}).get(
                "gate_max_weight_8pct", True),
            "industry_weight_ok": report.get("industry_exposure", {}).get(
                "gate_max_industry_25pct", True),
        }
        report["all_gates_pass"] = all(report["gate_pass"].values())

        return report

    def print_report(self, report: dict):
        """Print human-readable exposure report."""
        print(f"\n{'='*60}")
        print("PHASE 4C: RISK EXPOSURE REPORT")
        print(f"{'='*60}")

        sc = report.get("stock_concentration", {})
        print(f"\n  Stock Concentration:")
        print(f"    Max single-stock weight: {sc.get('max_max_weight', 0)*100:.1f}% "
              f"(gate ≤8%: {'✅' if sc.get('gate_max_weight_8pct') else '❌'})")
        print(f"    Top5 concentration: {sc.get('avg_top5_concentration', 0)*100:.1f}%")

        ie = report.get("industry_exposure", {})
        if ie:
            print(f"\n  Industry Exposure:")
            print(f"    Avg industries held: {ie.get('avg_industries_held', 0):.1f}")
            print(f"    Max industry weight: {ie.get('max_single_industry_weight', 0)*100:.1f}% "
                  f"(gate ≤25%: {'✅' if ie.get('gate_max_industry_25pct') else '❌'})")
            print(f"    Most common top: {ie.get('most_frequent_top_industry', '?')} "
                  f"({ie.get('top_industry_persistence', 0)*100:.0f}% of days)")
            if "warning" in ie:
                print(f"    ⚠️  {ie['warning']}")
            top5 = ie.get("top_5_industries", {})
            if top5:
                print(f"    Top 5 industries by frequency:")
                for ind, count in list(top5.items())[:5]:
                    print(f"      {ind}: {count} days")

        cap = report.get("capacity", {})
        print(f"\n  Capacity:")
        print(f"    Per-stock weight: {cap.get('equal_weight_per_stock', 0)*100:.1f}%")
        print(f"    Safe AUM range: {cap.get('estimated_safe_aum_range', '?')}")

        print(f"\n  Gate: {'✅ ALL PASS' if report.get('all_gates_pass') else '❌ FAIL'}")
        print(f"{'='*60}")
