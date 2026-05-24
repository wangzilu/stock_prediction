"""Market Regime Controller — detect risk/opportunity regimes from macro + micro data.

NOT for stock selection. Controls:
  - Total position size
  - Max turnover
  - Small-cap exposure
  - Whether to enable LLM event overlay

8 regime scores (CX design):
  1. policy_support_score: 政策支持力度
  2. liquidity_score: 流动性充裕度
  3. credit_stress_score: 信用压力
  4. leverage_unwind_score: 杠杆解除风险
  5. microcap_crash_risk: 小微盘踩踏风险
  6. external_shock_score: 海外冲击
  7. theme_breadth_score: 题材扩散宽度
  8. risk_on_score: 综合风险偏好

Usage:
    from signals.regime_controller import RegimeController
    rc = RegimeController()
    regime = rc.compute(date="2026-05-22")
    print(regime["risk_on_score"])  # -1 to +1
    print(regime["alert_level"])    # "normal" / "watch" / "warning" / "critical"
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "storage"


class RegimeController:

    def compute(self, date: str = None) -> dict:
        """Compute all regime scores for a given date."""
        date = date or datetime.now().strftime("%Y-%m-%d")

        scores = {
            "date": date,
            "liquidity_score": self._liquidity(date),
            "credit_stress_score": self._credit_stress(date),
            "leverage_unwind_score": self._leverage(date),
            "microcap_crash_risk": self._microcap_crash(date),
            "external_shock_score": self._external_shock(date),
            "policy_support_score": self._policy_support(date),
            "theme_breadth_score": self._theme_breadth(date),
        }

        # Composite risk_on_score: weighted average
        weights = {
            "liquidity_score": 0.20,
            "credit_stress_score": 0.15,
            "leverage_unwind_score": 0.20,
            "microcap_crash_risk": 0.15,
            "external_shock_score": 0.15,
            "policy_support_score": 0.10,
            "theme_breadth_score": 0.05,
        }
        risk_on = sum(scores[k] * w for k, w in weights.items())
        scores["risk_on_score"] = round(risk_on, 3)

        # Alert level
        if risk_on < -0.5:
            scores["alert_level"] = "critical"
        elif risk_on < -0.25:
            scores["alert_level"] = "warning"
        elif risk_on < -0.1:
            scores["alert_level"] = "watch"
        else:
            scores["alert_level"] = "normal"

        # Trading parameter adjustments
        scores["suggested_adjustments"] = self._suggest_adjustments(scores)

        return scores

    def _as_of(self, df: pd.DataFrame, date: str, date_col: str) -> pd.DataFrame:
        """Filter DataFrame to rows on or before date (point-in-time safe)."""
        df[date_col] = pd.to_datetime(df[date_col], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=[date_col])
        target = pd.Timestamp(date)
        return df[df[date_col] <= target].sort_values(date_col)

    def _liquidity(self, date: str) -> float:
        """M2 growth + Shibor level → liquidity score [-1, +1]."""
        score = 0.0
        n = 0

        # M2
        try:
            m2 = pd.read_parquet(DATA_DIR / "st_cn_m.parquet")
            m2["m2_yoy"] = pd.to_numeric(m2["m2_yoy"], errors="coerce")
            m2["month"] = pd.to_datetime(m2["month"], format="%Y%m", errors="coerce")
            m2 = m2.dropna(subset=["m2_yoy", "month"])
            m2 = m2[m2["month"] <= pd.Timestamp(date)].sort_values("month")
            if not m2.empty:
                m2_yoy = m2.iloc[-1]["m2_yoy"]
                score += max(-1, min(1, (m2_yoy - 9) / 3))
                n += 1
        except Exception:
            pass

        # Shibor overnight
        try:
            shibor = self._as_of(
                pd.read_parquet(DATA_DIR / "st_shibor.parquet"), date, "date")
            shibor["on"] = pd.to_numeric(shibor["on"], errors="coerce")
            shibor = shibor.dropna(subset=["on"])
            if not shibor.empty:
                on_rate = shibor.iloc[-1]["on"]
                score += max(-1, min(1, -(on_rate - 2.0)))
                n += 1
        except Exception:
            pass

        return round(score / max(n, 1), 3)

    def _credit_stress(self, date: str) -> float:
        """Shibor term spread + level → credit stress [-1, +1]."""
        try:
            shibor = self._as_of(
                pd.read_parquet(DATA_DIR / "st_shibor.parquet"), date, "date")
            for col in ["on", "3m"]:
                shibor[col] = pd.to_numeric(shibor[col], errors="coerce")
            shibor = shibor.dropna(subset=["on", "3m"])
            if shibor.empty:
                return 0.0
            latest = shibor.iloc[-1]
            spread = latest["3m"] - latest["on"]
            spread_score = max(-1, min(1, -(spread - 0.5) / 1.0))
            level_score = max(-1, min(1, -(latest["3m"] - 2.0) / 2.0))
            return round((spread_score + level_score) / 2, 3)
        except Exception:
            return 0.0

    def _leverage(self, date: str) -> float:
        """Margin balance change → leverage unwind risk [-1, +1]."""
        try:
            md = pd.read_parquet(DATA_DIR / "st_margin_detail.parquet")
            md["trade_date"] = pd.to_datetime(md["trade_date"], format="%Y%m%d", errors="coerce")
            md["rzye"] = pd.to_numeric(md["rzye"], errors="coerce")
            md = md.dropna(subset=["trade_date", "rzye"])
            md = md[md["trade_date"] <= pd.Timestamp(date)]

            daily_stats = md.groupby("trade_date").agg(
                total_rzye=("rzye", "sum"),
                n_stocks=("rzye", "count"),
            ).sort_index()

            daily_stats = daily_stats[daily_stats["n_stocks"] >= 3000]
            if len(daily_stats) < 20:
                return 0.0

            ewma = daily_stats["total_rzye"].ewm(span=20).mean()
            if len(ewma) < 20:
                return 0.0

            change_rate = (ewma.iloc[-1] - ewma.iloc[-20]) / ewma.iloc[-20]
            return round(max(-1, min(1, change_rate / 0.10)), 3)
        except Exception:
            pass
        return 0.0

    def _microcap_crash(self, date: str) -> float:
        """Limit-down count → crash risk [-1, +1]."""
        try:
            ld = pd.read_parquet(DATA_DIR / "st_limit_list_d.parquet")
            ld["trade_date"] = pd.to_datetime(ld["trade_date"], format="%Y%m%d", errors="coerce")
            ld = ld.dropna(subset=["trade_date"])
            ld = ld[ld["trade_date"] <= pd.Timestamp(date)]

            if "limit" in ld.columns:
                down = ld[ld["limit"].astype(str).str.upper() == "D"]
            else:
                ld["pct_chg"] = pd.to_numeric(ld.get("pct_chg", pd.Series()), errors="coerce")
                down = ld[ld["pct_chg"] < -9]

            daily_count = down.groupby("trade_date").size().sort_index()
            if len(daily_count) < 5:
                return 0.0

            recent_avg = daily_count.iloc[-5:].mean()
            score = max(-1, min(0, -(recent_avg - 10) / 40))
            return round(score, 3)
        except Exception:
            return 0.0

    def _external_shock(self, date: str) -> float:
        """US treasury + cross-market indices → external shock [-1, +1]."""
        score = 0.0
        n = 0

        # US treasury yield change
        try:
            us = pd.read_parquet(DATA_DIR / "st_us_tycr.parquet")
            # Try to find date column and filter
            for dcol in ["date", "trade_date"]:
                if dcol in us.columns:
                    us[dcol] = pd.to_datetime(us[dcol], format="%Y%m%d", errors="coerce")
                    us = us.dropna(subset=[dcol])
                    us = us[us[dcol] <= pd.Timestamp(date)].sort_values(dcol)
                    break
            for col in us.columns:
                if "10" in str(col) or "y10" in str(col).lower():
                    us[col] = pd.to_numeric(us[col], errors="coerce")
                    vals = us[col].dropna()
                    if len(vals) >= 2:
                        change = vals.iloc[-1] - vals.iloc[-2]
                        score += max(-1, min(0, -change / 0.15))
                        n += 1
                    break
        except Exception:
            pass

        # Cross-market: nasdaq from cache
        try:
            cache = pd.read_parquet(DATA_DIR / "feature_cache_174_holder_regime_ma.parquet",
                                    columns=["nasdaq_ret1d"])
            cache_dates = cache.index.get_level_values(0)
            target = pd.Timestamp(date)
            avail = cache_dates[cache_dates <= target]
            if len(avail) > 0:
                use_date = avail.max()
                nasdaq_ret = cache.loc[use_date, "nasdaq_ret1d"].mean()
                if np.isfinite(nasdaq_ret):
                    score += max(-1, min(0, nasdaq_ret / 0.03)) if nasdaq_ret < 0 else 0
                    n += 1
        except Exception:
            pass

        return round(score / max(n, 1), 3)

    def _policy_support(self, date: str) -> float:
        """LLM event: policy-related events → support score [-1, +1]."""
        try:
            events_dir = DATA_DIR / "llm_events"
            # Check last 5 days of events for policy signals
            policy_scores = []
            target = pd.Timestamp(date)

            for f in sorted(events_dir.glob("*.jsonl"), reverse=True)[:5]:
                for line in open(f):
                    e = json.loads(line)
                    etype = e.get("event_type", "")
                    if "policy" in etype:
                        impact = e.get("impact_1d", 0)
                        policy_scores.append(impact)

            if policy_scores:
                avg = np.mean(policy_scores)
                return round(max(-1, min(1, avg * 10)), 3)  # amplify small impacts
        except Exception:
            pass
        return 0.0

    def _theme_breadth(self, date: str) -> float:
        """Popularity ranking breadth → theme heat [-1, +1]."""
        try:
            guba_dir = DATA_DIR / "guba"
            files = sorted(guba_dir.glob("*.jsonl"), reverse=True)
            if not files:
                return 0.0

            latest = files[0]
            n_hot = sum(1 for _ in open(latest))
            # 100 hot stocks is normal, fewer means narrow market
            return round(max(-1, min(1, (n_hot - 50) / 50)), 3)
        except Exception:
            return 0.0

    def _suggest_adjustments(self, scores: dict) -> dict:
        """Suggest trading parameter changes based on regime."""
        alert = scores["alert_level"]
        risk_on = scores["risk_on_score"]

        if alert == "critical":
            return {
                "max_position": 0.3,       # 30% invested, 70% cash
                "max_turnover": 0.05,       # minimal trading
                "smallcap_exposure": 0.0,   # no small caps
                "event_overlay": False,     # disable experimental signals
                "reason": "多重风险信号共振，大幅降仓",
            }
        elif alert == "warning":
            return {
                "max_position": 0.6,
                "max_turnover": 0.08,
                "smallcap_exposure": 0.1,
                "event_overlay": True,
                "reason": "风险偏高，适度降仓",
            }
        elif alert == "watch":
            return {
                "max_position": 0.8,
                "max_turnover": 0.10,
                "smallcap_exposure": 0.2,
                "event_overlay": True,
                "reason": "关注风险，略降仓",
            }
        else:
            return {
                "max_position": 1.0,
                "max_turnover": 0.10,
                "smallcap_exposure": 0.3,
                "event_overlay": True,
                "reason": "正常市场",
            }


def main():
    """Quick test."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    logging.basicConfig(level=logging.INFO)

    rc = RegimeController()
    result = rc.compute()

    print(f"\n=== Regime Controller: {result['date']} ===\n")
    for k, v in result.items():
        if k == "suggested_adjustments":
            print(f"\n  Adjustments:")
            for ak, av in v.items():
                print(f"    {ak}: {av}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
