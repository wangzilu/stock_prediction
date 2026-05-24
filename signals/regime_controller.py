"""Market Regime Controller — detect risk/opportunity regimes from macro + micro data.

NOT for stock selection. Controls:
  - Total position size
  - Max turnover
  - Small-cap exposure
  - Whether to enable LLM event overlay

12 regime scores:
  1. liquidity_score: 流动性充裕度 (M2 + Shibor)
  2. credit_stress_score: 信用压力 (Shibor spread)
  3. leverage_unwind_score: 杠杆解除风险 (融资余额)
  4. microcap_crash_risk: 小微盘踩踏风险 (跌停家数)
  5. external_shock_score: 海外冲击 (美债 + 纳指)
  6. policy_support_score: 政策支持力度 (LLM events) [NOT PIT-safe for replay]
  7. theme_breadth_score: 题材扩散宽度 (人气榜) [NOT PIT-safe for replay]
  8. inflation_score: 通胀/通缩压力 (CPI)
  9. northbound_score: 北向资金 (HSGT)
  10. futures_basis_score: IC 期货真基差 (期货-现货)
  11. fx_risk_score: 汇率风险 (USD/CNY)
  12. risk_on_score: 加权综合

Alert logic: weighted average + hard/soft break override.
  - hard_break: PIT-safe + semantically accurate scores only.
  - soft_break: reported but not auto-triggered until validated.

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

# --- Hard/soft break admission rules (CX-approved 2026-05-24) ---
# hard_break: PIT-safe + semantically accurate → can auto-trigger alert override
# soft_break: reported only, no auto-trigger until PIT validation complete
HARD_BREAK_THRESHOLDS = {
    "microcap_crash_risk": -0.8,
    "leverage_unwind_score": -0.8,
    "credit_stress_score": -0.8,
    "external_shock_score": -0.9,
    # futures_basis_score: NOT in hard_break until real basis validated
    # fx_risk_score: NOT in hard_break until 百倍报价 fix validated over time
}
SOFT_BREAK_THRESHOLDS = {
    "northbound_score": -0.8,
    "policy_support_score": -0.8,   # not PIT-safe
    "futures_basis_score": -0.8,    # real basis, but newly implemented
    "fx_risk_score": -0.8,          # newly fixed
}


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
            "inflation_score": self._inflation(date),
            "northbound_score": self._northbound(date),
            "futures_basis_score": self._futures_basis(date),
            "fx_risk_score": self._fx_risk(date),
        }

        # Composite risk_on_score: weighted average
        weights = {
            "liquidity_score": 0.18,
            "credit_stress_score": 0.12,
            "leverage_unwind_score": 0.18,
            "microcap_crash_risk": 0.12,
            "external_shock_score": 0.12,
            "policy_support_score": 0.08,
            "theme_breadth_score": 0.05,
            "inflation_score": 0.06,
            "northbound_score": 0.06,
            "futures_basis_score": 0.08,
            "fx_risk_score": 0.06,
        }
        risk_on = sum(scores[k] * w for k, w in weights.items())
        scores["risk_on_score"] = round(risk_on, 3)

        # Alert level — provisional thresholds (pending PIT-safe replay calibration)
        if risk_on < -0.15:
            alert = "critical"
        elif risk_on < -0.08:
            alert = "warning"
        elif risk_on < -0.02:
            alert = "watch"
        else:
            alert = "normal"

        # --- Hard break override ---
        hard_triggered = {
            k: scores[k] for k, thresh in HARD_BREAK_THRESHOLDS.items()
            if scores.get(k, 0) <= thresh
        }
        soft_triggered = {
            k: scores[k] for k, thresh in SOFT_BREAK_THRESHOLDS.items()
            if scores.get(k, 0) <= thresh
        }

        if hard_triggered:
            # Special combo: microcap + leverage → direct warning
            if (hard_triggered.get("microcap_crash_risk", 0) <= -0.8
                    and scores.get("leverage_unwind_score", 0) <= -0.5):
                if alert in ("normal", "watch"):
                    alert = "warning"
            # 2+ hard breaks → at least warning
            elif len(hard_triggered) >= 2:
                if alert in ("normal", "watch"):
                    alert = "warning"
            # 1 hard break → at least watch
            else:
                if alert == "normal":
                    alert = "watch"

        scores["alert_level"] = alert
        scores["hard_break_signals"] = hard_triggered if hard_triggered else None
        scores["soft_break_signals"] = soft_triggered if soft_triggered else None

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
        """LLM event: policy-related events → support score [-1, +1].

        PIT-safe: only reads event files with filename date <= target date.
        Note: only 18 days of data (2026-04-27~), so historical replay before
        that range will return 0.0.
        """
        try:
            events_dir = DATA_DIR / "llm_events"
            target = pd.Timestamp(date)
            target_str = target.strftime("%Y-%m-%d")

            policy_scores = []
            count = 0
            for f in sorted(events_dir.glob("*.jsonl"), reverse=True):
                # PIT filter: only use files with date <= target
                file_date = f.stem  # e.g. "2026-05-22"
                if file_date > target_str:
                    continue
                if count >= 5:
                    break
                count += 1

                for line in open(f):
                    e = json.loads(line)
                    etype = e.get("event_type", "")
                    if "policy" in etype:
                        impact = e.get("impact_1d", 0)
                        if isinstance(impact, (int, float)):
                            policy_scores.append(impact)

            if policy_scores:
                avg = np.mean(policy_scores)
                return round(max(-1, min(1, avg * 10)), 3)
        except Exception:
            pass
        return 0.0

    def _theme_breadth(self, date: str) -> float:
        """Popularity ranking breadth → theme heat [-1, +1].

        PIT-safe: only reads guba files with filename date <= target date.
        Note: very sparse data (currently 1 file), so most historical replay
        will return 0.0.
        """
        try:
            guba_dir = DATA_DIR / "guba"
            target_str = pd.Timestamp(date).strftime("%Y-%m-%d")

            # Find latest file on or before target date
            best_file = None
            for f in sorted(guba_dir.glob("*.jsonl"), reverse=True):
                if f.stem <= target_str:
                    best_file = f
                    break

            if not best_file:
                return 0.0

            n_hot = sum(1 for _ in open(best_file))
            return round(max(-1, min(1, (n_hot - 50) / 50)), 3)
        except Exception:
            return 0.0

    def _inflation(self, date: str) -> float:
        """CPI → inflation pressure [-1, +1]. High inflation = negative."""
        try:
            cpi = pd.read_parquet(DATA_DIR / "st_cn_cpi.parquet")
            if "nt_yoy" not in cpi.columns:
                return 0.0
            cpi["nt_yoy"] = pd.to_numeric(cpi["nt_yoy"], errors="coerce")
            cpi["month"] = pd.to_datetime(cpi["month"], format="%Y%m", errors="coerce")
            cpi = cpi.dropna(subset=["nt_yoy", "month"])
            cpi = cpi[cpi["month"] <= pd.Timestamp(date)].sort_values("month")
            if cpi.empty:
                return 0.0
            latest = cpi.iloc[-1]["nt_yoy"]
            # CPI YoY ~2% is neutral; >4% is inflationary pressure; <0% is deflationary
            return round(max(-1, min(1, -(latest - 2) / 2)), 3)
        except Exception:
            pass
        return 0.0

    def _northbound(self, date: str) -> float:
        """Northbound (北向) capital flow → foreign sentiment [-1, +1]."""
        try:
            hsgt = pd.read_parquet(DATA_DIR / "st_moneyflow_hsgt.parquet")
            if "trade_date" not in hsgt.columns:
                return 0.0
            hsgt["trade_date"] = pd.to_datetime(hsgt["trade_date"], format="%Y%m%d", errors="coerce")
            hsgt = hsgt.dropna(subset=["trade_date"])
            hsgt = hsgt[hsgt["trade_date"] <= pd.Timestamp(date)].sort_values("trade_date")

            for col in ["north_money", "ggt_ss", "hgt"]:
                if col in hsgt.columns:
                    hsgt[col] = pd.to_numeric(hsgt[col], errors="coerce")
                    vals = hsgt[col].dropna()
                    if len(vals) >= 5:
                        recent = vals.iloc[-5:].mean()
                        hist_std = vals.std()
                        if hist_std > 0:
                            z = recent / hist_std
                            return round(max(-1, min(1, z / 2)), 3)
                    break
        except Exception:
            pass
        return 0.0

    def _futures_basis(self, date: str) -> float:
        """IC futures real basis vs CSI 500 spot → quant crowding risk [-1, +1].

        basis = futures_close / spot_close - 1
        Large discount (basis << 0) = quant short pressure / forced unwind.
        Falls back to IC momentum if spot data unavailable.
        """
        try:
            ic = pd.read_parquet(DATA_DIR / "ak_futures_ic0.parquet")
            ic["日期"] = pd.to_datetime(ic["日期"], errors="coerce")
            ic = ic.dropna(subset=["日期"])
            ic["收盘价"] = pd.to_numeric(ic["收盘价"], errors="coerce")
            ic = ic.dropna(subset=["收盘价"])
            ic = ic[ic["日期"] <= pd.Timestamp(date)].sort_values("日期")

            if len(ic) < 20:
                return 0.0

            # Try real basis with CSI 500 spot
            spot_path = DATA_DIR / "ak_index_csi500.parquet"
            if spot_path.exists():
                spot = pd.read_parquet(spot_path)
                spot["date"] = pd.to_datetime(spot["date"], errors="coerce")
                spot["close"] = pd.to_numeric(spot["close"], errors="coerce")
                spot = spot.dropna(subset=["date", "close"])
                spot = spot[spot["date"] <= pd.Timestamp(date)].sort_values("date")

                if not spot.empty:
                    # Match latest dates
                    ic_latest = ic.iloc[-1]
                    # Find spot on same date or closest prior date
                    spot_on_date = spot[spot["date"] <= ic_latest["日期"]]
                    if not spot_on_date.empty:
                        futures_close = ic_latest["收盘价"]
                        spot_close = spot_on_date.iloc[-1]["close"]
                        if spot_close > 0:
                            basis = futures_close / spot_close - 1
                            # Typical IC basis: -3% to +1%
                            # -5% = extreme discount = quant panic → -1.0
                            # 0% = fair value → 0.0
                            score = max(-1, min(1, basis / 0.05))
                            return round(score, 3)

            # Fallback: IC 5-day momentum (less accurate proxy)
            close = ic["收盘价"]
            ret_5d = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
            score = max(-1, min(1, ret_5d / 0.05))
            return round(score, 3)
        except Exception:
            return 0.0

    def _fx_risk(self, date: str) -> float:
        """USD/CNY movement → currency risk [-1, +1].

        Rapid CNY depreciation = negative (capital outflow risk).
        Data source: 中行折算价 (百倍报价, e.g. 683.73 = 6.8373).
        """
        try:
            fx = pd.read_parquet(DATA_DIR / "ak_usdcny.parquet")

            # Use 中行折算价 explicitly (most complete column, 1488 rows)
            rate_col = "中行折算价"
            if rate_col not in fx.columns:
                # Fallback: find column with 百倍报价 range (600~800)
                for col in fx.columns:
                    vals = pd.to_numeric(fx[col], errors="coerce").dropna()
                    if len(vals) > 100 and 500 < vals.mean() < 1000:
                        rate_col = col
                        break
                else:
                    return 0.0

            # Date column
            date_col = "日期"
            if date_col not in fx.columns:
                for col in fx.columns:
                    if "日期" in col or "date" in col.lower():
                        date_col = col
                        break
                else:
                    return 0.0

            fx[date_col] = pd.to_datetime(fx[date_col], errors="coerce")
            fx = fx.dropna(subset=[date_col])
            fx = fx[fx[date_col] <= pd.Timestamp(date)].sort_values(date_col)

            # Convert 百倍报价 to actual rate (683.73 → 6.8373)
            rate = pd.to_numeric(fx[rate_col], errors="coerce").dropna() / 100.0
            if len(rate) < 20:
                return 0.0

            # 5-day change: USD/CNY going up = CNY weakening = negative
            change_5d = (rate.iloc[-1] - rate.iloc[-5]) / rate.iloc[-5]
            # +1% in 5 days (CNY depreciation) → -0.5 score
            score = max(-1, min(1, -change_5d / 0.02))
            return round(score, 3)
        except Exception:
            return 0.0

    def _suggest_adjustments(self, scores: dict) -> dict:
        """Suggest trading parameter changes based on regime.

        NOTE: These are advisory only. Do NOT auto-execute in OMS until
        PIT-safe replay validation and 24-split backtest are complete.
        """
        alert = scores["alert_level"]

        if alert == "critical":
            return {
                "max_position": 0.3,
                "max_turnover": 0.05,
                "smallcap_exposure": 0.0,
                "event_overlay": False,
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
        elif k in ("hard_break_signals", "soft_break_signals"):
            if v:
                print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
