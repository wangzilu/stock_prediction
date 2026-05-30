from __future__ import annotations

import os
import logging
import math
import json
from dataclasses import is_dataclass, replace
from datetime import datetime, timedelta

from data.collectors.market import MarketCollector
# data.collectors.crypto is lazy-imported via DailyPipeline._get_crypto_collector
# when LEGACY_MARKET_CONTEXT_ENABLED is true. Default-off per quarantine
# (plans/cc-crypto-implementation-spec-2026-05-30.md §6.5).
from data.collectors.gold import GoldCollector
from data.collectors.sentiment import SentimentCollector
from data.collectors.macro import MacroCollector
from data.collectors.global_indices import GlobalIndicesCollector
from factors.candidate_sanitizer import CandidateSanitizer
from factors.sentiment import SentimentScorer
from signals.scorer import SignalScorer
from signals.risk_monitor import RiskMonitor
from signals.market_judge import MarketJudge
from signals.llm_analyst import LLMAnalyst
from signals.index_predictor import OvernightIndexPredictor
from push.wechat import WeChatPusher
from tracker.verifier import Verifier
from config.watchlist import (
    WATCHLIST, MARKET_STOCK, MARKET_CRYPTO, MARKET_GOLD,
    SENTIMENT_TOP_N, to_akshare_code,
)
from config.settings import (
    TAKE_PROFIT_PCT, STOP_LOSS_PCT, LGB_FLIP_THRESHOLD,
    LGB_MODEL_PATH, RL_MODEL_PATH, MID_MODEL_PATH, PREDICTION_HORIZON_DAYS,
    LGB_MIN_PREDICTIONS, OVERNIGHT_STOCK_SNAPSHOT_PATH, DATA_DIR,
)
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HORIZON_BUCKET_SIZE = 3


def _finite_float(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _sanitize_push_text(text: str) -> str:
    """Hide implementation-specific model names from user-facing pushes."""
    replacements = {
        "Qlib": "个股模型",
        "qlib": "个股模型",
        "LightGBM": "短线模型",
        "LGB": "短线模型",
        "lgb": "短线模型",
    }
    cleaned = text or ""
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    return cleaned


class DailyPipeline:
    """Orchestrates the daily recommendation pipeline for all markets."""

    def __init__(self):
        self.market_collector = MarketCollector()
        # Lazy via _get_crypto_collector(); see quarantine §6.5.
        self._crypto_collector = None
        self.gold_collector = GoldCollector()
        self.sentiment_collector = SentimentCollector()
        self.macro_collector = MacroCollector()
        self.sentiment_scorer = SentimentScorer()
        self.global_indices = GlobalIndicesCollector()
        self.signal_scorer = SignalScorer()
        self.risk_monitor = RiskMonitor()
        self.pusher = WeChatPusher()
        self.verifier = Verifier()
        self.market_judge = MarketJudge()
        self.llm_analyst = LLMAnalyst()
        self.index_predictor = OvernightIndexPredictor()

        # Cached data (computed once per run)
        self._geo_factors = None
        self._headlines = None
        self._capital_flow_signals = None  # {qlib_code: {"net_mf": ..., "nb_days": ...}}

        # Pre-trained models (loaded lazily)
        self._lgb_predictions = None
        self._lgb_status = {"status": "unknown", "count": 0, "error": ""}
        self._rl_agent = None
        self._mid_model = None
        self._mid_model_checked = False

    def _get_crypto_collector(self):
        """Lazy accessor for the legacy CryptoCollector.

        Returns None when LEGACY_MARKET_CONTEXT_ENABLED is False (default).
        Module-level import of data.collectors.crypto has been removed
        (see §6.5 quarantine); when the flag is off, the module is never
        loaded and no ccxt/network risk affects A-share startup.
        """
        from config.feature_flags import LEGACY_MARKET_CONTEXT_ENABLED
        if not LEGACY_MARKET_CONTEXT_ENABLED:
            return None
        if self._crypto_collector is None:
            from data.collectors.crypto import CryptoCollector
            self._crypto_collector = CryptoCollector()
        return self._crypto_collector

    def _fetch_crypto_market_data(self):
        """Fetch BTC/ETH realtime market data dict.

        Returns empty dict when legacy crypto is quarantined off. Used
        by morning/evening report paths that previously inlined the
        fetch loop.
        """
        collector = self._get_crypto_collector()
        if collector is None:
            return {}
        data = {}
        for symbol in ("BTC/USDT", "ETH/USDT"):
            q = collector.fetch_realtime(symbol)
            if q:
                data[symbol] = q
        return data

    def _get_quote(self, code, market):
        """Get realtime quote based on market type."""
        if market == MARKET_STOCK:
            return self.market_collector.fetch_realtime(to_akshare_code(code))
        elif market == MARKET_CRYPTO:
            collector = self._get_crypto_collector()
            if collector is None:
                return {}
            return collector.fetch_realtime(code)
        elif market == MARKET_GOLD:
            return self.gold_collector.fetch_realtime()
        return {}

    def _get_daily(self, code, market, days=10):
        """Get daily data based on market type."""
        if market == MARKET_STOCK:
            return self.market_collector.fetch_daily(to_akshare_code(code), days)
        elif market == MARKET_CRYPTO:
            collector = self._get_crypto_collector()
            if collector is None:
                import pandas as pd
                return pd.DataFrame()
            return collector.fetch_daily(code, days)
        elif market == MARKET_GOLD:
            return self.gold_collector.fetch_daily(days)
        import pandas as pd
        return pd.DataFrame()

    def _market_label(self, market):
        """Get display label for market type."""
        return {
            MARKET_STOCK: "A股",
            MARKET_CRYPTO: "加密货币",
            MARKET_GOLD: "黄金",
        }.get(market, "")

    def _load_lgb_predictions(self):
        """Load LGB model and get latest predictions for all stocks."""
        cached = getattr(self, "_lgb_predictions", None)
        if cached is not None:
            return cached

        def _use_cache(reason: str, live_count: int = 0):
            try:
                from models.lgb_cache import load_prediction_cache

                cache_preds, payload = load_prediction_cache()
                self._lgb_predictions = cache_preds
                self._lgb_status = {
                    "status": "ok",
                    "count": len(cache_preds),
                    "min_required": LGB_MIN_PREDICTIONS,
                    "source": "cache",
                    "latest_date": payload.get("latest_date", ""),
                    "error": "",
                    "fallback_reason": reason,
                }
                logger.warning(
                    "Live LGB inference failed (%s); using cached LGB predictions: %s stocks, latest_date=%s",
                    reason,
                    len(cache_preds),
                    payload.get("latest_date", ""),
                )
                return self._lgb_predictions
            except Exception as cache_exc:
                self._lgb_predictions = {}
                self._lgb_status = {
                    "status": "degraded",
                    "count": live_count,
                    "min_required": LGB_MIN_PREDICTIONS,
                    "source": "none",
                    "error": f"{reason}; cache unavailable: {cache_exc}",
                }
                logger.warning("Failed to load LGB model/cache: %s", self._lgb_status["error"])
                return self._lgb_predictions

        try:
            from models.short_term import ShortTermModel
            from models.lgb_cache import finite_prediction_map, write_prediction_cache

            model = ShortTermModel.load_from_pickle(
                str(LGB_MODEL_PATH)
            )
            preds = model.predict_batch()
            finite_preds = finite_prediction_map(preds)
            if len(finite_preds) < LGB_MIN_PREDICTIONS:
                return _use_cache(
                    f"finite LGB predictions {len(finite_preds)} "
                    f"< required {LGB_MIN_PREDICTIONS}",
                    len(finite_preds),
                )

            self._lgb_predictions = finite_preds
            self._lgb_status = {
                "status": "ok",
                "count": len(self._lgb_predictions),
                "min_required": LGB_MIN_PREDICTIONS,
                "source": "live",
                "latest_date": getattr(model, "latest_prediction_date", ""),
                "error": "",
            }
            try:
                write_prediction_cache(
                    self._lgb_predictions,
                    latest_date=getattr(model, "latest_prediction_date", ""),
                    model_path=str(LGB_MODEL_PATH),
                    source="scheduler_live",
                )
            except Exception as cache_exc:
                logger.warning("Failed to update LGB prediction cache: %s", cache_exc)
            logger.info(f"Loaded LGB predictions for {len(self._lgb_predictions)} stocks")
        except Exception as e:
            return _use_cache(str(e))
        return self._lgb_predictions

    def _load_capital_flow_signals(self) -> dict:
        """Load latest capital flow signals from parquet files.

        Returns dict: {qlib_code: {"net_mf": float, "net_mf_pct": float,
                                    "net_mf_5d": float, "nb_present": bool}}
        """
        if self._capital_flow_signals is not None:
            return self._capital_flow_signals

        signals = {}

        # --- Fund flow ---
        flow_path = DATA_DIR / "fund_flow_history.parquet"
        if flow_path.exists():
            try:
                df = pd.read_parquet(flow_path, columns=[
                    "qlib_code", "trade_date", "net_mf_amount",
                ])
                df = df.dropna(subset=["net_mf_amount"])
                df["trade_date"] = df["trade_date"].astype(str)
                # Latest 5 trading days per stock
                latest_dates = sorted(df["trade_date"].unique())[-5:]
                recent = df[df["trade_date"].isin(latest_dates)]

                for code, grp in recent.groupby("qlib_code"):
                    latest_row = grp.loc[grp["trade_date"].idxmax()]
                    net_mf = _finite_float(latest_row.get("net_mf_amount"))
                    net_mf_5d = _finite_float(grp["net_mf_amount"].sum())
                    signals[code] = {
                        "net_mf": net_mf,         # latest day main force net (万元)
                        "net_mf_5d": net_mf_5d,   # 5-day cumulative
                    }
                logger.info(f"Loaded capital flow signals for {len(signals)} stocks")
            except Exception as e:
                logger.warning(f"Failed to load fund flow signals: {e}")

        # --- Northbound ---
        nb_path = DATA_DIR / "northbound_history.parquet"
        if nb_path.exists():
            try:
                # Try to load holding quantity columns (from hk_hold API)
                want_cols = ["qlib_code", "trade_date"]
                # vol/ratio may come from hk_hold; 持股数量/占比 from akshare
                detail_cols = ["vol", "ratio", "持股数量", "持股数量占A股百分比"]
                nb_all_cols = pd.read_parquet(nb_path, columns=["qlib_code"]).columns  # dummy
                nb = pd.read_parquet(nb_path)
                nb = nb.dropna(subset=["trade_date"])
                nb["trade_date"] = nb["trade_date"].astype(str)
                latest_nb_dates = sorted(nb["trade_date"].unique())[-5:]
                recent_nb = nb[nb["trade_date"].isin(latest_nb_dates)]

                # Detect which holding-amount column is available
                hold_col = None
                for c in ["vol", "持股数量"]:
                    if c in recent_nb.columns and recent_nb[c].notna().sum() > 100:
                        hold_col = c
                        break
                ratio_col = None
                for c in ["ratio", "持股数量占A股百分比"]:
                    if c in recent_nb.columns and recent_nb[c].notna().sum() > 100:
                        ratio_col = c
                        break

                has_detail = hold_col is not None
                nb_stocks = set(recent_nb["qlib_code"].unique())
                for code in nb_stocks:
                    if code not in signals:
                        signals[code] = {}
                    signals[code]["nb_present"] = True
                    grp = recent_nb[recent_nb["qlib_code"] == code]
                    signals[code]["nb_days"] = grp["trade_date"].nunique()

                    if has_detail:
                        grp_sorted = grp.sort_values("trade_date")
                        vals = pd.to_numeric(grp_sorted[hold_col], errors="coerce")
                        if len(vals) >= 2 and vals.notna().sum() >= 2:
                            # 5-day change in holding quantity
                            signals[code]["nb_hold_change"] = float(
                                vals.iloc[-1] - vals.iloc[0])
                        if ratio_col:
                            ratios = pd.to_numeric(grp_sorted[ratio_col], errors="coerce")
                            if ratios.notna().any():
                                signals[code]["nb_hold_ratio"] = float(
                                    ratios.iloc[-1])

                detail_info = f", with holding detail ({hold_col})" if has_detail else " (presence only)"
                logger.info(f"Loaded northbound signals for {len(nb_stocks)} stocks{detail_info}")
            except Exception as e:
                logger.warning(f"Failed to load northbound signals: {e}")

        self._capital_flow_signals = signals
        return signals

    def _capital_flow_score(self, qlib_code: str) -> float:
        """Compute a [-1, 1] capital flow factor for a single stock.

        Positive = main force buying + northbound increasing holdings.
        """
        signals = self._load_capital_flow_signals()
        s = signals.get(qlib_code)
        if not s:
            return 0.0

        # Main force: normalize net_mf_5d (万元) to [-1, 1]
        net_mf_5d = _finite_float(s.get("net_mf_5d"))
        mf_score = max(-1.0, min(1.0, net_mf_5d / 200_000.0))

        # Northbound: use holding change if available, else presence bonus
        nb_hold_change = _finite_float(s.get("nb_hold_change"))
        if nb_hold_change != 0.0:
            # Normalize: 1M shares change → ±0.2 score
            nb_score = max(-0.3, min(0.3, nb_hold_change / 5_000_000.0))
        elif s.get("nb_present"):
            nb_score = 0.1
        else:
            nb_score = 0.0

        return round(max(-1.0, min(1.0, mf_score + nb_score)), 4)

    def _model_status_text(self):
        """Return a compact status block for pushed reports."""
        lgb_status = getattr(self, "_lgb_status", {"status": "unknown", "count": 0, "error": ""})
        if lgb_status.get("status") == "ok":
            date_text = (
                f"，数据日期{lgb_status.get('latest_date')}"
                if lgb_status.get("latest_date") else ""
            )
            model_line = (
                f"短线模型：正常，覆盖{lgb_status.get('count', 0)}只标的"
                f"{date_text}"
            )
        elif lgb_status.get("status") == "degraded":
            model_line = (
                "短线模型：降级，"
                f"有效覆盖{lgb_status.get('count', 0)}/"
                f"{lgb_status.get('min_required', LGB_MIN_PREDICTIONS)}，"
                "已改用全A因子量化分作为备选"
            )
        else:
            model_line = "短线模型：状态待确认，必要时使用全A因子量化分作为备选"
        return f"【数据状态】\n{model_line}"

    def _estimate_next_day_change_pct(self, candidate: dict) -> float:
        """Estimate next-trading-day stock return for short-term recommendations."""
        short_score = _finite_float(candidate.get("short_score"))
        intraday_change = _finite_float(candidate.get("change_pct"))

        if candidate.get("has_lgb"):
            expected = short_score * 100.0 / max(PREDICTION_HORIZON_DAYS, 1)
            expected = expected * 0.80 + intraday_change * 0.20
        else:
            expected = intraday_change * 0.35

        return round(max(-10.0, min(10.0, expected)), 2)

    def _stock_recommendations_only(self, recommendations: list) -> list:
        return [
            rec for rec in recommendations
            if is_dataclass(rec)
            if isinstance(getattr(rec, "code", ""), str)
            and rec.code[:2] in ("SH", "SZ", "BJ")
            and "多" in getattr(rec, "signal", "")
        ]

    def _classify_recommendations_by_horizon(
        self,
        recommendations: list,
        per_bucket: int = HORIZON_BUCKET_SIZE,
    ) -> dict[str, list]:
        """Build disjoint short/mid/long stock recommendation buckets."""
        bullish = self._stock_recommendations_only(recommendations)
        selected: set[str] = set()
        groups: dict[str, list] = {"短线": [], "中线": [], "长线": []}

        short_ranked = sorted(
            bullish,
            key=lambda rec: (
                _finite_float(getattr(rec, "next_day_change_pct", 0)),
                _finite_float(getattr(rec, "short_term_score", 0)),
                _finite_float(getattr(rec, "final_score", 0)),
            ),
            reverse=True,
        )
        for rec in short_ranked:
            if len(groups["短线"]) >= per_bucket:
                break
            if rec.code in selected:
                continue
            next_day = getattr(rec, "next_day_change_pct", None)
            groups["短线"].append(
                replace(
                    rec,
                    horizon="短线",
                    horizon_score=_finite_float(getattr(rec, "short_term_score", 0)),
                    next_day_change_pct=next_day,
                )
            )
            selected.add(rec.code)

        mid_ranked = sorted(
            bullish,
            key=lambda rec: (
                _finite_float(getattr(rec, "mid_term_score", 0)),
                _finite_float(getattr(rec, "final_score", 0)),
            ),
            reverse=True,
        )
        for rec in mid_ranked:
            if len(groups["中线"]) >= per_bucket:
                break
            if rec.code in selected:
                continue
            score = _finite_float(getattr(rec, "mid_term_score", 0))
            if score <= 0:
                score = _finite_float(getattr(rec, "final_score", 0))
            groups["中线"].append(replace(rec, horizon="中线", horizon_score=score))
            selected.add(rec.code)

        # Sentiment weight zeroed: SnowNLP has no backtest evidence.
        # Redistributed to final_score (model signal).
        # Will re-enable with validated contrarian overlay after 60d accumulation.
        long_ranked = sorted(
            bullish,
            key=lambda rec: (
                _finite_float(getattr(rec, "final_score", 0)) * 0.70
                + _finite_float(getattr(rec, "macro_score", 0)) * 0.30
                + _finite_float(getattr(rec, "sentiment_score", 0)) * 0.00
            ),
            reverse=True,
        )
        for rec in long_ranked:
            if len(groups["长线"]) >= per_bucket:
                break
            if rec.code in selected:
                continue
            score = (
                _finite_float(getattr(rec, "final_score", 0)) * 0.70
                + _finite_float(getattr(rec, "macro_score", 0)) * 0.30
                + _finite_float(getattr(rec, "sentiment_score", 0)) * 0.00
            )
            groups["长线"].append(replace(rec, horizon="长线", horizon_score=round(score, 2)))
            selected.add(rec.code)

        return groups

    def _flatten_horizon_recommendations(self, groups: dict[str, list]) -> list:
        """Flatten grouped recommendations while preserving bucket order."""
        flattened = []
        seen = set()
        for horizon in ("短线", "中线", "长线"):
            for rec in groups.get(horizon, []):
                if rec.code in seen:
                    continue
                flattened.append(rec)
                seen.add(rec.code)
        return flattened

    def _format_horizon_recommendations(self, groups: dict[str, list]) -> str:
        """Format grouped stock recommendations for deterministic push content."""
        lines = ["【长中短线分类推荐】"]
        specs = [
            ("短线", "短线（明日）", "明日预测"),
            ("中线", "中线（1-4周）", "中线评分"),
            ("长线", "长线（1-3月）", "长线评分"),
        ]
        for key, title, metric_label in specs:
            lines.append(f"\n{title}")
            items = groups.get(key, [])
            if not items:
                lines.append("暂无满足条件的标的")
                continue
            for i, rec in enumerate(items, 1):
                display_code = rec.code[2:] if rec.code[:2] in ("SH", "SZ", "BJ") else rec.code
                if key == "短线" and rec.next_day_change_pct is not None:
                    metric = f"{metric_label}{rec.next_day_change_pct:+.2f}%"
                else:
                    metric = f"{metric_label}{_finite_float(getattr(rec, 'horizon_score', 0)):+.2f}"
                score_display = round((rec.final_score + 1) * 5, 1)
                lines.append(
                    f"{i}. {rec.name}({display_code}) | {rec.signal} | "
                    f"{metric} | 综合{score_display}"
                )
                lines.append(f"   {rec.reason}")
        return "\n".join(lines)

    def _format_lgb_short_candidates(self, predictions: list[tuple[str, float]], limit: int = 5) -> str:
        """Format 22:00 short-term candidates with next-day return estimates."""
        lines = ["五、个股预测（明日短线候选）"]
        positive_predictions = [
            (code, score) for code, score in predictions
            if _finite_float(score) > 0
        ]
        if not positive_predictions:
            lines.append("暂无有效短线模型候选，明日个股层面先控制仓位，等待盘中确认。")
            return "\n".join(lines)

        for i, (code, score) in enumerate(positive_predictions[:limit], 1):
            expected = round(
                max(-10.0, min(10.0, _finite_float(score) * 100.0 / max(PREDICTION_HORIZON_DAYS, 1))),
                2,
            )
            display_code = code[2:] if code[:2] in ("SH", "SZ", "BJ") else code
            lines.append(
                f"{i}. {display_code}：模型分{score:+.4f}，明日预测{expected:+.2f}%"
            )
        return "\n".join(lines)

    def _qlib_code_from_spot_code(self, code_num) -> str:
        text = str(code_num).strip().zfill(6)
        if text.startswith(("6", "9")):
            return f"SH{text}"
        if text.startswith(("8", "4")):
            return f"BJ{text}"
        return f"SZ{text}"

    def _spot_lookup(self) -> dict:
        """Return qlib-code keyed spot rows without forcing callers to know the schema."""
        try:
            self.market_collector._load_spot_cache()
            spot = getattr(self.market_collector, "_spot_cache", None)
            if spot is None or spot.empty:
                return {}
            lookup = {}
            for _, row in spot.iterrows():
                code = self._qlib_code_from_spot_code(row.get("代码", ""))
                lookup[code] = row
            return lookup
        except Exception as e:
            logger.warning("Failed to load spot lookup for evening stock forecast: %s", e)
            return {}

    def _build_evening_stock_forecasts(self, lgb_preds: dict, limit: int = 10) -> dict[str, list[dict]]:
        """Build short/mid/long/composite stock forecast lists for evening report."""
        spot = self._spot_lookup()
        # Quote may be None when spot collector failed; require_quote=False
        # lets the ST/code rules still apply while skipping suspended/一字板.
        sanitizer = self._make_sanitizer(require_quote=False)
        rows = []
        if spot:
            universe = [(code, quote) for code, quote in spot.items()]
        else:
            universe = [(code, None) for code in lgb_preds]

        for code, quote in universe:
            name = str(quote.get("名称", "")) if quote is not None else ""
            ok, _reason = sanitizer.check(code, name, quote=quote)
            if not ok:
                continue
            has_lgb = code in lgb_preds
            change_pct = _finite_float(quote.get("涨跌幅")) if quote is not None else 0.0
            volume = _finite_float(quote.get("成交量")) if quote is not None else 0.0
            lgb_score = _finite_float(lgb_preds.get(code))
            model_score = lgb_score if has_lgb else self._fallback_quant_score(
                change_pct=change_pct,
                volume=volume,
                macro_score=0.0,
            )
            if model_score <= 0:
                continue
            price = _finite_float(quote.get("最新价")) if quote is not None else 0.0
            liquidity_score = min(volume / 1_000_000.0, 1.0)
            short_expected = round(
                max(-10.0, min(10.0, model_score * 100.0 / max(PREDICTION_HORIZON_DAYS, 1) * 0.80 + change_pct * 0.20)),
                2,
            )
            flow_score = self._capital_flow_score(code)
            mid_score = model_score * 0.65 + (change_pct / 100.0) * 0.15 + liquidity_score * 0.10 + flow_score * 0.10
            long_score = model_score * 0.40 + liquidity_score * 0.30 + max(change_pct, 0.0) / 100.0 * 0.15 + flow_score * 0.15
            composite_score = model_score * 0.50 + mid_score * 0.25 + long_score * 0.15 + flow_score * 0.10
            rows.append(
                {
                    "code": code,
                    "name": name,
                    "price": price,
                    "lgb_score": model_score,
                    "model_score": model_score,
                    "has_lgb": has_lgb,
                    "score_source": "ml_model" if has_lgb else "factor_fallback",
                    "change_pct": change_pct,
                    "short_expected": short_expected,
                    "mid_score": round(mid_score, 4),
                    "long_score": round(long_score, 4),
                    "composite_score": round(composite_score, 4),
                }
            )

        sanitizer.log_summary(label="evening_stock_forecasts")
        return {
            "短线": sorted(rows, key=lambda item: (item["has_lgb"], item["short_expected"], item["model_score"]), reverse=True)[:limit],
            "中线": sorted(rows, key=lambda item: (item["has_lgb"], item["mid_score"]), reverse=True)[:limit],
            "长线": sorted(rows, key=lambda item: (item["has_lgb"], item["long_score"]), reverse=True)[:limit],
            "综合": sorted(rows, key=lambda item: (item["has_lgb"], item["composite_score"]), reverse=True)[:limit],
        }

    def _format_evening_stock_forecasts(self, forecast_groups: dict[str, list[dict]]) -> str:
        """Format evening stock forecasts with trading strategy for each horizon."""
        lines = ["五、个股预测"]

        # Strategy specs per horizon
        strategy = {
            "短线": {"hold": "5个交易日", "buy": "下一开盘日", "tp": 8, "sl": 5},
            "中线": {"hold": "20个交易日", "buy": "回调时分批", "tp": 15, "sl": 8},
            "长线": {"hold": "60个交易日", "buy": "分3批建仓", "tp": 25, "sl": 10},
            "综合": {"hold": "5-20个交易日", "buy": "下一开盘日", "tp": 10, "sl": 6},
        }

        specs = [
            ("短线", "短线前十", "明日预测"),
            ("中线", "中线前十", "中线分"),
            ("长线", "长线观察榜前十（仅供参考，非长期持有建议）", "长线分"),
            ("综合", "综合前十", "综合分"),
        ]
        for key, title, metric_label in specs:
            strat = strategy[key]
            lines.append(f"{title}：")
            lines.append(f"  策略：持有{strat['hold']}｜{strat['buy']}买入｜止盈{strat['tp']}%｜止损{strat['sl']}%")
            items = forecast_groups.get(key, [])
            if not items:
                lines.append("  暂无有效候选")
                continue
            for i, item in enumerate(items, 1):
                display_code = item["code"][2:] if item["code"][:2] in ("SH", "SZ", "BJ") else item["code"]
                name = f"{item['name']} " if item.get("name") else ""
                price = f"¥{item['price']:.2f}" if item.get("price") and item["price"] > 0 else ""

                if key == "短线":
                    metric = f"预测{item['short_expected']:+.2f}%"
                elif key == "中线":
                    metric = f"中线分{item['mid_score']:+.4f}"
                elif key == "长线":
                    metric = f"长线分{item['long_score']:+.4f}"
                else:
                    metric = f"综合{item['composite_score']:+.4f}"

                lines.append(
                    f"  {i}. {name}{display_code} {price}｜{metric}｜"
                    f"模型{item['lgb_score']:+.4f}｜今日涨跌{item['change_pct']:+.2f}%"
                )
        return "\n".join(lines)

    def _flatten_stock_forecast_groups(self, forecast_groups: dict) -> list[dict]:
        """Flatten short/mid/long/composite forecast groups into unique rows."""
        if not isinstance(forecast_groups, dict):
            return []

        merged: dict[str, dict] = {}
        order: list[str] = []
        for horizon in ("短线", "中线", "长线", "综合"):
            items = forecast_groups.get(horizon, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                code = str(item.get("code", "")).upper().strip()
                if not code:
                    continue
                if code not in merged:
                    merged[code] = {**item, "code": code, "horizons": []}
                    order.append(code)
                merged[code]["horizons"].append(horizon)

        return [merged[code] for code in order]

    def _write_overnight_stock_snapshot(
        self,
        forecast_groups: dict,
        *,
        target_date: str,
    ) -> None:
        """Persist the 22:00 stock candidate pool for the next 9:20 correction."""
        items = self._flatten_stock_forecast_groups(forecast_groups)
        if not items:
            logger.info("Skip overnight stock snapshot: no stock candidates")
            return

        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_date": datetime.now().strftime("%Y-%m-%d"),
            "target_date": target_date,
            "lgb_status": getattr(self, "_lgb_status", {}),
            "groups": forecast_groups,
            "items": items,
        }
        try:
            path = OVERNIGHT_STOCK_SNAPSHOT_PATH
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(f"{path.name}.tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_path, path)
            logger.info(
                "Wrote overnight stock snapshot for %s: %s unique candidates",
                target_date,
                len(items),
            )
        except Exception as e:
            logger.warning("Failed to write overnight stock snapshot: %s", e)

    def _load_overnight_stock_snapshot(self, *, target_date: str) -> dict | None:
        """Load the 22:00 stock candidate pool when it matches the requested day."""
        path = OVERNIGHT_STOCK_SNAPSHOT_PATH
        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to read overnight stock snapshot: %s", e)
            return None

        if payload.get("target_date") != target_date:
            logger.info(
                "Ignoring overnight stock snapshot for target_date=%s while today=%s",
                payload.get("target_date"),
                target_date,
            )
            return None

        try:
            created_at = datetime.fromisoformat(str(payload.get("created_at", "")))
        except ValueError:
            logger.info("Ignoring overnight stock snapshot with invalid created_at")
            return None

        if datetime.now() - created_at > timedelta(days=4):
            logger.info("Ignoring stale overnight stock snapshot created_at=%s", created_at)
            return None

        items = payload.get("items")
        if not isinstance(items, list) or not items:
            items = self._flatten_stock_forecast_groups(payload.get("groups", {}))
            payload["items"] = items
        if not items:
            return None
        return payload

    def _make_sanitizer(self, *, require_quote: bool = True,
                        max_prediction_age_days: int = 3,
                        target_date: str | None = None) -> CandidateSanitizer:
        """Build a CandidateSanitizer for the current pipeline call.

        target_date: the date the pipeline is generating signals FOR. For
        live runs equals system today; for backfill / --date / shadow replay
        it can differ, and IPO-age / chain-stale / cooldown checks must use
        that date, not wall-clock today. Defaults to the pipeline's recorded
        target_date if set on the instance, otherwise system today.

        Per-call instance so reject reasons / counts are scoped to this
        recommendation cycle and logged in summary form. Sources loaded:
        - crash_predictions_latest.json    → high_crash_prob block
        - global_chain_factors.parquet     → chain_negative block (alpha < -2.0)
        - paper_shadow/risk_guard_state.json → in_cooldown block
        The OMS RiskGuard layers (crash + supply chain + cooldown) thus apply
        at recommendation time too, not just at OMS legacy execution.
        """
        today = (
            target_date
            or getattr(self, "_pipeline_target_date", None)
            or datetime.now().strftime("%Y-%m-%d")
        )
        crash_probs = self._load_crash_probs_for_sanitizer()
        chain_alpha = self._load_chain_alpha_for_sanitizer(today)
        cooldown_set = self._load_cooldown_for_sanitizer(today)
        return CandidateSanitizer(
            today=today,
            require_quote=require_quote,
            max_prediction_age_days=max_prediction_age_days,
            crash_probs=crash_probs,
            chain_alpha=chain_alpha,
            cooldown_set=cooldown_set,
        )

    def _load_chain_alpha_for_sanitizer(self, today: str) -> dict | None:
        """Read global_chain_factors.parquet for today's chain alpha."""
        from config.settings import DATA_DIR
        cached = getattr(self, "_chain_alpha_cache", None)
        if cached is not None:
            return cached
        path = DATA_DIR / "global_chain_factors.parquet"
        if not path.exists():
            self._chain_alpha_cache = None
            return None
        try:
            df = pd.read_parquet(path)
            if df.empty or "global_chain_alpha" not in df.columns:
                self._chain_alpha_cache = None
                return None
            dt = pd.Timestamp(today)
            dates = df.index.get_level_values("datetime")
            if dt in dates:
                snap = df.xs(dt, level="datetime")
            else:
                latest = dates.max()
                age = (dt - latest).days
                if age > 2:
                    logger.warning(
                        "Chain factors stale (%s, %d days) — skipping chain block in sanitizer",
                        latest.date(), age,
                    )
                    self._chain_alpha_cache = None
                    return None
                snap = df.xs(latest, level="datetime")
            alpha = snap["global_chain_alpha"]
            alpha.index = alpha.index.str.upper()
            out = {c: float(v) for c, v in alpha.items() if pd.notna(v)}
            self._chain_alpha_cache = out or None
            return self._chain_alpha_cache
        except Exception as e:
            logger.warning("Failed to load chain alpha for sanitizer: %s", e)
            self._chain_alpha_cache = None
            return None

    def _load_cooldown_for_sanitizer(self, today: str) -> set | None:
        """Read RiskGuard state files (champion + shadow) and return union
        of codes whose cooldown is still active on `today`."""
        from config.settings import DATA_DIR
        cached = getattr(self, "_cooldown_cache", None)
        if cached is not None:
            return cached
        codes: set[str] = set()
        for sub in ("paper", "paper_shadow"):
            state_path = DATA_DIR / sub / "risk_guard_state.json"
            if not state_path.exists():
                continue
            try:
                state = json.loads(state_path.read_text())
                for code, until in (state.get("cooldowns", {}) or {}).items():
                    if isinstance(until, str) and today < until:
                        codes.add(code.upper())
            except Exception as e:
                logger.warning("Failed to read %s for cooldown: %s", state_path, e)
        self._cooldown_cache = codes or None
        return self._cooldown_cache

    def _load_crash_probs_for_sanitizer(self) -> dict | None:
        """Read crash_predictions_latest.json and return {code_upper: prob}.

        Cached on the instance; staleness measured against the pipeline's
        target_date (set by run_daily_recommendation) so backfill / shadow
        replay correctly disables crash hard-block when the prediction file
        is from a different epoch than the signal date. Falls back to
        wall-clock today only when no target_date is recorded.
        """
        from config.settings import DATA_DIR
        cached = getattr(self, "_crash_probs_cache", None)
        if cached is not None:
            return cached
        path = DATA_DIR / "crash_predictions_latest.json"
        if not path.exists():
            self._crash_probs_cache = None
            return None
        try:
            payload = json.loads(path.read_text())
            pred_date = payload.get("date", "")
            try:
                pd_dt = datetime.strptime(str(pred_date)[:10], "%Y-%m-%d")
            except (ValueError, TypeError):
                logger.warning(
                    "crash_predictions_latest.json has unparseable date=%r — skipping crash hard-block",
                    pred_date,
                )
                self._crash_probs_cache = None
                return None
            ref = getattr(self, "_pipeline_target_date", None) or datetime.now().strftime("%Y-%m-%d")
            try:
                ref_dt = datetime.strptime(str(ref)[:10], "%Y-%m-%d")
            except (ValueError, TypeError):
                ref_dt = datetime.now()
            age = (ref_dt - pd_dt).days
            if age > 3:
                logger.warning(
                    "crash_predictions_latest.json is %d days stale (file=%s vs signal=%s) — skipping crash hard-block",
                    age, pred_date, ref_dt.strftime("%Y-%m-%d"),
                )
                self._crash_probs_cache = None
                return None
            preds = payload.get("predictions", {}) or {}
            self._crash_probs_cache = {str(k).upper(): float(v) for k, v in preds.items()}
            return self._crash_probs_cache
        except Exception as e:
            logger.warning("Failed to load crash predictions: %s", e)
            self._crash_probs_cache = None
            return None

    def _candidates_from_stock_snapshot(
        self,
        snapshot: dict,
        *,
        stock_macro: float,
    ) -> list[dict]:
        """Convert a persisted 22:00 stock snapshot into morning candidates."""
        candidates = []
        sanitizer = self._make_sanitizer(require_quote=False)
        for item in snapshot.get("items", []):
            if not isinstance(item, dict):
                continue
            code = str(item.get("code", "")).upper().strip()
            if code[:2] not in ("SH", "SZ", "BJ"):
                continue
            name = str(item.get("name") or code[-6:])
            ok, _reason = sanitizer.check(code, name)
            if not ok:
                continue
            short_score = _finite_float(item.get("model_score", item.get("lgb_score")))
            if short_score <= 0:
                continue
            has_lgb = bool(item.get("has_lgb", item.get("score_source") == "ml_model"))
            candidate = {
                "code": code,
                "name": name,
                "market": MARKET_STOCK,
                "short_score": short_score,
                "has_lgb": has_lgb,
                "score_source": item.get("score_source") or ("ml_model" if has_lgb else "factor_fallback"),
                "change_pct": _finite_float(item.get("change_pct")),
                "macro_score": _finite_float(stock_macro),
                "price": _finite_float(item.get("price")),
                "mid_score_hint": _finite_float(item.get("mid_score")),
                "long_score_hint": _finite_float(item.get("long_score")),
                "snapshot_horizons": item.get("horizons", []),
            }
            if item.get("short_expected") is not None:
                candidate["next_day_change_pct"] = _finite_float(item.get("short_expected"))
            else:
                candidate["next_day_change_pct"] = self._estimate_next_day_change_pct(candidate)
            candidates.append(candidate)

        sanitizer.log_summary(label="snapshot_candidates")
        return candidates

    def _build_stock_candidates(self, lgb_preds: dict, stock_macro: float) -> list[dict]:
        """Build A-share candidates with model-covered stocks preferred when available."""
        candidates = []
        sanitizer = self._make_sanitizer(require_quote=True)
        self.market_collector._load_spot_cache()
        spot = self.market_collector._spot_cache
        lgb_available = bool(lgb_preds) and getattr(self, "_lgb_status", {}).get("status") == "ok"

        if spot is None or spot.empty:
            logger.warning("Spot cache empty, falling back to watchlist")
            for code, name, market in WATCHLIST:
                if market != MARKET_STOCK:
                    continue
                quote = self._get_quote(code, market)
                if not quote:
                    continue
                ok, _reason = sanitizer.check(code, name, quote=quote)
                if not ok:
                    continue
                has_lgb = code in lgb_preds
                change_pct = _finite_float(quote.get("change_pct"))
                volume = _finite_float(quote.get("volume"))
                short_score = _finite_float(lgb_preds.get(code)) if has_lgb else self._fallback_quant_score(
                    change_pct=change_pct,
                    volume=volume,
                    macro_score=stock_macro,
                )
                if short_score <= 0:
                    continue
                candidate = {
                    "code": code, "name": name, "market": market,
                    "short_score": short_score,
                    "has_lgb": has_lgb,
                    "score_source": "ml_model" if has_lgb else "factor_fallback",
                    "change_pct": change_pct,
                    "macro_score": _finite_float(stock_macro),
                    "price": _finite_float(quote.get("price")),
                }
                candidate["next_day_change_pct"] = self._estimate_next_day_change_pct(candidate)
                candidates.append(candidate)
            return sorted(
                candidates,
                key=lambda c: (
                    bool(c.get("has_lgb")),
                    _finite_float(c.get("short_score")),
                    _finite_float(c.get("next_day_change_pct")),
                ),
                reverse=True,
            )

        for _, row in spot.iterrows():
            try:
                code_num = str(row["代码"])
                price = _finite_float(row["最新价"])
                change_pct = _finite_float(row["涨跌幅"])
                if price <= 0:
                    continue

                qlib_code = self._qlib_code_from_spot_code(code_num)
                name = str(row.get("名称", code_num))
                ok, _reason = sanitizer.check(qlib_code, name, quote=row.to_dict())
                if not ok:
                    continue
                has_lgb = qlib_code in lgb_preds

                short_score = _finite_float(lgb_preds.get(qlib_code)) if has_lgb else self._fallback_quant_score(
                    change_pct=change_pct,
                    volume=_finite_float(row.get("成交量")),
                    macro_score=stock_macro,
                )
                if short_score <= 0:
                    continue

                flow_score = self._capital_flow_score(qlib_code)
                candidate = {
                    "code": qlib_code,
                    "name": name,
                    "market": MARKET_STOCK,
                    "short_score": short_score,
                    "has_lgb": has_lgb,
                    "score_source": "ml_model" if has_lgb else "factor_fallback",
                    "change_pct": change_pct,
                    "macro_score": _finite_float(stock_macro),
                    "price": price,
                    "flow_score": flow_score,
                }
                candidate["next_day_change_pct"] = self._estimate_next_day_change_pct(candidate)
                candidates.append(candidate)
            except Exception:
                continue

        candidates.sort(
            key=lambda c: (
                bool(c.get("has_lgb")),
                _finite_float(c.get("short_score")),
                _finite_float(c.get("next_day_change_pct")),
                _finite_float(c.get("change_pct")),
            ),
            reverse=True,
        )
        logger.info(
            "Screened %s A-share candidates from spot (%s model scores, lgb_available=%s)",
            len(candidates),
            len(lgb_preds),
            lgb_available,
        )
        sanitizer.log_summary(label="build_stock_candidates")
        return candidates

    def _fallback_quant_score(self, *, change_pct: float, volume: float = 0.0, macro_score: float = 0.0) -> float:
        """Conservative all-A quantitative score when the ML score is unavailable."""
        momentum = max(-1.0, min(1.0, _finite_float(change_pct) / 10.0))
        liquidity = min(max(_finite_float(volume), 0.0) / 1_000_000.0, 1.0)
        liquidity_tilt = (liquidity - 0.5) * 0.10
        macro = max(-1.0, min(1.0, _finite_float(macro_score)))
        score = momentum * 0.80 + liquidity_tilt + macro * 0.10
        return round(max(-1.0, min(1.0, score)), 4)

    def _build_intraday_index_forecast(
        self,
        *,
        geo_factors: dict,
        lgb_preds: dict,
        crypto_data: dict,
        gold_data: dict,
        global_index_data: dict,
    ) -> tuple[object, str]:
        """Build the 14:30 next-open forecast for the requested A-share indices."""
        sorted_preds = sorted(lgb_preds.items(), key=lambda item: item[1], reverse=True)
        # Sanitize before slicing — index sentiment shouldn't be contaminated by
        # ST/BJ/suspended/一字板 tickers even though they're not user-pushed.
        spot = self._spot_lookup()
        sanitizer = self._make_sanitizer(require_quote=False)
        sanitized_preds = []
        for code, score in sorted_preds:
            quote = spot.get(code)  # pandas Series or None — `Series or {}` raises
            name = str(quote.get("名称", "")) if quote is not None else ""
            if sanitizer.check(code, name, quote=quote)[0]:
                sanitized_preds.append((code, score))
        sanitizer.log_summary(label="intraday_index_forecast")
        market_prediction = self.index_predictor.predict(
            global_indices=global_index_data,
            geo_factors=geo_factors,
            crypto_data=crypto_data,
            gold_data=gold_data,
            top_bullish=[{"code": code, "score": score} for code, score in sanitized_preds[:10]],
            top_bearish=[{"code": code, "score": score} for code, score in sanitized_preds[-5:]],
            lgb_status=getattr(self, "_lgb_status", {}),
        )
        segments = self.index_predictor.predict_a_share_segments(
            market_prediction,
            global_index_data,
            calibration=self._market_prediction_calibration(source="intraday_decision"),
            targets=[
                ("上证", "上证指数", 1.00),
                ("深证", "深证成指", 1.10),
                ("北证", "北证50", 1.30),
                ("创业板", "创业板指", 1.25),
            ],
        )
        title = f"一、下一开盘日指数预测（{market_prediction.target_date}：上证/深证/北证/创业板）"
        return market_prediction, self.index_predictor.format_segment_predictions(segments, title=title)

    def _market_prediction_calibration(self, source: str = None) -> dict:
        """Read recent verified forecast errors for online calibration."""
        try:
            calibration = self.verifier.get_market_prediction_calibration(source=source)
            if isinstance(calibration, dict):
                return calibration
        except Exception as e:
            logger.warning("Failed to load market prediction calibration: %s", e)
        return {}

    def _segment_prediction_record(
        self,
        segment: dict,
        base_prediction,
        *,
        source: str,
    ) -> dict:
        """Convert a formatted segment forecast into a verifier record."""
        return {
            "pred_date": base_prediction.pred_date,
            "target_date": base_prediction.target_date,
            "target_index": segment["index"],
            "direction": segment["direction"],
            "expected_change_pct": segment["expected_change_pct"],
            "lower_bound_pct": segment["lower_bound_pct"],
            "upper_bound_pct": segment["upper_bound_pct"],
            "up_probability": segment["up_probability"],
            "confidence": segment["confidence"],
            "source": source,
            "drivers": base_prediction.drivers,
            "risks": base_prediction.risks,
            "quote_change_pct": segment.get("quote_change_pct"),
            "data_status": {
                **base_prediction.data_status,
                "segment_status": segment.get("data_status", ""),
                "market": segment.get("market", ""),
                "calibration_bias_pct": segment.get("calibration_bias_pct", 0.0),
                "calibration_samples": segment.get("calibration_samples", 0),
            },
        }

    def _build_morning_final_index_forecast(
        self,
        *,
        geo_factors: dict,
        lgb_preds: dict,
        crypto_data: dict,
        gold_data: dict,
        global_index_data: dict,
    ) -> tuple[str, list[dict]]:
        """Build the 9:20 same-day final forecast for after-close comparison."""
        now = datetime.now()
        global_index_data = global_index_data if isinstance(global_index_data, dict) else {}
        sorted_preds = sorted(lgb_preds.items(), key=lambda item: item[1], reverse=True)
        spot = self._spot_lookup()
        sanitizer = self._make_sanitizer(require_quote=False)
        sanitized_preds = []
        for code, score in sorted_preds:
            quote = spot.get(code)
            name = str(quote.get("名称", "")) if quote is not None else ""
            if sanitizer.check(code, name, quote=quote)[0]:
                sanitized_preds.append((code, score))
        sanitizer.log_summary(label="morning_final_index_forecast")
        market_prediction = self.index_predictor.predict(
            as_of=now,
            target_date=now.strftime("%Y-%m-%d"),
            global_indices=global_index_data,
            geo_factors=geo_factors,
            crypto_data=crypto_data,
            gold_data=gold_data,
            top_bullish=[{"code": code, "score": score} for code, score in sanitized_preds[:10]],
            top_bearish=[{"code": code, "score": score} for code, score in sanitized_preds[-5:]],
            lgb_status=getattr(self, "_lgb_status", {}),
        )
        segments = self.index_predictor.predict_a_share_segments(
            market_prediction,
            global_index_data,
            calibration=self._market_prediction_calibration(source="morning_final"),
            targets=[
                ("上证", "上证指数", 1.00),
                ("深证", "深证成指", 1.10),
                ("北证", "北证50", 1.30),
                ("创业板", "创业板指", 1.25),
            ],
        )
        title = f"【9:20最近交易日收盘预测】（{market_prediction.target_date}：上证/深证/北证/创业板）"
        records = [
            self._segment_prediction_record(segment, market_prediction, source="morning_final")
            for segment in segments
        ]
        return self.index_predictor.format_segment_predictions(segments, title=title), records

    def _build_intraday_buy_candidates(self, lgb_preds: dict, limit: int = 10) -> list[dict]:
        """Build 14:30 strong-buy candidates from model scores plus live tape."""
        spot = self._spot_lookup()
        sanitizer = self._make_sanitizer(require_quote=True)
        rows = []
        for code, score in lgb_preds.items():
            short_score = _finite_float(score)
            if short_score <= 0:
                continue

            quote = spot.get(code)
            if quote is None:
                continue
            ok, _reason = sanitizer.check(code, str(quote.get("名称", "")), quote=quote)
            if not ok:
                continue

            change_pct = _finite_float(quote.get("涨跌幅"))
            price = _finite_float(quote.get("最新价"))
            volume = _finite_float(quote.get("成交量"))
            liquidity_score = min(volume / 1_000_000.0, 1.0)
            expected = self._estimate_next_day_change_pct(
                {
                    "short_score": short_score,
                    "change_pct": change_pct,
                    "has_lgb": True,
                }
            )
            if expected <= 0:
                continue

            strength = round(
                short_score * 0.70
                + max(change_pct, 0.0) / 100.0 * 0.20
                + liquidity_score * 0.10,
                4,
            )
            rows.append(
                {
                    "code": code,
                    "name": str(quote.get("名称", "")),
                    "price": price,
                    "change_pct": change_pct,
                    "expected_change_pct": expected,
                    "model_score": short_score,
                    "strength": strength,
                    "label": "强烈推荐" if expected >= 1.0 and short_score >= 0.04 else "重点关注",
                }
            )

        sanitizer.log_summary(label="intraday_buy_candidates")
        return sorted(
            rows,
            key=lambda item: (
                item["label"] == "强烈推荐",
                item["expected_change_pct"],
                item["strength"],
            ),
            reverse=True,
        )[:limit]

    def _format_intraday_buy_candidates(self, buy_items: list[dict]) -> str:
        """Format 14:30 buy candidates."""
        lines = ["二、14:30强买候选"]
        strong_items = [item for item in buy_items if item.get("label") == "强烈推荐"]
        if not buy_items:
            lines.append("暂无达到强买阈值的标的，尾盘不主动开新仓。")
            return "\n".join(lines)
        if not strong_items:
            lines.append("暂无强烈推荐买入，以下只列为重点观察。")

        for i, item in enumerate(buy_items, 1):
            display_code = item["code"][2:] if item["code"][:2] in ("SH", "SZ", "BJ") else item["code"]
            name = f"{item['name']} " if item.get("name") else ""
            price_text = f"现价{item['price']:.2f}，" if item.get("price") else ""
            lines.append(
                f"{i}. {item['label']}：{name}{display_code}，{price_text}"
                f"最近交易日{item['change_pct']:+.2f}%，下一开盘日预测{item['expected_change_pct']:+.2f}%，"
                f"强度{item['strength']:+.4f}"
            )
        return "\n".join(lines)

    def _build_sell_items(self, recent_recs: list[dict], lgb_preds: dict) -> list[dict]:
        """Build mandatory sell items from recent recommendation records."""
        sell_items = []
        for rec in recent_recs:
            code = rec["code"]
            rec_price = rec.get("price_at_rec")
            if not rec_price or rec_price <= 0:
                continue

            try:
                market = next((m for c, n, m in WATCHLIST if c == code), MARKET_STOCK)
                quote = self._get_quote(code, market)
                if not quote:
                    continue
                current_price = quote.get("price", 0)
                if current_price <= 0:
                    continue
            except Exception:
                continue

            gain_pct = (current_price - rec_price) / rec_price * 100
            reasons = []

            if gain_pct >= TAKE_PROFIT_PCT:
                reasons.append(f"止盈达标，涨{gain_pct:.1f}%")
            if gain_pct <= -STOP_LOSS_PCT:
                reasons.append(f"止损触发，跌{abs(gain_pct):.1f}%")

            lgb_score = _finite_float(lgb_preds.get(code, 0))
            if lgb_score < LGB_FLIP_THRESHOLD:
                reasons.append(f"短线模型翻空，模型分{lgb_score:.3f}")

            if reasons:
                sell_items.append(
                    {
                        "code": code,
                        "name": rec.get("name", code),
                        "reason": "；".join(reasons),
                        "gain_pct": gain_pct,
                        "rec_date": rec.get("rec_date", ""),
                        "current_price": current_price,
                    }
                )
        return sorted(sell_items, key=lambda item: abs(item["gain_pct"]), reverse=True)

    def _format_intraday_sell_items(self, sell_items: list[dict]) -> str:
        """Format mandatory sell list for the 14:30 report."""
        lines = ["三、历史推荐必卖清单"]
        if not sell_items:
            lines.append("暂无必须卖出的历史推荐，已有持仓继续按止盈/止损线观察。")
            return "\n".join(lines)

        for i, item in enumerate(sell_items, 1):
            gain = item["gain_pct"]
            sign = "+" if gain >= 0 else ""
            lines.append(
                f"{i}. 必须卖出：{item['name']}({item['code'][-6:]})，"
                f"推荐日{item['rec_date']}，当前收益{sign}{gain:.1f}%，"
                f"现价{item['current_price']:.2f}"
            )
            lines.append(f"   触发：{item['reason']}")
        return "\n".join(lines)

    def _format_intraday_decision_report(
        self,
        *,
        index_forecast_text: str,
        buy_items: list[dict],
        sell_items: list[dict],
    ) -> str:
        """Compose the 14:30 push report."""
        return _sanitize_push_text(
            "\n\n".join(
                [
                    f"【14:30盘中决策】{datetime.now().strftime('%Y-%m-%d')}",
                    index_forecast_text,
                    self._format_intraday_buy_candidates(buy_items),
                    self._format_intraday_sell_items(sell_items),
                    self._model_status_text(),
                ]
            )
        )

    def _record_intraday_buy_recommendations(self, buy_items: list[dict]) -> None:
        """Track 14:30 strong-buy recommendations for later sell checks."""
        today = datetime.now().strftime("%Y-%m-%d")
        for item in buy_items:
            if item.get("label") != "强烈推荐":
                continue
            self.verifier.record_recommendation(
                date_str=today,
                code=item["code"],
                name=item.get("name") or item["code"],
                signal="强烈看多",
                score=_finite_float(item.get("strength")),
                price_at_rec=_finite_float(item.get("price")),
            )

    def _pct_direction(self, value: float) -> str:
        value = _finite_float(value)
        if value >= 0.35:
            return "偏多"
        if value <= -0.35:
            return "偏空"
        return "震荡"

    def _format_gold_forecast(self, gold_data, geo_factors: dict) -> str:
        """Format concise gold forecast for evening report."""
        gold_data = gold_data or {}
        change = _finite_float(gold_data.get("change_pct"))
        price = _finite_float(gold_data.get("price"))
        safe_haven = _finite_float(geo_factors.get("safe_haven_signal"))
        policy = _finite_float(geo_factors.get("policy_signal"))
        expected = round(max(-3.0, min(3.0, change * 0.25 + (safe_haven - 0.5) * 0.85 + policy * 0.15)), 2)
        price_text = f"现价{price:,.1f}，" if price else ""
        logic = "避险需求抬升" if safe_haven >= 0.6 else "避险需求一般"
        return (
            "六、黄金预测\n"
            f"黄金：{self._pct_direction(expected)}，{price_text}最近交易日{change:+.2f}%，"
            f"明日参考{expected:+.2f}%附近。核心逻辑：{logic}，政策因子{policy:+.2f}。"
        )

    def _format_crypto_forecast(self, crypto_data, geo_factors: dict) -> str:
        """Format concise BTC/ETH forecast for evening report.

        Per quarantine §6.5 L6 and code-review I1: distinguish two
        distinct empty paths:
          (a) flag off  → "crypto context disabled" stub (quarantine intent)
          (b) flag on but no data fetched → "暂无实时数据" fallback
              (network failure / collector returned nothing)
        Conflating them previously caused flag-on+network-fail runs to
        emit the quarantine-off text, which is misleading.
        """
        from config.feature_flags import LEGACY_MARKET_CONTEXT_ENABLED
        crypto_data = crypto_data or {}
        if not LEGACY_MARKET_CONTEXT_ENABLED:
            return (
                "七、加密货币预测\n"
                "BTC/ETH：crypto context disabled（legacy quarantine off, "
                "见 §6.5）。"
            )
        if not crypto_data:
            return (
                "七、加密货币预测\n"
                "BTC/ETH：暂无实时数据，先按震荡处理。"
            )
        policy = _finite_float(geo_factors.get("policy_signal"))
        geo_risk = _finite_float(geo_factors.get("geo_risk_index"))
        safe_haven = _finite_float(geo_factors.get("safe_haven_signal"))
        lines = ["七、加密货币预测"]
        for symbol in ("BTC/USDT", "ETH/USDT"):
            data = crypto_data.get(symbol, {})
            name = "BTC" if "BTC" in symbol else "ETH"
            change = _finite_float(data.get("change_pct"))
            price = _finite_float(data.get("price"))
            expected = round(
                max(-5.0, min(5.0, change * 0.30 + policy * 0.25 + geo_risk * 0.10 - safe_haven * 0.20)),
                2,
            )
            price_text = f"${price:,.0f}，" if price else ""
            lines.append(
                f"{name}：{self._pct_direction(expected)}，{price_text}最近交易日{change:+.2f}%，"
                f"明日参考{expected:+.2f}%附近"
            )
        return "\n".join(lines)

    def _fallback_world_outlook(self, geo_factors: dict, headlines: list) -> str:
        """Fallback first-three-section narrative when LLM is unavailable."""
        key_events = geo_factors.get("key_events") or headlines[:3]
        event_text = "；".join(str(item) for item in key_events[:3]) if key_events else "夜间暂无明确主线事件"
        reasoning = geo_factors.get("reasoning") or {}
        market_reason = reasoning.get("market") or reasoning.get("geo_risk") or "外部风险与政策预期共同影响风险偏好"
        risk = _finite_float(geo_factors.get("geo_risk_index"))
        policy = _finite_float(geo_factors.get("policy_signal"))
        market = _finite_float(geo_factors.get("market_direction"))
        return (
            "一、世界大事\n"
            f"{event_text}。\n\n"
            "二、对世界格局的影响\n"
            f"当前地缘风险{risk:+.2f}，政策因子{policy:+.2f}。主线不是单点消息，而是风险偏好和流动性预期的再定价。\n\n"
            "三、对投资的影响\n"
            f"{market_reason}。A股方向因子{market:+.2f}，明日先看指数确认，再决定个股进攻力度。"
        )

    def _build_portfolio_risk_line(self) -> str:
        """Build portfolio risk status line for push reports."""
        try:
            import json as _json
            # Read backtest results for risk metrics
            bt_path = DATA_DIR / "lgb_backtest_latest.json"
            if not bt_path.exists():
                return ""
            bt = _json.loads(bt_path.read_text())
            m = bt.get("metrics", {})
            sharpe = m.get("sharpe_ratio", 0)
            maxdd = m.get("max_drawdown_pct", 0)
            turnover = m.get("avg_daily_turnover", 0)
            win = m.get("win_rate", 0)

            parts = [f"Sharpe={sharpe:.2f}"]
            parts.append(f"回撤={maxdd:.1f}%")
            parts.append(f"换手={turnover:.0%}")
            parts.append(f"胜率={win:.0%}")

            # Risk warnings
            warnings = []
            if maxdd < -15:
                warnings.append("回撤预警")
            if turnover > 0.3:
                warnings.append("换手偏高")

            line = f"风控：{'｜'.join(parts)}"
            if warnings:
                line += f" ⚠️{'、'.join(warnings)}"
            return line
        except Exception:
            return ""

    def _build_monster_radar(self, top_n: int = 10) -> str:
        """Build monster stock (妖股) radar section for push reports."""
        try:
            from data.collectors.limit_up import LimitUpCollector
            from factors.monster_stock import MonsterStockScorer

            collector = LimitUpCollector()
            scorer = MonsterStockScorer(limit_up_collector=collector)

            pool = collector.fetch_today_pool()
            if pool.empty:
                return ""

            # Score each limit-up stock
            scores = []
            for _, row in pool.iterrows():
                code = row.get("qlib_code", "")
                name = row.get("name", "")
                if not code:
                    continue
                ms = scorer.score(
                    code=code,
                    name=name,
                    sector_limit_up_count=0,  # TODO: compute from pool
                )
                scores.append(ms)

            if not scores:
                return ""

            # Sort by score, take top N
            scores.sort(key=lambda s: s.monster_score, reverse=True)
            top = [s for s in scores[:top_n] if s.risk_filter_passed]

            if not top:
                return ""

            lines = ["【妖股雷达】"]
            for i, s in enumerate(top, 1):
                boards = s.details.get("consecutive_boards", 0)
                board_str = f"{boards}连板" if boards > 0 else "首板"
                lines.append(
                    f"{i}. {s.name}({s.code[-6:]}) "
                    f"| {s.category} | {board_str} "
                    f"| 评分{s.monster_score:.2f}"
                )

            premium = scorer._board_premium or 0
            lines.append(f"涨停次日溢价率: {premium:+.1f}%")
            lines.append("风险提示: 妖股高风险，单只仓位≤5%")

            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Monster radar failed: {e}")
            return ""

    def _load_model_quality_line(self) -> str:
        """Load model quality + attribution + decay status for push reports."""
        lines = []
        try:
            import json as _json

            # Eval metrics
            eval_path = DATA_DIR / "lgb_eval_latest.json"
            if eval_path.exists():
                data = _json.loads(eval_path.read_text())
                m = data.get("metrics", {})
                quality = data.get("quality", "unknown")
                ic = m.get("ic_mean", 0)
                spread = m.get("top20_bot20_spread", m.get("top20_return_mean", 0))
                n_dates = data.get("n_dates", 0)
                label = {"normal": "正常", "marginal": "偏弱", "weak": "弱"}.get(quality, "未知")
                lines.append(f"模型：IC={ic:.3f}, Spread={spread*100:.2f}%, {n_dates}日, {label}")

            # Attribution
            attr_path = DATA_DIR / "lgb_attribution_latest.json"
            if attr_path.exists():
                attr = _json.loads(attr_path.read_text())
                if "error" not in attr:
                    alloc = attr.get("allocation_share", 0)
                    select = attr.get("selection_share", 0)
                    lines.append(f"归因：选股贡献{select:.0f}%｜行业配置{alloc:.0f}%")

            # Decay status
            decay_path = DATA_DIR / "factor_decay_status.json"
            if decay_path.exists():
                decay = _json.loads(decay_path.read_text())
                status = decay.get("status", "unknown")
                if status == "degraded":
                    lines.append("⚠️ 模型信号衰退，建议降低仓位")
                elif status == "warning":
                    lines.append("⚡ 模型信号减弱，注意风险")
        except Exception:
            pass

        return "\n".join(lines) if lines else ""

    def _format_evening_outlook_report(
        self,
        *,
        world_text: str,
        geo_factors: dict,
        headlines: list[str],
        a_share_forecast_text: str,
        stock_forecast_text: str,
        gold_forecast_text: str,
        crypto_forecast_text: str,
    ) -> str:
        """Compose the compact evening push in a fixed order."""
        world_text = (world_text or "").strip()
        if not world_text:
            world_text = self._fallback_world_outlook(geo_factors, headlines)

        model_quality = self._load_model_quality_line()

        monster_radar = self._build_monster_radar()

        sections = [
            f"【明日策略】{datetime.now().strftime('%Y-%m-%d')}",
            world_text,
            a_share_forecast_text,
            stock_forecast_text,
        ]
        if monster_radar:
            sections.append(monster_radar)
        sections.extend([
            gold_forecast_text,
            crypto_forecast_text,
        ])
        if model_quality:
            sections.append(model_quality)
        risk_line = self._build_portfolio_risk_line()
        if risk_line:
            sections.append(risk_line)

        return _sanitize_push_text("\n\n".join(sections))

    def _load_rl_agent(self):
        """Load RL agent for stock timing signals."""
        cached = getattr(self, "_rl_agent", None)
        if cached is not None:
            return cached

        try:
            from models.rl_agent import RLAgent
            if os.path.exists(str(RL_MODEL_PATH)):
                self._rl_agent = RLAgent(str(RL_MODEL_PATH))
                logger.info("RL agent loaded")
            else:
                logger.warning("RL model not found, skipping")
                from models.rl_agent import RLAgent as _RL
                self._rl_agent = _RL()  # empty agent, returns hold
        except Exception as e:
            logger.warning(f"Failed to load RL agent: {e}")
            from models.rl_agent import RLAgent
            self._rl_agent = RLAgent()
        return self._rl_agent

    def _load_mid_model(self):
        """Load mid-term model only when trained weights exist."""
        if getattr(self, "_mid_model_checked", False):
            return getattr(self, "_mid_model", None)

        self._mid_model_checked = True
        self._mid_model = None
        if not os.path.exists(str(MID_MODEL_PATH)):
            logger.info("Mid-term model not found; mid_score disabled")
            return None

        try:
            from models.mid_term import MidTermModel
            self._mid_model = MidTermModel(
                lookback_days=20,
                model_path=str(MID_MODEL_PATH),
            )
            logger.info("Mid-term model loaded")
        except Exception as e:
            logger.warning(f"Failed to load mid-term model: {e}")
            self._mid_model = None
        return self._mid_model

    def _fetch_geo_factors(self):
        """Fetch geopolitical factors via LLM analysis of global news (once per run)."""
        if self._geo_factors is not None:
            return self._geo_factors

        logger.info("Fetching global news from RSS...")
        all_news = self.macro_collector.fetch_all(max_per_source=15)
        headlines = [item.get("title", "") for item in all_news if item.get("title")]

        # MacroCollector returns [] in domestic profile (proxy unavailable). Fall back
        # to the daily global_industry_news collection (cron 16:25, --network global).
        if not headlines:
            from pathlib import Path
            from config.settings import DATA_DIR
            import json as _json_gn
            gn_dir = DATA_DIR / "global_industry_news"
            if gn_dir.exists():
                candidates = sorted(gn_dir.glob("*.jsonl"), reverse=True)[:3]
                for gn_path in candidates:
                    try:
                        items = []
                        with open(gn_path, encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    items.append(_json_gn.loads(line))
                                except _json_gn.JSONDecodeError:
                                    continue
                        gn_headlines = [it.get("title", "") for it in items if it.get("title")]
                        if gn_headlines:
                            headlines = gn_headlines[:120]
                            logger.info(
                                "MacroCollector returned 0; using cached global_industry_news %s (%d headlines)",
                                gn_path.name, len(headlines),
                            )
                            break
                    except Exception as e:
                        logger.warning("Failed to read cached global news %s: %s", gn_path.name, e)
            if not headlines:
                logger.warning(
                    "No global news available — geo factors will be default zeros (silent fallback)"
                )

        self._headlines = headlines
        logger.info(f"Fetched {len(headlines)} headlines from {len(all_news)} news items")

        # Use MiniMax LLM for deep geopolitical analysis
        logger.info("Running LLM geopolitical analysis...")
        self._geo_factors = self.llm_analyst.analyze_geopolitics(headlines)
        reasoning = self._geo_factors.get("reasoning", {})
        logger.info(f"Geo factors (LLM): {self._geo_factors}")
        if reasoning:
            for key, reason in reasoning.items():
                logger.info(f"  {key}: {reason}")
        return self._geo_factors

    def _fetch_global_indices(self):
        """Fetch global index quotes once and format them for reports."""
        data = self.global_indices.fetch_all()
        return data, self.global_indices.format_for_report(data)

    def _verify_market_predictions(self, index_quotes: dict) -> str:
        """Verify due 22:00 market forecasts after close."""
        try:
            verified = self.verifier.verify_due_market_predictions(index_quotes)
            return self.verifier.generate_market_prediction_report(verified)
        except Exception as e:
            logger.warning(f"Market prediction verification failed: {e}")
            return ""

    def _verify_morning_final_predictions(self, index_quotes: dict) -> str:
        """Verify today's 9:20 final index forecasts and explain misses."""
        try:
            verified = self.verifier.verify_due_market_prediction_snapshots(
                index_quotes,
                today=datetime.now().strftime("%Y-%m-%d"),
                source="morning_final",
                target_date=datetime.now().strftime("%Y-%m-%d"),
            )
            report = self.verifier.generate_morning_prediction_error_report(verified)
            if report:
                return report
            return (
                "📌 早盘最终预测复盘\n"
                "─────────────\n"
                "今天没有找到可验证的9:20结构化预测记录，无法做严格复盘。"
                "该记录链路已补齐，从下一次早盘推送开始会自动比对并分析误差主因。"
            )
        except Exception as e:
            logger.warning(f"Morning final prediction verification failed: {e}")
            return ""

    def run_daily_recommendation(self, use_overnight_snapshot: bool = False,
                                 target_date: str | None = None):
        """Run the full daily recommendation pipeline.

        Two-stage approach:
        1. Fast screen: reuse 22:00 candidates or rank model-covered stocks
        2. Deep analysis: sentiment + mid-term model for top-N candidates

        target_date overrides system today for backfill / shadow replay; all
        downstream sanitizers will pick it up via _pipeline_target_date so
        IPO-age / chain-stale / cooldown checks use the SIGNAL date, not
        wall-clock today.
        """
        logger.info("Starting daily recommendation pipeline...")
        today = target_date or datetime.now().strftime("%Y-%m-%d")
        self._pipeline_target_date = today
        # Per-call caches that depend on date — clear them so a second run
        # for a different date doesn't reuse the previous date's snapshot.
        self._crash_probs_cache = None
        self._chain_alpha_cache = None
        self._cooldown_cache = None
        self._geo_factors = None  # Reset cache
        self._headlines = None  # Reset cache
        self.market_collector.invalidate_cache()  # Fresh spot data

        # Fetch geo factors once
        geo = self._fetch_geo_factors()

        # Market index judgment
        market_judgment = self.market_judge.judge(
            geo_factors=geo,
        )
        logger.info(f"Market judgment: {market_judgment['direction']} ({market_judgment['reason']})")

        # === Stage 1: Fast screening stock candidates + crypto + gold ===
        logger.info("Stage 1: Screening stock candidates...")
        candidates = []

        # Load LGB predictions
        lgb_preds = self._load_lgb_predictions()

        stock_macro = (
            _finite_float(geo.get("china_us_temperature"))
            + _finite_float(geo.get("policy_signal"))
        ) / 2

        snapshot = None
        if use_overnight_snapshot:
            snapshot = self._load_overnight_stock_snapshot(target_date=today)

        if snapshot:
            candidates.extend(
                self._candidates_from_stock_snapshot(
                    snapshot,
                    stock_macro=stock_macro,
                )
            )
            logger.info(
                "Using overnight stock snapshot for %s: %s candidates",
                today,
                len(candidates),
            )
        else:
            candidates.extend(self._build_stock_candidates(lgb_preds, stock_macro))

        # Add crypto + gold. Crypto path is quarantine-gated; when
        # LEGACY_MARKET_CONTEXT_ENABLED is false (default), _get_crypto_collector()
        # returns None and BTC/ETH never enter the candidate pool.
        crypto_collector = self._get_crypto_collector()
        if crypto_collector is not None:
            for symbol in ["BTC/USDT", "ETH/USDT"]:
                q = crypto_collector.fetch_realtime(symbol)
                if q:
                    name = "比特币" if "BTC" in symbol else "以太坊"
                    candidates.append({
                        "code": symbol, "name": name, "market": MARKET_CRYPTO,
                        "short_score": _finite_float(q.get("change_pct")) / 10,
                        "has_lgb": False,
                        "change_pct": _finite_float(q.get("change_pct")),
                        "macro_score": _finite_float(geo.get("geo_risk_index")),
                        "price": _finite_float(q.get("price")),
                    })
        gold_q = self.gold_collector.fetch_realtime()
        if gold_q:
            candidates.append({
                "code": "AU", "name": "黄金", "market": MARKET_GOLD,
                "short_score": _finite_float(gold_q.get("change_pct")) / 10,
                "has_lgb": False,
                "change_pct": _finite_float(gold_q.get("change_pct")),
                "macro_score": _finite_float(geo.get("safe_haven_signal")) * 2 - 1,
                "price": _finite_float(gold_q.get("price")),
            })

        # Keep the production stock pool aligned with the 22:00 model-ranked pool.
        candidates.sort(
            key=lambda c: (
                bool(c.get("has_lgb")),
                _finite_float(c.get("short_score")),
                _finite_float(c.get("next_day_change_pct")),
                _finite_float(c.get("change_pct")),
            ),
            reverse=True,
        )

        # Cross-sectional rank normalization for the signal_scorer input.
        # LGB raw scores are O(1e-3) (next-day return prediction), so the
        # downstream MID_THRESHOLD=0.3 / HIGH_THRESHOLD=0.7 thresholds in
        # signals.scorer would never be reached and every stock degraded to
        # "观望", emptying the horizon classifier. Rank-normalize within the
        # has_lgb subset so top-half stocks register as 看多 and the top
        # decile clears the 强烈看多 bar. Raw "short_score" is left intact —
        # the new "ranked_score" feeds signal_scorer; ranking-vs-magnitude
        # decoupling preserves the strength/expected calculations elsewhere
        # that depend on the raw scale.
        lgb_pool = [c for c in candidates if c.get("has_lgb")]
        n = len(lgb_pool)
        if n > 0:
            scored = sorted(
                lgb_pool,
                key=lambda c: _finite_float(c.get("short_score")),
                reverse=True,
            )
            for rank, cand in enumerate(scored):
                # rank 0 → +1, rank N-1 → -1, linear interpolation
                normalized = 1.0 - 2.0 * (rank / max(n - 1, 1))
                cand["ranked_score"] = round(normalized, 4)
        # Non-LGB candidates (crypto / gold) keep raw short_score as ranked
        for cand in candidates:
            cand.setdefault("ranked_score", _finite_float(cand.get("short_score")))

        top_candidates = candidates[:SENTIMENT_TOP_N]
        logger.info(f"Stage 1 done: {len(candidates)} total, top {len(top_candidates)} selected for deep analysis")

        # === Stage 2: Deep analysis (sentiment + optional mid-term model) ===
        logger.info("Stage 2: Deep analysis with sentiment + optional mid-term model...")
        mid_model = self._load_mid_model()

        recommendations = []
        for cand in top_candidates:
            code, name, market = cand["code"], cand["name"], cand["market"]
            try:
                # Sentiment
                if market == MARKET_STOCK:
                    posts = self.sentiment_collector.fetch_all(code, limit_per_source=10)
                    sentiment = self.sentiment_scorer.score_batch(posts)
                else:
                    sentiment = {"sentiment_score": 0.0, "heat": 0.0, "post_count": 0}

                # Mid-term model
                mid_score = _finite_float(cand.get("mid_score_hint", 0.0))
                if mid_model is not None:
                    try:
                        df = self._get_daily(code, market, days=30)
                        if not df.empty and len(df) >= 20:
                            pred = mid_model.predict(df)
                            mid_score = _finite_float(pred.get("trend_score", 0.0))
                    except Exception:
                        pass

                display_name = f"[{self._market_label(market)}] {name}"
                rec = self.signal_scorer.score_stock(
                    code=code,
                    name=display_name,
                    # Use ranked_score (cross-sectional [-1, 1]) instead of raw
                    # LGB output so the 0.3/0.7 thresholds in signals.scorer
                    # can actually be reached. See Stage 1 ranking comment above.
                    model_score=_finite_float(cand.get("ranked_score", cand.get("short_score"))),
                    sentiment_score=_finite_float(sentiment["sentiment_score"]),
                    sentiment_heat=_finite_float(sentiment["heat"]),
                    mid_term_score=mid_score,
                    macro_score=_finite_float(cand["macro_score"]),
                )
                if market == MARKET_STOCK:
                    rec.next_day_change_pct = _finite_float(cand.get("next_day_change_pct"))
                recommendations.append(rec)

            except Exception as e:
                logger.error(f"Deep analysis failed for {code}: {e}")

        recommendations.sort(key=lambda r: r.final_score, reverse=True)
        horizon_groups = self._classify_recommendations_by_horizon(recommendations)
        horizon_block = self._format_horizon_recommendations(horizon_groups)
        top_recs = self._flatten_horizon_recommendations(horizon_groups)

        if not top_recs and recommendations:
            # Fallback: if horizon classification filtered everything,
            # use top 5 by final_score directly
            logger.warning(f"Horizon classification produced 0 recs from {len(recommendations)} candidates — using top 5 by score")
            top_recs = sorted(recommendations, key=lambda r: r.final_score, reverse=True)[:5]

        # Fetch global market data for report. Crypto path quarantined
        # behind LEGACY_MARKET_CONTEXT_ENABLED — when off (default), the
        # helper returns {} and crypto sections of the report degrade
        # gracefully.
        crypto_data = self._fetch_crypto_market_data()
        gold_data = self.gold_collector.fetch_realtime()
        global_index_data, global_indices = self._fetch_global_indices()
        morning_market_block, morning_market_records = self._build_morning_final_index_forecast(
            geo_factors=geo,
            lgb_preds=lgb_preds,
            crypto_data=crypto_data,
            gold_data=gold_data,
            global_index_data=global_index_data,
        )
        for prediction in morning_market_records:
            self.verifier.record_market_prediction(prediction)

        # Generate LLM-powered professional report (with fallback on timeout)
        logger.info("Generating LLM analyst report...")
        try:
            report = self.llm_analyst.generate_report(
                headlines=self._headlines or [],
                market_judgment=market_judgment,
                recommendations=top_recs,
                geo_factors=geo,
                crypto_data=crypto_data,
                gold_data=gold_data,
                global_indices_text=global_indices,
                horizon_recommendations_text=horizon_block,
            )
        except Exception as e:
            logger.warning(f"LLM report failed: {e}")
            report = ""

        # Fallback: if LLM failed, generate plain text report from candidates
        if not report or len(report.strip()) < 50:
            logger.warning("LLM report empty — using fallback plain text")
            lines = ["📋 AI分析报告生成失败，以下为模型直选结果：\n"]
            for i, rec in enumerate(top_recs[:20]):
                code = getattr(rec, "code", str(rec)) if not isinstance(rec, dict) else rec.get("code", "?")
                name = getattr(rec, "name", "") if not isinstance(rec, dict) else rec.get("name", "")
                score = getattr(rec, "score", 0) if not isinstance(rec, dict) else rec.get("score", 0)
                lines.append(f"  {i+1}. {code} {name} (score={score:.3f})")
            report = "\n".join(lines)
        model_quality = self._load_model_quality_line()
        status_block = self._model_status_text()
        if model_quality:
            status_block = f"{status_block}\n{model_quality}"
        report = _sanitize_push_text(
            f"{status_block}\n\n{morning_market_block}\n\n{horizon_block}\n\n{report}"
        )

        success = self.pusher.send_recommendation(report)
        logger.info(f"Push {'success' if success else 'failed'}: {len(top_recs)} recommendations")

        # Record for verification
        for rec in top_recs:
            market = next((m for c, n, m in WATCHLIST if c == rec.code), MARKET_STOCK)
            quote = self._get_quote(rec.code, market)
            price = quote.get("price") if quote else None

            self.verifier.record_recommendation(
                date_str=today,
                code=rec.code,
                name=rec.name,
                signal=rec.signal,
                score=rec.final_score,
                price_at_rec=price,
            )

    def run_verification(self):
        """Check and verify due recommendations."""
        logger.info("Running verification check...")

        due = self.verifier.get_due_verifications()
        if not due:
            logger.info("No verifications due today")
            return

        dates_to_verify = set()

        for rec in due:
            try:
                code = rec["code"]
                market = next((m for c, n, m in WATCHLIST if c == code), MARKET_STOCK)
                df = self._get_daily(code, market, days=30)

                if df.empty:
                    continue

                df = df.sort_index()
                rec_dt = datetime.strptime(rec["rec_date"], "%Y-%m-%d")
                rec_window = df[df.index > rec_dt].head(PREDICTION_HORIZON_DAYS)
                if len(rec_window) < PREDICTION_HORIZON_DAYS:
                    logger.info(
                        f"{code} has only {len(rec_window)} post-recommendation "
                        f"bars; need {PREDICTION_HORIZON_DAYS}"
                    )
                    continue

                current_price = rec_window.iloc[-1]["close"]
                high_price = rec_window["high"].max()
                low_price = rec_window["low"].min()
                price_at_rec = rec.get("price_at_rec")
                if not price_at_rec:
                    prior = df[df.index <= rec_dt]
                    if prior.empty:
                        continue
                    price_at_rec = prior.iloc[-1]["close"]

                self.verifier.verify(
                    date_str=rec["rec_date"],
                    code=rec["code"],
                    price_at_rec=price_at_rec,
                    price_at_verify=current_price,
                    high_price=high_price,
                    low_price=low_price,
                )
                dates_to_verify.add(rec["rec_date"])

            except Exception as e:
                logger.error(f"Error verifying {rec['code']}: {e}")

        for rec_date in dates_to_verify:
            report = self.verifier.generate_verification_report(rec_date)
            if report:
                self.pusher.send_verification(report)
                logger.info(f"Verification report sent for {rec_date}")

    def run_risk_check(self):
        """Hourly risk check: detect abnormal events and push alerts."""
        logger.info("Running risk check...")

        # Fetch fresh geo factors
        self._geo_factors = None
        geo = self._fetch_geo_factors()

        # Check sentiment for watchlist stocks
        sentiment_by_stock = {}
        for code, name, market in WATCHLIST:
            if market != MARKET_STOCK:
                continue
            try:
                posts = self.sentiment_collector.fetch_all(code, limit_per_source=10)
                sentiment = self.sentiment_scorer.score_batch(posts)
                sentiment_by_stock[code] = {
                    "name": name,
                    "sentiment_score": sentiment["sentiment_score"],
                    "heat": sentiment["heat"],
                }
            except Exception as e:
                logger.warning(f"Sentiment fetch failed for {code}: {e}")

        # Run all risk checks
        alerts = self.risk_monitor.check_all(geo, sentiment_by_stock)

        if not alerts:
            logger.info("No risk alerts")
            return

        # Push each alert
        for alert in alerts:
            msg = self.signal_scorer.generate_alert_message(alert)
            success = self.pusher.send_alert(msg)
            logger.info(
                f"Risk alert {'sent' if success else 'failed'}: "
                f"{alert.severity} - {alert.trigger[:50]}"
            )

    # ── New 4-slot schedule methods ────────────────────────────────────

    def run_morning_recommendation(self):
        """9:20 AM: Pre-market recommendation push with LGB + RL signals."""
        logger.info("=== Morning Recommendation (9:20) ===")
        self._lgb_predictions = None  # refresh
        self._rl_agent = None
        self._geo_factors = None
        self._headlines = None
        self._capital_flow_signals = None
        self.run_daily_recommendation(use_overnight_snapshot=True)

    def run_spot_cache_warmup(self):
        """17:05: Warm the after-close full-market spot cache for later pushes."""
        logger.info("=== Spot Cache Warmup (17:05) ===")
        info = self.market_collector.warm_spot_cache()
        logger.info(
            "Spot cache warmed: %s stocks, source=%s, created_at=%s, path=%s",
            info.get("row_count", 0),
            info.get("source", "unknown"),
            info.get("created_at", ""),
            info.get("cache_path", ""),
        )
        return info

    def run_sell_check(self):
        """14:30: Push next-open index forecast, strong buys, and mandatory sells."""
        logger.info("=== Intraday Decision (14:30) ===")

        self._geo_factors = None
        self._headlines = None
        geo = self._fetch_geo_factors()
        lgb_preds = self._load_lgb_predictions()
        buy_items = self._build_intraday_buy_candidates(lgb_preds, limit=10)

        crypto_data = self._fetch_crypto_market_data()
        gold_data = self.gold_collector.fetch_realtime()
        global_index_data, _ = self._fetch_global_indices()
        _, index_forecast_text = self._build_intraday_index_forecast(
            geo_factors=geo,
            lgb_preds=lgb_preds,
            crypto_data=crypto_data,
            gold_data=gold_data,
            global_index_data=global_index_data,
        )

        recent_recs = self.verifier.get_recent_recommendations(days=20)
        sell_items = self._build_sell_items(recent_recs, lgb_preds)
        report = self._format_intraday_decision_report(
            index_forecast_text=index_forecast_text,
            buy_items=buy_items,
            sell_items=sell_items,
        )

        if hasattr(self.pusher, "send_intraday_decision"):
            success = self.pusher.send_intraday_decision(report)
        else:
            success = self.pusher.send_sell_check(report)
        logger.info(
            "Intraday decision push %s: %s buy candidates, %s mandatory sells",
            "success" if success else "failed",
            len(buy_items),
            len(sell_items),
        )
        if success:
            self._record_intraday_buy_recommendations(buy_items)

    def run_daily_summary(self):
        """15:30: Post-close daily summary with verification."""
        logger.info("=== Daily Summary (15:30) ===")

        # Run verification first
        self.run_verification()

        # Generate summary via LLM
        self._geo_factors = None
        geo = self._fetch_geo_factors()

        # Collect market data
        crypto_data = self._fetch_crypto_market_data()
        gold_data = self.gold_collector.fetch_realtime()
        global_index_data, global_indices = self._fetch_global_indices()
        morning_prediction_report = self._verify_morning_final_predictions(global_index_data)
        market_prediction_report = self._verify_market_predictions(global_index_data)

        prompt_data = {
            "headlines": self._headlines or [],
            "geo_factors": geo,
            "crypto_data": crypto_data,
            "gold_data": gold_data,
            "global_indices": global_indices,
        }

        report = self.llm_analyst.generate_summary(prompt_data)
        verification_sections = [
            section for section in [morning_prediction_report, market_prediction_report]
            if section
        ]
        if verification_sections:
            verification_text = "\n\n".join(verification_sections)
            report = f"{verification_text}\n\n{report}" if report else verification_text
        report = _sanitize_push_text(report)
        if report:
            self.pusher.send_daily_summary(report)
            logger.info("Daily summary pushed")
        else:
            logger.warning("Failed to generate daily summary")

    def run_evening_outlook(self):
        """22:00: Evening outlook for next trading day."""
        logger.info("=== Evening Outlook (22:00) ===")

        self._geo_factors = None
        self._lgb_predictions = None  # refresh in case model was retrained
        self._capital_flow_signals = None  # refresh with latest data
        geo = self._fetch_geo_factors()
        lgb_preds = self._load_lgb_predictions()

        # Top bullish and bearish from LGB — sanitized for ST/BJ/suspended
        sorted_preds = sorted(lgb_preds.items(), key=lambda x: x[1], reverse=True)
        spot = self._spot_lookup()
        idx_sanitizer = self._make_sanitizer(require_quote=False)
        sanitized_preds = []
        for code, score in sorted_preds:
            quote = spot.get(code)
            name = str(quote.get("名称", "")) if quote is not None else ""
            if idx_sanitizer.check(code, name, quote=quote)[0]:
                sanitized_preds.append((code, score))
        idx_sanitizer.log_summary(label="evening_index_top_bull_bear")
        top_bull = sanitized_preds[:10]
        top_bear = sanitized_preds[-5:]
        stock_forecast_groups = self._build_evening_stock_forecasts(lgb_preds, limit=10)
        stock_forecast_block = self._format_evening_stock_forecasts(stock_forecast_groups)

        crypto_data = self._fetch_crypto_market_data()
        gold_data = self.gold_collector.fetch_realtime()
        global_index_data, global_indices = self._fetch_global_indices()

        market_prediction = self.index_predictor.predict(
            global_indices=global_index_data,
            geo_factors=geo,
            crypto_data=crypto_data,
            gold_data=gold_data,
            top_bullish=[{"code": c, "score": s} for c, s in top_bull],
            top_bearish=[{"code": c, "score": s} for c, s in top_bear],
            lgb_status=getattr(self, "_lgb_status", {}),
        )
        self._write_overnight_stock_snapshot(
            stock_forecast_groups,
            target_date=market_prediction.target_date,
        )
        market_prediction_block = self.index_predictor.format_prediction(market_prediction)
        a_share_forecasts = self.index_predictor.predict_a_share_segments(
            market_prediction,
            global_index_data,
            calibration=self._market_prediction_calibration(source="evening_outlook"),
        )
        a_share_forecast_block = self.index_predictor.format_segment_predictions(a_share_forecasts)
        self.verifier.record_market_prediction(market_prediction.to_dict())
        gold_forecast_block = self._format_gold_forecast(gold_data, geo)
        crypto_forecast_block = self._format_crypto_forecast(crypto_data, geo)

        prompt_data = {
            "headlines": self._headlines or [],
            "geo_factors": geo,
            "top_bullish": [{"code": c, "score": s} for c, s in top_bull],
            "top_bearish": [{"code": c, "score": s} for c, s in top_bear],
            "crypto_data": crypto_data,
            "gold_data": gold_data,
            "global_indices": global_indices,
            "market_prediction": market_prediction.to_dict(),
            "market_prediction_text": market_prediction_block,
            "a_share_forecast_text": a_share_forecast_block,
            "short_candidates_text": stock_forecast_block,
            "gold_forecast_text": gold_forecast_block,
            "crypto_forecast_text": crypto_forecast_block,
        }

        world_text = self.llm_analyst.generate_outlook(prompt_data)
        report = self._format_evening_outlook_report(
            world_text=world_text,
            geo_factors=geo,
            headlines=self._headlines or [],
            a_share_forecast_text=a_share_forecast_block,
            stock_forecast_text=stock_forecast_block,
            gold_forecast_text=gold_forecast_block,
            crypto_forecast_text=crypto_forecast_block,
        )
        if report:
            self.pusher.send_evening_outlook(report)
            logger.info("Evening outlook pushed")
        else:
            logger.warning("Failed to generate evening outlook")
