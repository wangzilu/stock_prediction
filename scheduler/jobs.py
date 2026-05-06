import logging
from datetime import datetime

from data.collectors.market import MarketCollector
from data.collectors.crypto import CryptoCollector
from data.collectors.gold import GoldCollector
from data.collectors.sentiment import SentimentCollector
from data.collectors.macro import MacroCollector
from data.collectors.global_indices import GlobalIndicesCollector
from factors.sentiment import SentimentScorer
from signals.scorer import SignalScorer
from signals.risk_monitor import RiskMonitor
from signals.market_judge import MarketJudge
from signals.llm_analyst import LLMAnalyst
from push.wechat import WeChatPusher
from tracker.verifier import Verifier
from config.watchlist import (
    WATCHLIST, MARKET_STOCK, MARKET_CRYPTO, MARKET_GOLD,
    SENTIMENT_TOP_N, to_akshare_code,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyPipeline:
    """Orchestrates the daily recommendation pipeline for all markets."""

    def __init__(self):
        self.market_collector = MarketCollector()
        self.crypto_collector = CryptoCollector()
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

        # Cached data (computed once per run)
        self._geo_factors = None
        self._headlines = None

    def _get_quote(self, code, market):
        """Get realtime quote based on market type."""
        if market == MARKET_STOCK:
            return self.market_collector.fetch_realtime(to_akshare_code(code))
        elif market == MARKET_CRYPTO:
            return self.crypto_collector.fetch_realtime(code)
        elif market == MARKET_GOLD:
            return self.gold_collector.fetch_realtime()
        return {}

    def _get_daily(self, code, market, days=10):
        """Get daily data based on market type."""
        if market == MARKET_STOCK:
            return self.market_collector.fetch_daily(to_akshare_code(code), days)
        elif market == MARKET_CRYPTO:
            return self.crypto_collector.fetch_daily(code, days)
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

    def _fetch_geo_factors(self):
        """Fetch geopolitical factors via LLM analysis of global news (once per run)."""
        if self._geo_factors is not None:
            return self._geo_factors

        logger.info("Fetching global news from RSS...")
        all_news = self.macro_collector.fetch_all(max_per_source=15)
        headlines = [item.get("title", "") for item in all_news if item.get("title")]
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

    def run_daily_recommendation(self):
        """Run the full daily recommendation pipeline.

        Two-stage approach for 300+ stocks:
        1. Fast screen: use price change only to rank all stocks
        2. Deep analysis: sentiment + mid-term model for top-N candidates
        """
        logger.info("Starting daily recommendation pipeline...")
        today = datetime.now().strftime("%Y-%m-%d")
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

        # === Stage 1: Fast screening ALL A-shares + crypto + gold ===
        logger.info("Stage 1: Screening all A-shares from spot cache...")
        candidates = []

        # Load full market spot data once (5800+ stocks)
        self.market_collector._load_spot_cache()
        spot = self.market_collector._spot_cache
        stock_macro = (geo["china_us_temperature"] + geo["policy_signal"]) / 2

        if spot is not None and not spot.empty:
            for _, row in spot.iterrows():
                try:
                    code_num = str(row["代码"])
                    price = float(row["最新价"]) if row["最新价"] else 0
                    change_pct = float(row["涨跌幅"]) if row["涨跌幅"] else 0
                    if price <= 0:
                        continue

                    prefix = "SH" if code_num.startswith("6") else "SZ"
                    qlib_code = f"{prefix}{code_num}"
                    name = str(row.get("名称", code_num))

                    candidates.append({
                        "code": qlib_code,
                        "name": name,
                        "market": MARKET_STOCK,
                        "short_score": change_pct / 10,
                        "macro_score": stock_macro,
                        "price": price,
                    })
                except Exception:
                    continue
            logger.info(f"Screened {len(candidates)} A-shares from spot cache")
        else:
            logger.warning("Spot cache empty, falling back to watchlist")
            for code, name, market in WATCHLIST:
                if market != MARKET_STOCK:
                    continue
                quote = self._get_quote(code, market)
                if quote:
                    candidates.append({
                        "code": code, "name": name, "market": market,
                        "short_score": quote.get("change_pct", 0) / 10,
                        "macro_score": stock_macro, "price": quote.get("price", 0),
                    })

        # Add crypto + gold
        for symbol in ["BTC/USDT", "ETH/USDT"]:
            q = self.crypto_collector.fetch_realtime(symbol)
            if q:
                name = "比特币" if "BTC" in symbol else "以太坊"
                candidates.append({
                    "code": symbol, "name": name, "market": MARKET_CRYPTO,
                    "short_score": q.get("change_pct", 0) / 10,
                    "macro_score": geo["geo_risk_index"], "price": q.get("price", 0),
                })
        gold_q = self.gold_collector.fetch_realtime()
        if gold_q:
            candidates.append({
                "code": "AU", "name": "黄金", "market": MARKET_GOLD,
                "short_score": gold_q.get("change_pct", 0) / 10,
                "macro_score": geo["safe_haven_signal"] * 2 - 1, "price": gold_q.get("price", 0),
            })

        # Sort by absolute short_score (strongest movers first)
        candidates.sort(key=lambda c: abs(c["short_score"]), reverse=True)
        top_candidates = candidates[:SENTIMENT_TOP_N]
        logger.info(f"Stage 1 done: {len(candidates)} total, top {len(top_candidates)} selected for deep analysis")

        # === Stage 2: Deep analysis (sentiment + mid-term model) ===
        logger.info("Stage 2: Deep analysis with sentiment + mid-term model...")
        from models.mid_term import MidTermModel
        mid_model = MidTermModel(lookback_days=20)

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
                mid_score = 0.0
                try:
                    df = self._get_daily(code, market, days=30)
                    if not df.empty and len(df) >= 20:
                        pred = mid_model.predict(df)
                        mid_score = pred.get("trend_score", 0.0)
                except Exception:
                    pass

                display_name = f"[{self._market_label(market)}] {name}"
                rec = self.signal_scorer.score_stock(
                    code=code,
                    name=display_name,
                    model_score=cand["short_score"],
                    sentiment_score=sentiment["sentiment_score"],
                    sentiment_heat=sentiment["heat"],
                    mid_term_score=mid_score,
                    macro_score=cand["macro_score"],
                )
                recommendations.append(rec)

            except Exception as e:
                logger.error(f"Deep analysis failed for {code}: {e}")

        recommendations.sort(key=lambda r: r.final_score, reverse=True)
        top_recs = [r for r in recommendations if r.signal in ("强烈看多", "看多")][:5]

        if not top_recs:
            top_recs = []  # LLM report will mention no signals

        # Fetch global market data for report
        crypto_data = {}
        for symbol in ["BTC/USDT", "ETH/USDT"]:
            q = self.crypto_collector.fetch_realtime(symbol)
            if q:
                crypto_data[symbol] = q
        gold_data = self.gold_collector.fetch_realtime()
        global_indices = self.global_indices.format_for_report()

        # Generate LLM-powered professional report
        logger.info("Generating LLM analyst report...")
        report = self.llm_analyst.generate_report(
            headlines=self._headlines or [],
            market_judgment=market_judgment,
            recommendations=top_recs,
            geo_factors=geo,
            crypto_data=crypto_data,
            gold_data=gold_data,
            global_indices_text=global_indices,
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
                df = self._get_daily(code, market, days=10)

                if df.empty:
                    continue

                current_price = df.iloc[-1]["close"]
                high_price = df["high"].max()
                low_price = df["low"].min()
                price_at_rec = rec.get("price_at_rec") or df.iloc[0]["close"]

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
