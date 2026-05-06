import logging
from datetime import datetime

from data.collectors.market import MarketCollector
from data.collectors.crypto import CryptoCollector
from data.collectors.gold import GoldCollector
from data.collectors.sentiment import SentimentCollector
from data.collectors.macro import MacroCollector
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
        all_news = self.macro_collector.fetch_all(max_per_source=10)
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

        # === Stage 1: Fast screening (price-based, all stocks) ===
        logger.info(f"Stage 1: Screening {len(WATCHLIST)} instruments...")
        candidates = []

        for code, name, market in WATCHLIST:
            try:
                quote = self._get_quote(code, market)
                if not quote:
                    continue

                short_score = quote.get("change_pct", 0) / 10

                # Quick macro adjustment
                macro_score = 0.0
                if market == MARKET_GOLD:
                    macro_score = geo["safe_haven_signal"] * 2 - 1
                elif market == MARKET_STOCK:
                    macro_score = (geo["china_us_temperature"] + geo["policy_signal"]) / 2
                elif market == MARKET_CRYPTO:
                    macro_score = geo["geo_risk_index"]

                candidates.append({
                    "code": code,
                    "name": name,
                    "market": market,
                    "short_score": short_score,
                    "macro_score": macro_score,
                    "price": quote.get("price", 0),
                })
            except Exception as e:
                logger.warning(f"Screen failed for {code}: {e}")

        # Sort by absolute short_score (strongest movers first)
        candidates.sort(key=lambda c: abs(c["short_score"]), reverse=True)
        top_candidates = candidates[:SENTIMENT_TOP_N]
        logger.info(f"Stage 1 done: {len(candidates)} screened, top {len(top_candidates)} selected")

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

        # Fetch crypto and gold data for report
        crypto_data = {}
        for symbol in ["BTC/USDT", "ETH/USDT"]:
            q = self.crypto_collector.fetch_realtime(symbol)
            if q:
                crypto_data[symbol] = q
        gold_data = self.gold_collector.fetch_realtime()

        # Generate LLM-powered professional report
        logger.info("Generating LLM analyst report...")
        report = self.llm_analyst.generate_report(
            headlines=self._headlines or [],
            market_judgment=market_judgment,
            recommendations=top_recs,
            geo_factors=geo,
            crypto_data=crypto_data,
            gold_data=gold_data,
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
