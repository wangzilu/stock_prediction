import logging
from datetime import datetime

from data.collectors.market import MarketCollector
from data.collectors.crypto import CryptoCollector
from data.collectors.gold import GoldCollector
from data.collectors.sentiment import SentimentCollector
from factors.sentiment import SentimentScorer
from signals.scorer import SignalScorer
from push.wechat import WeChatPusher
from tracker.verifier import Verifier
from config.watchlist import (
    WATCHLIST, MARKET_STOCK, MARKET_CRYPTO, MARKET_GOLD,
    to_akshare_code,
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
        self.sentiment_scorer = SentimentScorer()
        self.signal_scorer = SignalScorer()
        self.pusher = WeChatPusher()
        self.verifier = Verifier()

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
        return __import__("pandas").DataFrame()

    def _market_label(self, market):
        """Get display label for market type."""
        return {
            MARKET_STOCK: "A股",
            MARKET_CRYPTO: "加密货币",
            MARKET_GOLD: "黄金",
        }.get(market, "")

    def run_daily_recommendation(self):
        """Run the full daily recommendation pipeline."""
        logger.info("Starting daily recommendation pipeline...")
        today = datetime.now().strftime("%Y-%m-%d")
        recommendations = []

        for code, name, market in WATCHLIST:
            try:
                quote = self._get_quote(code, market)
                if not quote:
                    logger.warning(f"No market data for {code} ({market}), skipping")
                    continue

                # Sentiment (only for stocks; crypto/gold use price-only signal)
                if market == MARKET_STOCK:
                    posts = self.sentiment_collector.fetch_all(code, limit_per_source=20)
                    sentiment = self.sentiment_scorer.score_batch(posts)
                else:
                    sentiment = {"sentiment_score": 0.0, "heat": 0.0, "post_count": 0}

                # MVP: use price change as model score proxy
                model_score = quote.get("change_pct", 0) / 10

                display_name = f"[{self._market_label(market)}] {name}"
                rec = self.signal_scorer.score_stock(
                    code=code,
                    name=display_name,
                    model_score=model_score,
                    sentiment_score=sentiment["sentiment_score"],
                    sentiment_heat=sentiment["heat"],
                )
                recommendations.append(rec)

            except Exception as e:
                logger.error(f"Error processing {code}: {e}")
                continue

        recommendations.sort(key=lambda r: r.final_score, reverse=True)
        top_recs = [r for r in recommendations if r.signal in ("强烈看多", "看多")][:5]

        if not top_recs:
            logger.info("No strong signals today, sending neutral report")
            self.pusher.send("📊 今日无明确推荐信号，建议观望")
            return

        report = self.signal_scorer.generate_report(top_recs)
        success = self.pusher.send_recommendation(report)
        logger.info(f"Push {'success' if success else 'failed'}: {len(top_recs)} recommendations")

        # Record for verification
        for rec in top_recs:
            # Find market type for this code
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
