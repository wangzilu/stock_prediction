import logging
from datetime import datetime

from data.collectors.market import MarketCollector
from data.collectors.sentiment import SentimentCollector
from factors.sentiment import SentimentScorer
from signals.scorer import SignalScorer
from push.wechat import WeChatPusher
from tracker.verifier import Verifier
from config.watchlist import WATCHLIST, to_akshare_code

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyPipeline:
    """Orchestrates the daily recommendation pipeline."""

    def __init__(self):
        self.market_collector = MarketCollector()
        self.sentiment_collector = SentimentCollector()
        self.sentiment_scorer = SentimentScorer()
        self.signal_scorer = SignalScorer()
        self.pusher = WeChatPusher()
        self.verifier = Verifier()

    def run_daily_recommendation(self):
        """Run the full daily recommendation pipeline at 14:00."""
        logger.info("Starting daily recommendation pipeline...")
        today = datetime.now().strftime("%Y-%m-%d")
        recommendations = []

        for qlib_code, name in WATCHLIST:
            try:
                ak_code = to_akshare_code(qlib_code)
                quote = self.market_collector.fetch_realtime(ak_code)
                if not quote:
                    logger.warning(f"No market data for {qlib_code}, skipping")
                    continue

                posts = self.sentiment_collector.fetch_all(qlib_code, limit_per_source=20)
                sentiment = self.sentiment_scorer.score_batch(posts)

                # MVP: use price change as model score proxy
                model_score = quote.get("change_pct", 0) / 10

                rec = self.signal_scorer.score_stock(
                    code=qlib_code,
                    name=name,
                    model_score=model_score,
                    sentiment_score=sentiment["sentiment_score"],
                    sentiment_heat=sentiment["heat"],
                )
                recommendations.append(rec)

            except Exception as e:
                logger.error(f"Error processing {qlib_code}: {e}")
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

        for rec in top_recs:
            ak_code = to_akshare_code(rec.code)
            quote = self.market_collector.fetch_realtime(ak_code)
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
                ak_code = to_akshare_code(rec["code"])
                df = self.market_collector.fetch_daily(ak_code, days=10)

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
