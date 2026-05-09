import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

from config.settings import DB_PATH, PREDICTION_HORIZON_DAYS


class Verifier:
    """Tracks recommendations and verifies results after 5 trading days."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rec_date TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    score REAL NOT NULL,
                    price_at_rec REAL,
                    price_at_verify REAL,
                    high_price REAL,
                    low_price REAL,
                    return_pct REAL,
                    max_drawdown_pct REAL,
                    is_correct INTEGER,
                    verified INTEGER DEFAULT 0,
                    verify_date TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(rec_date, code)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pred_date TEXT NOT NULL,
                    target_date TEXT NOT NULL,
                    target_index TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    expected_change_pct REAL NOT NULL,
                    lower_bound_pct REAL NOT NULL,
                    upper_bound_pct REAL NOT NULL,
                    up_probability REAL NOT NULL,
                    confidence REAL NOT NULL,
                    actual_change_pct REAL,
                    direction_correct INTEGER,
                    interval_hit INTEGER,
                    verified INTEGER DEFAULT 0,
                    verify_date TEXT,
                    source TEXT DEFAULT 'unknown',
                    payload_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(pred_date, target_date, target_index)
                )
            """)
            self._ensure_column(conn, "market_predictions", "source", "TEXT DEFAULT 'unknown'")

    def _ensure_column(self, conn, table: str, column: str, definition: str) -> None:
        """Add a column to existing SQLite tables when the schema evolves."""
        columns = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def record_recommendation(
        self,
        date_str: str,
        code: str,
        name: str,
        signal: str,
        score: float,
        price_at_rec: float = None,
    ):
        """Record a new recommendation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO recommendations
                   (rec_date, code, name, signal, score, price_at_rec)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (date_str, code, name, signal, score, price_at_rec),
            )

    def verify(
        self,
        date_str: str,
        code: str,
        price_at_rec: float,
        price_at_verify: float,
        high_price: float,
        low_price: float,
    ):
        """Verify a recommendation with actual results."""
        return_pct = round((price_at_verify - price_at_rec) / price_at_rec * 100, 2)
        max_drawdown_pct = round((low_price - price_at_rec) / price_at_rec * 100, 2)

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT signal FROM recommendations WHERE rec_date=? AND code=?",
                (date_str, code),
            ).fetchone()

            if row is None:
                return

            signal = row[0]
            is_correct = (
                ("多" in signal and return_pct > 0)
                or ("空" in signal and return_pct < 0)
            )

            conn.execute(
                """UPDATE recommendations SET
                   price_at_rec=?, price_at_verify=?, high_price=?, low_price=?,
                   return_pct=?, max_drawdown_pct=?, is_correct=?,
                   verified=1, verify_date=?
                   WHERE rec_date=? AND code=?""",
                (
                    price_at_rec, price_at_verify, high_price, low_price,
                    return_pct, max_drawdown_pct, int(is_correct),
                    datetime.now().strftime("%Y-%m-%d"),
                    date_str, code,
                ),
            )

    def get_pending_verifications(self) -> list:
        """Get all unverified recommendations."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM recommendations WHERE verified=0"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_due_verifications(self, today: str = None) -> list:
        """Get pending recommendations for trading-day verification.

        The scheduler counts actual market bars after rec_date before
        marking anything verified; this query intentionally avoids a
        calendar-day cutoff.
        """
        if today is None:
            today = datetime.now().strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM recommendations WHERE verified=0 AND rec_date < ?",
                (today,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_verified(self, date_str: str) -> list:
        """Get verified recommendations for a specific date."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM recommendations WHERE rec_date=? AND verified=1",
                (date_str,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_recommendations(self, days: int = 5) -> list:
        """Get recommendations from the last N days that haven't been verified."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT rec_date, code, name, signal, score, price_at_rec
                   FROM recommendations
                   WHERE rec_date >= ? AND verified = 0
                   ORDER BY rec_date DESC""",
                (cutoff,),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_cumulative_stats(self) -> dict:
        """Get cumulative verification statistics."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) as total, SUM(is_correct) as correct "
                "FROM recommendations WHERE verified=1"
            ).fetchone()

            total = row[0] or 0
            correct = row[1] or 0
            win_rate = (correct / total * 100) if total > 0 else 0.0

            return {
                "total": total,
                "correct": int(correct),
                "win_rate": round(win_rate, 1),
            }

    def record_market_prediction(self, prediction: dict):
        """Record a structured next-day index prediction."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO market_predictions
                   (pred_date, target_date, target_index, direction,
                    expected_change_pct, lower_bound_pct, upper_bound_pct,
                    up_probability, confidence, source, payload_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    prediction["pred_date"],
                    prediction["target_date"],
                    prediction["target_index"],
                    prediction["direction"],
                    prediction["expected_change_pct"],
                    prediction["lower_bound_pct"],
                    prediction["upper_bound_pct"],
                    prediction["up_probability"],
                    prediction["confidence"],
                    prediction.get("source", "unknown"),
                    json.dumps(prediction, ensure_ascii=False),
                ),
            )

    def get_due_market_predictions(self, today: str = None, source: str = None) -> list:
        """Get unverified market predictions due by target date."""
        if today is None:
            today = datetime.now().strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sql = """SELECT * FROM market_predictions
                     WHERE verified=0 AND target_date <= ?"""
            params = [today]
            if source:
                sql += " AND source = ?"
                params.append(source)
            sql += " ORDER BY target_date ASC, pred_date ASC, target_index ASC"
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def verify_market_prediction(
        self,
        prediction_id: int,
        actual_change_pct: float,
        verify_date: str = None,
    ) -> dict:
        """Verify one market prediction with the actual index change."""
        verify_date = verify_date or datetime.now().strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM market_predictions WHERE id=?",
                (prediction_id,),
            ).fetchone()
            if row is None:
                return {}

            pred = dict(row)
            direction_correct = self._market_direction_correct(
                pred["direction"],
                actual_change_pct,
            )
            interval_hit = (
                pred["lower_bound_pct"] <= actual_change_pct <= pred["upper_bound_pct"]
            )
            conn.execute(
                """UPDATE market_predictions SET
                   actual_change_pct=?, direction_correct=?, interval_hit=?,
                   verified=1, verify_date=?
                   WHERE id=?""",
                (
                    round(actual_change_pct, 2),
                    int(direction_correct),
                    int(interval_hit),
                    verify_date,
                    prediction_id,
                ),
            )

            pred.update({
                "actual_change_pct": round(actual_change_pct, 2),
                "direction_correct": int(direction_correct),
                "interval_hit": int(interval_hit),
                "verified": 1,
                "verify_date": verify_date,
            })
            return pred

    def verify_due_market_predictions(
        self,
        index_quotes: dict,
        today: str = None,
        target_index: str = "沪深300",
        source: str = None,
    ) -> list:
        """Verify due index predictions using current index quote change_pct."""
        quote = index_quotes.get(target_index, {}) if index_quotes else {}
        if "change_pct" not in quote:
            return []

        actual_change_pct = float(quote["change_pct"])
        verified = []
        for pred in self.get_due_market_predictions(today=today, source=source):
            if pred["target_index"] != target_index:
                continue
            result = self.verify_market_prediction(pred["id"], actual_change_pct, today)
            if result:
                verified.append(result)
        return verified

    def verify_due_market_prediction_snapshots(
        self,
        index_quotes: dict,
        today: str = None,
        source: str = None,
        target_date: str = None,
    ) -> list:
        """Verify all due market predictions whose target_index is present in quotes."""
        verified = []
        index_quotes = index_quotes or {}
        for pred in self.get_due_market_predictions(today=today, source=source):
            if target_date and pred["target_date"] != target_date:
                continue
            quote = index_quotes.get(pred["target_index"], {})
            if "change_pct" not in quote:
                continue
            result = self.verify_market_prediction(
                pred["id"],
                float(quote["change_pct"]),
                today,
            )
            if result:
                verified.append(result)
        return verified

    def get_market_prediction_stats(self) -> dict:
        """Get cumulative stats for verified market predictions."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS total,
                          SUM(direction_correct) AS direction_correct,
                          SUM(interval_hit) AS interval_hit
                   FROM market_predictions WHERE verified=1"""
            ).fetchone()

        total = row[0] or 0
        correct = row[1] or 0
        interval_hit = row[2] or 0
        return {
            "total": total,
            "direction_accuracy": round(correct / total * 100, 1) if total else 0.0,
            "interval_hit_rate": round(interval_hit / total * 100, 1) if total else 0.0,
        }

    def get_market_prediction_calibration(
        self,
        source: str = None,
        lookback: int = 80,
        max_abs_bias: float = 0.80,
        shrink_samples: int = 5,
    ) -> dict:
        """Estimate recent per-index forecast bias from verified prediction errors.

        Bias is actual minus expected. A positive bias means recent forecasts
        underestimated the index and future forecasts should be nudged upward.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sql = """SELECT target_index, expected_change_pct, actual_change_pct
                     FROM market_predictions
                     WHERE verified=1 AND actual_change_pct IS NOT NULL"""
            params = []
            if source:
                sql += " AND source = ?"
                params.append(source)
            sql += " ORDER BY verify_date DESC, id DESC LIMIT ?"
            params.append(int(lookback))
            rows = conn.execute(sql, params).fetchall()

        grouped: dict[str, list[float]] = {}
        for row in rows:
            error = float(row["actual_change_pct"]) - float(row["expected_change_pct"])
            grouped.setdefault(row["target_index"], []).append(error)

        calibration = {}
        for target_index, errors in grouped.items():
            count = len(errors)
            mean_error = sum(errors) / count
            shrink = min(count / max(shrink_samples, 1), 1.0)
            bias = max(-max_abs_bias, min(max_abs_bias, mean_error * shrink * 0.50))
            calibration[target_index] = {
                "bias_pct": round(bias, 2),
                "mean_error_pct": round(mean_error, 2),
                "sample_count": count,
            }
        return calibration

    def generate_market_prediction_report(self, verified_predictions: list) -> str:
        """Generate a compact verification report for market forecasts."""
        if not verified_predictions:
            return ""

        stats = self.get_market_prediction_stats()
        lines = ["📊 大盘预测印证", "─────────────"]
        for pred in verified_predictions:
            direction_icon = "✅" if pred["direction_correct"] else "❌"
            interval_icon = "✅" if pred["interval_hit"] else "❌"
            lines.append(
                f"{pred['target_index']} {pred['target_date']} | "
                f"预测{pred['direction']} {pred['expected_change_pct']:+.2f}% "
                f"({pred['lower_bound_pct']:+.2f}%~{pred['upper_bound_pct']:+.2f}%)"
            )
            lines.append(
                f"实际{pred['actual_change_pct']:+.2f}% | "
                f"方向{direction_icon} 区间{interval_icon}"
            )

        lines.append("─────────────")
        lines.append(
            f"累计方向命中率：{stats['direction_accuracy']:.1f}% | "
            f"区间命中率：{stats['interval_hit_rate']:.1f}%"
        )
        return "\n".join(lines)

    def generate_morning_prediction_error_report(self, verified_predictions: list) -> str:
        """Generate after-close comparison for the 9:20 final index forecasts."""
        if not verified_predictions:
            return ""

        lines = ["📌 早盘最终预测复盘", "─────────────"]
        for pred in verified_predictions:
            direction_icon = "✅" if pred["direction_correct"] else "❌"
            interval_icon = "✅" if pred["interval_hit"] else "❌"
            error = round(pred["actual_change_pct"] - pred["expected_change_pct"], 2)
            lines.append(
                f"{pred['target_index']} {pred['target_date']} | "
                f"9:20预测{pred['direction']} {pred['expected_change_pct']:+.2f}% "
                f"({pred['lower_bound_pct']:+.2f}%~{pred['upper_bound_pct']:+.2f}%)"
            )
            lines.append(
                f"收盘实际{pred['actual_change_pct']:+.2f}% | "
                f"误差{error:+.2f}pct | 方向{direction_icon} 区间{interval_icon}"
            )
            lines.append(f"误差主因：{self._prediction_error_causes(pred)}")

        return "\n".join(lines)

    def _prediction_error_causes(self, pred: dict) -> str:
        """Explain likely error causes from stored forecast payload and result."""
        try:
            payload = json.loads(pred.get("payload_json") or "{}")
        except json.JSONDecodeError:
            payload = {}

        expected = float(pred.get("expected_change_pct") or 0)
        actual = float(pred.get("actual_change_pct") or 0)
        error = actual - expected
        confidence = float(pred.get("confidence") or 0)
        quote_change = payload.get("quote_change_pct")
        drivers = payload.get("drivers") or []
        risks = payload.get("risks") or []
        causes = []

        if abs(error) <= 0.20 and pred.get("interval_hit"):
            causes.append("预测与收盘基本一致，主要误差来自正常盘中波动")
        elif error > 0:
            causes.append("低估了上涨力度，盘中风险偏好或权重股承接强于早盘假设")
        else:
            causes.append("高估了上涨力度，盘中卖压或风险冲击强于早盘假设")

        if quote_change is not None:
            try:
                q = float(quote_change)
                if q * actual < 0:
                    causes.append("早盘/前收盘动量与全天收盘方向相反")
                elif abs(actual - q) > 0.8:
                    causes.append("盘中增量资金改变了早盘动量信号")
            except (TypeError, ValueError):
                pass

        if confidence < 0.50:
            causes.append("预测置信度偏低，数据覆盖或信号一致性不足")
        if risks:
            causes.append("早盘已提示风险：" + "；".join(str(item) for item in risks[:2]))
        elif drivers:
            causes.append("主要驱动假设：" + "；".join(str(item) for item in drivers[:2]))

        return "；".join(causes)

    def _market_direction_correct(self, direction: str, actual_change_pct: float) -> bool:
        """Evaluate market direction with a small neutral band."""
        if direction == "看涨":
            return actual_change_pct > 0
        if direction == "看跌":
            return actual_change_pct < 0
        return abs(actual_change_pct) <= 0.30

    def generate_verification_report(self, rec_date: str) -> str:
        """Generate formatted verification report for a recommendation date."""
        verified = self.get_verified(rec_date)
        if not verified:
            return ""

        today = datetime.now().strftime("%Y-%m-%d")
        stats = self.get_cumulative_stats()

        lines = [
            f"📋 荐股印证 (推荐日: {rec_date} → 今日: {today})",
            "─────────────",
        ]

        correct_count = 0
        total_count = len(verified)

        for i, rec in enumerate(verified, 1):
            result_icon = "✅" if rec["is_correct"] else "❌"
            code = rec['code']
            display_code = code[2:] if code[:2] in ("SH", "SZ") else code
            lines.append(
                f"{i}. {rec['name']}({display_code}) | 推荐{rec['signal']}"
            )
            lines.append(
                f"   结果：{rec['return_pct']:+.1f}% {result_icon} | "
                f"最高{rec['high_price']:.1f} | 最大回撤{rec['max_drawdown_pct']:.1f}%"
            )
            if rec["is_correct"]:
                correct_count += 1

        lines.append("─────────────")
        lines.append(
            f"本轮胜率：{correct_count}/{total_count} "
            f"({correct_count/total_count*100:.0f}%) | "
            f"累计胜率：{stats['win_rate']:.0f}%"
        )

        return "\n".join(lines)
