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
                """INSERT OR REPLACE INTO recommendations
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
        """Get recommendations due for verification (5+ trading days old)."""
        if today is None:
            today = datetime.now().strftime("%Y-%m-%d")

        cutoff = (
            datetime.strptime(today, "%Y-%m-%d") - timedelta(days=7)
        ).strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM recommendations WHERE verified=0 AND rec_date <= ?",
                (cutoff,),
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
            lines.append(
                f"{i}. {rec['name']}({rec['code'][2:]}) | 推荐{rec['signal']}"
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
