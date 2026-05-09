"""Monster stock (妖股) composite scorer.

Combines limit-up chains, volume anomaly, sector heat, float structure,
and sentiment to score 5x/10x candidate potential.

Output categories:
- 潜伏型: low attention, improving fundamentals → highest value early find
- 加速型: breakout + volume confirmation → entry signal
- 兑现型: already crowded, high risk → too late
- 排除型: hype without substance → avoid
"""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MonsterScore:
    code: str
    name: str
    monster_score: float
    category: str  # 潜伏型/加速型/兑现型/排除型
    limit_up_chain_score: float
    volume_anomaly_score: float
    sector_heat_score: float
    float_structure_score: float
    sentiment_spike_score: float
    risk_filter_passed: bool
    details: dict


class MonsterStockScorer:
    """Composite scorer for monster stock detection."""

    def __init__(self, limit_up_collector=None):
        self.limit_up = limit_up_collector
        self._consecutive_boards = None
        self._board_premium = None

    def _load_limit_up_data(self):
        if self._consecutive_boards is not None:
            return
        if self.limit_up is None:
            self._consecutive_boards = {}
            self._board_premium = 0.0
            return
        self._consecutive_boards = self.limit_up.get_consecutive_boards()
        self._board_premium = self.limit_up.compute_board_premium()

    def score(
        self,
        code: str,
        name: str,
        price_df: pd.DataFrame = None,
        spot_data: dict = None,
        sector_limit_up_count: int = 0,
        sentiment_mention_ratio: float = 1.0,
    ) -> MonsterScore:
        """Score a single stock for monster potential.

        Args:
            code: Qlib code (e.g. SH600519)
            name: Stock name
            price_df: Recent OHLCV DataFrame (20+ days)
            spot_data: Dict with 最新价, 涨跌幅, 成交量, 换手率, 流通市值 etc.
            sector_limit_up_count: Number of limit-ups in same sector last 5 days
            sentiment_mention_ratio: mentions_today / avg_mentions_7d
        """
        self._load_limit_up_data()

        # 1. Limit-up chain score
        boards = self._consecutive_boards.get(code, 0)
        limit_score = min(boards / 5.0, 1.0)

        # 2. Volume anomaly score
        vol_score = self._volume_anomaly(price_df)

        # 3. Sector heat score
        sector_score = min(sector_limit_up_count / 10.0, 1.0)

        # 4. Float structure score (small float = higher score)
        float_score = self._float_structure(spot_data)

        # 5. Sentiment spike score
        sent_score = min(sentiment_mention_ratio / 10.0, 1.0)

        # Composite score
        monster_score = (
            0.25 * limit_score
            + 0.20 * vol_score
            + 0.20 * sector_score
            + 0.15 * float_score
            + 0.10 * sent_score
            + 0.10 * min(boards > 0, 1.0)  # bonus for any limit-up
        )

        # Risk filter
        risk_passed = self._risk_filter(code, boards, price_df, spot_data)

        # Category
        category = self._categorize(monster_score, boards, vol_score, risk_passed)

        return MonsterScore(
            code=code,
            name=name,
            monster_score=round(monster_score, 4),
            category=category,
            limit_up_chain_score=round(limit_score, 4),
            volume_anomaly_score=round(vol_score, 4),
            sector_heat_score=round(sector_score, 4),
            float_structure_score=round(float_score, 4),
            sentiment_spike_score=round(sent_score, 4),
            risk_filter_passed=risk_passed,
            details={
                "consecutive_boards": boards,
                "board_premium": round(self._board_premium or 0, 2),
                "sector_limit_up_count": sector_limit_up_count,
            },
        )

    def _volume_anomaly(self, price_df: pd.DataFrame) -> float:
        if price_df is None or price_df.empty or "volume" not in price_df.columns:
            return 0.0
        if len(price_df) < 5:
            return 0.0
        try:
            vol = price_df["volume"].values
            avg_20 = np.mean(vol[-20:]) if len(vol) >= 20 else np.mean(vol)
            latest = vol[-1]
            if avg_20 <= 0:
                return 0.0
            ratio = latest / avg_20
            return float(min(ratio / 5.0, 1.0))
        except Exception:
            return 0.0

    def _float_structure(self, spot_data: dict) -> float:
        if not spot_data:
            return 0.5
        try:
            # 流通市值 in 亿
            float_mcap = float(spot_data.get("流通市值", 0)) / 1e8
            if float_mcap <= 0:
                return 0.5
            if float_mcap < 30:  # < 30亿
                return 1.0
            elif float_mcap < 50:  # < 50亿
                return 0.7
            elif float_mcap < 100:
                return 0.4
            else:
                return 0.1
        except Exception:
            return 0.5

    def _risk_filter(self, code: str, boards: int, price_df: pd.DataFrame, spot_data: dict) -> bool:
        # Rule 1: Don't chase after 5+ boards (unless 总龙头)
        if boards >= 5:
            return False

        # Rule 2: Turnover > 40% = distribution
        if spot_data:
            try:
                turnover = float(spot_data.get("换手率", 0))
                if turnover > 40:
                    return False
            except Exception:
                pass

        # Rule 3: Board premium too negative = risk-off market
        if self._board_premium is not None and self._board_premium < -2.0:
            return False

        return True

    def _categorize(self, score: float, boards: int, vol_score: float, risk_passed: bool) -> str:
        if not risk_passed:
            return "排除型"
        if boards == 0 and score > 0.3:
            return "潜伏型"
        elif boards >= 1 and vol_score > 0.3 and score > 0.4:
            return "加速型"
        elif boards >= 3 or score > 0.7:
            return "兑现型"
        else:
            return "潜伏型"
