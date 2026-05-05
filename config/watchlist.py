# Market types
MARKET_STOCK = "stock"
MARKET_CRYPTO = "crypto"
MARKET_GOLD = "gold"

# Crypto watchlist: (symbol, name)
WATCHLIST_CRYPTO = [
    ("BTC/USDT", "比特币"),
    ("ETH/USDT", "以太坊"),
]

# Gold
WATCHLIST_GOLD = [
    ("AU", "黄金"),
]


def load_csi300() -> list:
    """Dynamically load CSI300 constituents from AKShare.

    Returns:
        List of (qlib_code, name) tuples, e.g. [("SH600519", "贵州茅台"), ...]
    """
    try:
        import akshare as ak
        df = ak.index_stock_cons(symbol="000300")
        result = []
        for _, row in df.iterrows():
            code = str(row["品种代码"]).zfill(6)
            name = row["品种名称"]
            # Determine exchange prefix
            if code.startswith("6"):
                qlib_code = f"SH{code}"
            else:
                qlib_code = f"SZ{code}"
            result.append((qlib_code, name))
        return result
    except Exception:
        # Fallback to a small default list if network fails
        return [
            ("SH600519", "贵州茅台"),
            ("SH601318", "中国平安"),
            ("SZ000858", "五粮液"),
            ("SZ300750", "宁德时代"),
            ("SH600036", "招商银行"),
            ("SZ000001", "平安银行"),
            ("SH601012", "隆基绿能"),
            ("SZ002594", "比亚迪"),
            ("SH600276", "恒瑞医药"),
            ("SZ000333", "美的集团"),
        ]


# Load at import time (cached for session)
WATCHLIST_STOCK = load_csi300()

# Combined watchlist with market type: (code, name, market)
WATCHLIST = (
    [(code, name, MARKET_STOCK) for code, name in WATCHLIST_STOCK]
    + [(code, name, MARKET_CRYPTO) for code, name in WATCHLIST_CRYPTO]
    + [(code, name, MARKET_GOLD) for code, name in WATCHLIST_GOLD]
)

# Max stocks to collect sentiment for (top-N after initial screening)
SENTIMENT_TOP_N = 20


def to_akshare_code(qlib_code: str) -> str:
    """Convert Qlib code (SH600519) to AKShare code (sh600519)."""
    return qlib_code.lower()


def to_stock_code(qlib_code: str) -> str:
    """Convert Qlib code (SH600519) to pure numeric code (600519)."""
    return qlib_code[2:]
