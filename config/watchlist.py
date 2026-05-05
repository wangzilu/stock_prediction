# Market types
MARKET_STOCK = "stock"
MARKET_CRYPTO = "crypto"
MARKET_GOLD = "gold"

# A-share stock watchlist: (code, name, market)
WATCHLIST_STOCK = [
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

# Crypto watchlist: (symbol, name)
WATCHLIST_CRYPTO = [
    ("BTC/USDT", "比特币"),
    ("ETH/USDT", "以太坊"),
]

# Gold
WATCHLIST_GOLD = [
    ("AU", "黄金"),
]

# Combined watchlist with market type: (code, name, market)
WATCHLIST = (
    [(code, name, MARKET_STOCK) for code, name in WATCHLIST_STOCK]
    + [(code, name, MARKET_CRYPTO) for code, name in WATCHLIST_CRYPTO]
    + [(code, name, MARKET_GOLD) for code, name in WATCHLIST_GOLD]
)


def to_akshare_code(qlib_code: str) -> str:
    """Convert Qlib code (SH600519) to AKShare code (sh600519)."""
    return qlib_code.lower()


def to_stock_code(qlib_code: str) -> str:
    """Convert Qlib code (SH600519) to pure numeric code (600519)."""
    return qlib_code[2:]
