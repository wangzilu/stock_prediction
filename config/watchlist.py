WATCHLIST = [
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

def to_akshare_code(qlib_code: str) -> str:
    return qlib_code.lower()

def to_stock_code(qlib_code: str) -> str:
    return qlib_code[2:]
