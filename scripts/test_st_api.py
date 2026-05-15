"""Test which ST_CLIENT API endpoints work."""
import os
import sys
import requests
from pathlib import Path

for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import ST_TOKEN

URL = "http://111.229.164.2:8083/"
NO_PROXY = {"http": None, "https": None}

def test(endpoint, params, timeout=30):
    params["TOKEN"] = ST_TOKEN
    print(f"  {endpoint} (timeout={timeout}s) ...", end=" ", flush=True)
    try:
        r = requests.post(f"{URL}{endpoint}", data=params, timeout=timeout, proxies=NO_PROXY)
        if r.status_code != 200:
            print(f"HTTP {r.status_code}")
            return
        data = r.json()
        if isinstance(data, dict) and data.get("code") == 0:
            d = data.get("data", {})
            if isinstance(d, dict) and "items" in d:
                cols = d.get("fields", d.get("columns", []))
                print(f"OK: {len(d['items'])} rows, cols={cols[:6]}")
            else:
                print(f"OK: data={type(d)}")
        elif isinstance(data, list):
            print(f"OK: {len(data)} rows")
            if data:
                print(f"    keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'not dict'}")
        elif isinstance(data, dict):
            print(f"code={data.get('code')}, msg={data.get('msg', '')}")
        else:
            print(f"Unknown: {type(data)}")
    except requests.exceptions.Timeout:
        print(f"TIMEOUT ({timeout}s)")
    except Exception as e:
        print(f"FAIL: {e}")

print("=== Testing endpoints (large = 60s timeout) ===\n")

# 小数据量接口先测
test("margin", {"trade_date": "20260512"})
test("top_list", {"trade_date": "20260512"})
test("limit_list_d", {"trade_date": "20260512"})
test("moneyflow_hsgt", {"trade_date": "20260512"})

# 大数据量接口（全市场5000+只）用更长超时
test("daily_basic", {"trade_date": "20260512"}, timeout=60)
test("moneyflow", {"trade_date": "20260512"}, timeout=60)
test("stk_factor_pro", {"trade_date": "20260512"}, timeout=60)
test("margin_detail", {"trade_date": "20260512"}, timeout=60)

# 单只股票测试（数据量小）
print("\n=== Single stock test ===\n")
test("daily_basic", {"ts_code": "600519.SH", "start_date": "20260501", "end_date": "20260512"})
test("moneyflow", {"ts_code": "600519.SH", "start_date": "20260501", "end_date": "20260512"})
test("fina_indicator", {"ts_code": "600519.SH"})
test("stk_holdernumber", {"ts_code": "600519.SH"})

print("\nDone!")
