import json
from datetime import datetime

import pandas as pd

import data.collectors.market as market_module
from data.collectors.market import MarketCollector


class FrozenAfterClose(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 5, 7, 21, 30, 0)


class FrozenIntraday(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 5, 7, 10, 10, 0)


def test_load_spot_cache_uses_after_close_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(market_module, "datetime", FrozenAfterClose)
    collector = MarketCollector(
        spot_cache_path=tmp_path / "spot.csv",
        spot_cache_meta_path=tmp_path / "spot.meta.json",
    )
    df = pd.DataFrame([
        {"代码": "600519", "名称": "贵州茅台", "最新价": 1800.0, "涨跌幅": 1.2, "成交量": 100, "最高": 1810, "最低": 1790},
    ])
    collector._write_spot_disk_cache(df, "test")

    def fail_fetch():
        raise AssertionError("network fetch should not be called")

    monkeypatch.setattr(market_module.ak, "stock_zh_a_spot_em", fail_fetch)

    collector._load_spot_cache()

    assert collector._spot_loaded is True
    assert collector._spot_cache.iloc[0]["代码"] == "600519"


def test_intraday_stale_disk_cache_refreshes_from_akshare(tmp_path, monkeypatch):
    monkeypatch.setattr(market_module, "datetime", FrozenIntraday)
    cache_path = tmp_path / "spot.csv"
    meta_path = tmp_path / "spot.meta.json"
    pd.DataFrame([
        {"代码": "600519", "名称": "old", "最新价": 1.0, "涨跌幅": 0.0, "成交量": 1, "最高": 1, "最低": 1},
    ]).to_csv(cache_path, index=False)
    meta_path.write_text(
        json.dumps({"created_at": "2026-05-07T10:00:00", "source": "test", "row_count": 1}),
        encoding="utf-8",
    )
    collector = MarketCollector(
        spot_cache_path=cache_path,
        spot_cache_meta_path=meta_path,
        spot_cache_ttl_seconds=60,
    )

    monkeypatch.setattr(
        market_module.ak,
        "stock_zh_a_spot_em",
        lambda: pd.DataFrame([
            {"代码": "600519", "名称": "new", "最新价": 2.0, "涨跌幅": 1.0, "成交量": 2, "最高": 2, "最低": 1},
        ]),
    )

    collector._load_spot_cache()

    assert collector._spot_cache.iloc[0]["名称"] == "new"


def test_warm_spot_cache_forces_reload_even_when_memory_cache_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(market_module, "datetime", FrozenAfterClose)
    collector = MarketCollector(
        spot_cache_path=tmp_path / "spot.csv",
        spot_cache_meta_path=tmp_path / "spot.meta.json",
    )
    collector._spot_cache = pd.DataFrame([
        {"代码": "600519", "名称": "old-memory", "最新价": 1.0, "涨跌幅": 0.0, "成交量": 1, "最高": 1, "最低": 1},
    ])
    collector._spot_loaded = True

    monkeypatch.setattr(
        market_module.ak,
        "stock_zh_a_spot_em",
        lambda: pd.DataFrame([
            {"代码": "600519", "名称": "after-close", "最新价": 2.0, "涨跌幅": 1.0, "成交量": 2, "最高": 2, "最低": 1},
        ]),
    )

    info = collector.warm_spot_cache()

    assert info["row_count"] == 1
    assert info["source"] == "akshare"
    assert collector._spot_cache.iloc[0]["名称"] == "after-close"
