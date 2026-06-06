import numpy as np

from scripts.check_qlib_data_health import check_qlib_dir
from scripts.update_qlib_data import (
    is_a_share_stock_code,
    numeric_to_bs_code,
    parse_args,
    repair_legacy_bins,
    _main_inner,
)


def _make_minimal_qlib_dir(tmp_path):
    qlib_dir = tmp_path / "cn_data"
    (qlib_dir / "calendars").mkdir(parents=True)
    (qlib_dir / "calendars" / "day.txt").write_text(
        "2026-05-01\n2026-05-04\n2026-05-05\n"
    )
    (qlib_dir / "instruments").mkdir()
    (qlib_dir / "instruments" / "csi300.txt").write_text(
        "sh600000\t2026-05-01\t2026-05-05\n"
    )
    (qlib_dir / "instruments" / "csi500.txt").write_text("")
    feature_dir = qlib_dir / "features" / "sh600000"
    feature_dir.mkdir(parents=True)
    return qlib_dir, feature_dir


def test_health_check_rejects_legacy_bins_with_nan_start(tmp_path):
    qlib_dir, feature_dir = _make_minimal_qlib_dir(tmp_path)
    for field in ["open", "high", "low", "close", "volume", "amount"]:
        np.array([np.nan, 1.0, 2.0], dtype="<f4").tofile(
            feature_dir / f"{field}.day.bin"
        )

    report = check_qlib_dir(qlib_dir, min_coverage=0.95)

    assert not report.ok
    assert report.malformed_bins


def test_repair_legacy_bins_converts_to_qlib_header_format(tmp_path):
    qlib_dir, feature_dir = _make_minimal_qlib_dir(tmp_path)
    calendar = ["2026-05-01", "2026-05-04", "2026-05-05"]
    for field in ["open", "high", "low", "close", "volume", "amount"]:
        np.array([np.nan, 1.0, 2.0], dtype="<f4").tofile(
            feature_dir / f"{field}.day.bin"
        )

    repaired = repair_legacy_bins(qlib_dir, calendar)
    report = check_qlib_dir(qlib_dir, min_coverage=0.95)
    close_bin = np.fromfile(feature_dir / "close.day.bin", dtype="<f4")

    assert repaired == 6
    assert report.ok
    assert close_bin[0] == 1.0


def test_health_check_rejects_tiny_universe_when_min_instruments_is_required(tmp_path):
    qlib_dir, feature_dir = _make_minimal_qlib_dir(tmp_path)
    for field in ["open", "high", "low", "close", "volume", "amount"]:
        np.array([0.0, 1.0, 2.0, 3.0], dtype="<f4").tofile(
            feature_dir / f"{field}.day.bin"
        )

    report = check_qlib_dir(
        qlib_dir,
        universe="csi300",
        min_coverage=0.95,
        min_instruments=2,
    )

    assert not report.ok
    assert "instrument count 1 < required 2" in report.errors[0]


def test_numeric_to_bs_code_keeps_beijing_prefix():
    assert numeric_to_bs_code("832000") == "bj.832000"
    assert numeric_to_bs_code("430047") == "bj.430047"
    assert numeric_to_bs_code("600519") == "sh.600519"
    assert numeric_to_bs_code("300750") == "sz.300750"


def test_is_a_share_stock_code_filters_indices_and_b_shares():
    assert is_a_share_stock_code("sh.600000")
    assert is_a_share_stock_code("sh.688001")
    assert is_a_share_stock_code("sz.000001")
    assert is_a_share_stock_code("sz.300001")
    assert is_a_share_stock_code("bj.832000")
    assert not is_a_share_stock_code("sh.000001")
    assert not is_a_share_stock_code("sz.399001")
    assert not is_a_share_stock_code("sh.900901")


def test_update_qlib_noop_writes_health(monkeypatch, tmp_path):
    qlib_dir, feature_dir = _make_minimal_qlib_dir(tmp_path)
    for field in ["open", "high", "low", "close", "volume", "amount"]:
        np.array([0.0, 1.0, 2.0, 3.0], dtype="<f4").tofile(
            feature_dir / f"{field}.day.bin"
        )

    wrote = {}

    def fake_write_health(source, status, date=None):
        wrote["source"] = source
        wrote["status"] = status

    monkeypatch.setattr("scripts.update_qlib_data.write_health", fake_write_health)
    monkeypatch.setattr("scripts.update_qlib_data.load_manifest", lambda path: {})
    monkeypatch.setattr(
        "scripts.update_qlib_data.get_update_universe",
        lambda **kwargs: type("Universe", (), {
            "codes": {"sh600000"},
            "groups": {},
            "source": "instruments",
        })(),
    )
    monkeypatch.setattr("scripts.update_qlib_data.build_start_dates", lambda **kwargs: {})
    monkeypatch.setattr("scripts.update_qlib_data.validate_qlib_health", lambda *a, **k: True)

    args = parse_args([
        "--qlib-dir", str(qlib_dir),
        "--manifest", str(tmp_path / "manifest.json"),
        "--skip-health-check",
        "--end-date", "2026-05-05",
    ])

    assert _main_inner(args) == 0
    assert wrote["source"] == "qlib_data_update"
    assert wrote["status"].success is True
    assert wrote["status"].latest_date == "2026-05-05"
    assert wrote["status"].extra["noop"] is True
