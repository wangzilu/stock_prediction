import json

import pytest

from scheduler.job_status import run_with_status
from scripts.install_crontab import BEGIN_MARKER, END_MARKER, merge_crontab, render_block


def test_run_with_status_records_success(tmp_path):
    status_path = tmp_path / "job_status.json"

    result = run_with_status("unit_job", lambda: "ok", status_path=status_path)

    payload = json.loads(status_path.read_text())
    job = payload["jobs"]["unit_job"]
    assert result == "ok"
    assert job["status"] == "success"
    assert job["run_count"] == 1
    assert job["duration_seconds"] >= 0


def test_run_with_status_records_failure(tmp_path):
    status_path = tmp_path / "job_status.json"

    def fail():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        run_with_status("unit_job", fail, status_path=status_path)

    payload = json.loads(status_path.read_text())
    job = payload["jobs"]["unit_job"]
    assert job["status"] == "failed"
    assert "ValueError: boom" in job["error"]
    assert job["duration_seconds"] >= 0


def test_merge_crontab_replaces_legacy_project_lines():
    existing = "\n".join([
        "0 0 * * * /usr/bin/true",
        "50 9 * * 1-5 /Users/wangzilu/miniconda3/envs/tianshou/bin/python /Users/wangzilu/MyProjects/stockPrediction/main.py --run-now",
        "0 17 * * 1-5 /Users/wangzilu/miniconda3/envs/tianshou/bin/python /Users/wangzilu/MyProjects/stockPrediction/scripts/update_qlib_data.py",
    ])

    merged = merge_crontab(existing, python_bin="/tmp/python")

    assert "0 0 * * * /usr/bin/true" in merged
    assert "main.py --run-now" not in merged
    assert BEGIN_MARKER in merged
    assert END_MARKER in merged
    assert "--evening-outlook" in merged
    assert "--warm-spot-cache" in merged
    assert "5 17 * * 1-5" in merged
    assert "--lgb-smoke-check" not in merged
    assert "--universe all" in merged
    assert "--universe-source baostock" in merged
    assert "--min-lgb-data-instruments 4500" in merged
    assert "lgb_after_close_train" in merged
    assert "/tmp/python" in merged


def test_phase_a6_cron_renders_truthful_health_jobs():
    block = render_block(python_bin="/tmp/python")

    chain_extract = next(line for line in block.splitlines() if "global_chain_extract" in line)
    chain_factors = next(line for line in block.splitlines() if "global_chain_factors" in line)
    assert "--enforce-deps" in chain_extract
    assert "--dep-wait-seconds 3600" in chain_extract
    assert "--enforce-deps" in chain_factors
    assert "--dep-wait-seconds 3600" in chain_factors

    assert "fetch_st_daily_factors.py --days 60" in block
    assert "fetch_shareholder_data.py" in block
    assert "fetch_st_round3.py --only stk_holdernumber --force" in block
    assert "fetch_fundamental_features.py" in block
    assert "fetch_fundamental_quality.py" in block


def test_weekly_sources_do_not_block_daily_feature_cache_same_day():
    from scheduler.job_deps import JOB_DEPS

    deps = set(JOB_DEPS["feature_cache_rebuild"])
    assert "st_holder_number_update" not in deps
    assert "fundamental_update" not in deps
    assert "quality_update" not in deps
