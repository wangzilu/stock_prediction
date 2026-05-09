import json

import pytest

from scheduler.job_status import run_with_status
from scripts.install_crontab import BEGIN_MARKER, END_MARKER, merge_crontab


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
