import subprocess
import sys
from pathlib import Path

from qlib.contrib.data.handler import Alpha158

from config.qlib_runtime import init_qlib as _init_qlib
from config.settings import QLIB_PROVIDER_URI


def init_qlib():
    """Initialize Qlib with local A-share data."""
    _init_qlib(QLIB_PROVIDER_URI)


def get_alpha158_handler(
    start_time: str = "2020-01-01",
    end_time: str = "2026-05-01",
    instruments: str = "csi300",
):
    """Get Qlib Alpha158 data handler."""
    return Alpha158(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
    )


def prepare_qlib_data():
    """Download and prepare Qlib A-share data (one-time setup)."""
    data_dir = Path(QLIB_PROVIDER_URI)
    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"Qlib data already exists at {data_dir}")
        return

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "qlib.run.get_data",
            "qlib_data",
            "--target_dir", str(data_dir),
            "--region", "cn",
        ],
        check=True,
    )
    print(f"Qlib data downloaded to {data_dir}")
