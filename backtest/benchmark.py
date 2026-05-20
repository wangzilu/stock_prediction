"""Benchmark return loader for excess return calculation.

Usage:
    from backtest.benchmark import load_benchmark_returns

    bm = load_benchmark_returns("sh000300", start="2023-01-01", end="2024-12-31")
    result = bt.run(predictions, returns, benchmark_returns=bm)
"""
import numpy as np
import pandas as pd


def load_benchmark_returns(
    benchmark: str = "sh000300",
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Load daily returns for a benchmark index from Qlib.

    Args:
        benchmark: Qlib instrument code for the benchmark.
            Common choices:
            - "sh000300": CSI 300 (沪深300)
            - "sh000905": CSI 500 (中证500)
            - "sh000852": CSI 1000 (中证1000)
            - "sh000001": SSE Composite (上证综指)
        start: Start date string, e.g. "2023-01-01". None = no lower bound.
        end: End date string, e.g. "2024-12-31". None = no upper bound.

    Returns:
        pd.Series indexed by datetime with daily return (close-to-close).
        Index is single-level datetime (no instrument level).
    """
    from qlib.data import D

    kwargs = {}
    if start is not None:
        kwargs["start_time"] = start
    if end is not None:
        kwargs["end_time"] = end

    ret = D.features(
        [benchmark],
        ["$close / Ref($close, 1) - 1"],
        **kwargs,
    )

    if ret is None or ret.empty:
        raise ValueError(
            f"No data returned for benchmark '{benchmark}'. "
            f"Check that the instrument exists in your Qlib data."
        )

    ret.columns = ["benchmark_return"]
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()

    # Drop the instrument level — benchmark is a single instrument
    # Qlib returns MultiIndex (datetime, instrument); flatten to datetime only
    if isinstance(ret.index, pd.MultiIndex):
        ret = ret.droplevel("instrument")

    return ret["benchmark_return"]
