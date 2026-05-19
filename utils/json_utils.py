"""Unified JSON serializer for numpy/pandas types.

Usage:
    import json
    from utils.json_utils import json_default

    json.dump(data, f, default=json_default)
"""
import numpy as np
import pandas as pd


def json_default(obj):
    """Handle numpy/pandas types that json.dump can't serialize."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta,)):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
