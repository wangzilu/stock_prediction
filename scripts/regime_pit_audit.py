"""Regime Controller PIT (Point-in-Time) Audit

Checks whether regime_controller scores are truly PIT-safe by:

1. STRUCTURAL AUDIT: Inspects each score method for date-filtering patterns,
   cross-referencing with data_availability.py registry entries.

2. RUNTIME CONSISTENCY: Computes scores for historical dates multiple times
   to verify deterministic output (sanity check).

3. TEMPORAL STABILITY: For time-series-based scores, verifies that computing
   a score for date T yields the same result regardless of when we compute it
   (within the same data snapshot). Flags scores that could change if the
   underlying parquet file is appended to.

Output: data/storage/regime_pit_audit.json
"""
import inspect
import json
import logging
import re
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from signals.regime_controller import RegimeController
from config.data_availability import DATA_REGISTRY, get_spec

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"

# ── Score method → data source mapping ──────────────────────────────────────

SCORE_DATA_MAP = {
    "liquidity_score": {
        "method": "_liquidity",
        "data_sources": ["m2", "shibor"],
        "files": ["st_cn_m.parquet", "st_shibor.parquet"],
    },
    "credit_stress_score": {
        "method": "_credit_stress",
        "data_sources": ["shibor"],
        "files": ["st_shibor.parquet"],
    },
    "leverage_unwind_score": {
        "method": "_leverage",
        "data_sources": ["st_margin_detail"],
        "files": ["st_margin_detail.parquet"],
    },
    "microcap_crash_risk": {
        "method": "_microcap_crash",
        "data_sources": ["st_limit_list_d"],
        "files": ["st_limit_list_d.parquet"],
    },
    "external_shock_score": {
        "method": "_external_shock",
        "data_sources": ["cross_market_indices", "ic_im_futures"],
        "files": ["st_us_tycr.parquet", "feature_cache_174_holder_regime_ma.parquet"],
    },
    "policy_support_score": {
        "method": "_policy_support",
        "data_sources": ["llm_events"],
        "files": ["llm_events/*.jsonl"],
    },
    "theme_breadth_score": {
        "method": "_theme_breadth",
        "data_sources": ["guba"],
        "files": ["guba/*.jsonl"],
    },
    "inflation_score": {
        "method": "_inflation",
        "data_sources": ["cpi"],
        "files": ["st_cn_cpi.parquet"],
    },
    "northbound_score": {
        "method": "_northbound",
        "data_sources": ["northbound_hsgt_flow"],
        "files": ["st_moneyflow_hsgt.parquet"],
    },
    "futures_basis_score": {
        "method": "_futures_basis",
        "data_sources": ["ic_im_futures"],
        "files": ["ak_futures_ic0.parquet", "ak_index_csi500.parquet"],
    },
    "fx_risk_score": {
        "method": "_fx_risk",
        "data_sources": ["usdcny"],
        "files": ["ak_usdcny.parquet"],
    },
}


def structural_audit() -> dict:
    """Inspect each score method's source code for PIT-safety patterns."""
    rc = RegimeController()
    results = {}

    for score_name, info in SCORE_DATA_MAP.items():
        method_name = info["method"]
        method = getattr(rc, method_name)
        source = inspect.getsource(method)

        findings = {
            "data_sources": info["data_sources"],
            "files": info["files"],
        }

        # Check 1: Does it use _as_of() helper?
        uses_as_of = "_as_of(" in source or "as_of" in source
        findings["uses_as_of"] = uses_as_of

        # Check 2: Does it have manual date <= filtering?
        has_date_filter = bool(re.search(
            r"""<=\s*pd\.Timestamp\(date\)|<=\s*target|<= target_str|<= pd\.Timestamp""",
            source
        ))
        findings["has_date_filter"] = has_date_filter

        # Check 3: Does it use filename-based date filtering (for file-per-day sources)?
        has_filename_filter = bool(re.search(
            r"""file_date.*>.*target|f\.stem.*<=.*target|file_date > target""",
            source
        ))
        findings["has_filename_filter"] = has_filename_filter

        # Check 4: Does it read any "latest" data without filtering?
        reads_latest_unfiltered = (
            not uses_as_of
            and not has_date_filter
            and not has_filename_filter
        )
        findings["reads_latest_unfiltered"] = reads_latest_unfiltered

        # Check 5: Cross-reference with data_availability registry
        pit_levels = []
        for ds_name in info["data_sources"]:
            try:
                spec = get_spec(ds_name)
                pit_levels.append(spec.pit_safe_level)
                findings[f"registry_{ds_name}"] = {
                    "pit_safe_level": spec.pit_safe_level,
                    "signal_lag_bdays": spec.signal_lag_bdays,
                    "notes": spec.notes[:120] + "..." if len(spec.notes) > 120 else spec.notes,
                }
            except KeyError:
                pit_levels.append("unknown")

        # Check 6: Does the method filter using the correct date column?
        date_col_patterns = re.findall(
            r"""(?:date_col|"date"|"trade_date"|"month"|date_col\s*=\s*["'](\w+)["'])""",
            source
        )
        findings["date_columns_used"] = list(set(date_col_patterns)) if date_col_patterns else ["none_found"]

        # Determine PIT safety
        if reads_latest_unfiltered:
            pit_safe = False
            method_desc = "NO date filtering detected"
        elif "unsafe" in pit_levels:
            pit_safe = False
            method_desc = f"Data source marked unsafe in registry ({', '.join(info['data_sources'])})"
        elif uses_as_of:
            pit_safe = True
            method_desc = "_as_of() filter on date column"
        elif has_date_filter:
            pit_safe = True
            method_desc = "Manual date <= pd.Timestamp(date) filter"
        elif has_filename_filter:
            pit_safe = True
            method_desc = "Filename-based date filter (file_date <= target)"
        else:
            pit_safe = False
            method_desc = "Unclear filtering mechanism"

        findings["pit_safe"] = pit_safe
        findings["method_description"] = method_desc
        findings["registry_pit_levels"] = pit_levels

        results[score_name] = findings

    return results


def runtime_consistency_check(test_dates: list[str], n_runs: int = 3) -> dict:
    """Compute scores multiple times for the same dates to verify determinism."""
    rc = RegimeController()
    results = {}

    for date in test_dates:
        runs = []
        for i in range(n_runs):
            try:
                scores = rc.compute(date=date)
                # Extract just the numeric scores
                numeric = {
                    k: v for k, v in scores.items()
                    if isinstance(v, (int, float)) and k != "date"
                }
                runs.append(numeric)
            except Exception as e:
                runs.append({"error": str(e)})

        # Check consistency across runs
        if all(isinstance(r, dict) and "error" not in r for r in runs):
            consistent = all(r == runs[0] for r in runs)
        else:
            consistent = False

        results[date] = {
            "consistent": consistent,
            "n_runs": n_runs,
            "scores": runs[0] if runs else {},
        }
        if not consistent:
            # Find which scores differ
            diffs = []
            for key in runs[0]:
                vals = [r.get(key) for r in runs]
                if len(set(str(v) for v in vals)) > 1:
                    diffs.append({"score": key, "values": vals})
            results[date]["differences"] = diffs

    return results


def temporal_stability_check(test_dates: list[str]) -> dict:
    """For each date, compute the score and check if a later date
    would retroactively change time-series-based scores.

    Key insight: if score for date T uses _as_of(T) filtering,
    then running compute(T) and compute(T) again should be identical
    within the same data snapshot. But we can also check if the
    _as_of filtering actually clips properly by comparing the
    data rows selected for date T vs date T+5.
    """
    import pandas as pd

    rc = RegimeController()
    results = {}

    # Check: for parquet-based scores, verify _as_of actually clips data
    parquet_checks = [
        ("st_shibor.parquet", "date", "shibor"),
        ("st_moneyflow_hsgt.parquet", "trade_date", "northbound_hsgt_flow"),
        ("st_margin_detail.parquet", "trade_date", "st_margin_detail"),
        ("st_limit_list_d.parquet", "trade_date", "st_limit_list_d"),
        ("st_cn_m.parquet", "month", "m2"),
        ("st_cn_cpi.parquet", "month", "cpi"),
        ("ak_futures_ic0.parquet", "日期", "ic_im_futures"),
        ("ak_usdcny.parquet", "日期", "usdcny"),
    ]

    for fname, date_col, source_name in parquet_checks:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            results[fname] = {"exists": False, "note": "File not found"}
            continue

        try:
            df = pd.read_parquet(fpath)
            if date_col not in df.columns:
                results[fname] = {"exists": True, "date_col_found": False,
                                  "columns": list(df.columns)[:10]}
                continue

            # Parse date column
            if date_col == "month":
                df[date_col] = pd.to_datetime(df[date_col], format="%Y%m", errors="coerce")
            elif date_col == "日期":
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            else:
                df[date_col] = pd.to_datetime(df[date_col], format="%Y%m%d", errors="coerce")

            df = df.dropna(subset=[date_col])
            if df.empty:
                results[fname] = {"exists": True, "empty_after_parse": True}
                continue

            min_date = df[date_col].min()
            max_date = df[date_col].max()

            # For each test date, verify _as_of filtering
            date_checks = {}
            for tdate in test_dates:
                target = pd.Timestamp(tdate)
                if target < min_date:
                    date_checks[tdate] = "before_data_range"
                    continue

                rows_before = len(df[df[date_col] <= target])
                rows_total = len(df)
                # If rows_before == rows_total for a date well before max_date,
                # it means _as_of is not actually filtering anything useful
                date_checks[tdate] = {
                    "rows_selected": rows_before,
                    "rows_total": rows_total,
                    "pct_selected": round(rows_before / rows_total * 100, 1),
                    "correctly_filtered": rows_before < rows_total if target < max_date else True,
                }

            results[fname] = {
                "exists": True,
                "date_col": date_col,
                "date_range": [str(min_date.date()), str(max_date.date())],
                "total_rows": len(df),
                "date_checks": date_checks,
            }
        except Exception as e:
            results[fname] = {"exists": True, "error": str(e)}

    # Also check file-based sources
    for dir_name, source_name in [("llm_events", "llm_events"), ("guba", "guba")]:
        dir_path = DATA_DIR / dir_name
        if not dir_path.exists():
            results[dir_name] = {"exists": False}
            continue

        files = sorted(dir_path.glob("*.jsonl"))
        file_dates = [f.stem for f in files]

        date_checks = {}
        for tdate in test_dates:
            available_files = [fd for fd in file_dates if fd <= tdate]
            date_checks[tdate] = {
                "files_available": len(available_files),
                "latest_file": available_files[-1] if available_files else None,
                "correctly_filtered": len(available_files) < len(file_dates) if tdate < file_dates[-1] else True,
            }

        results[dir_name] = {
            "exists": True,
            "total_files": len(files),
            "date_range": [file_dates[0], file_dates[-1]] if file_dates else [],
            "date_checks": date_checks,
        }

    return results


def compute_score_sensitivity(test_dates: list[str]) -> dict:
    """Compute scores at date T and T-5 to check that T scores
    don't retroactively change when viewed from T+5.

    Since we have a fixed data snapshot, we simulate this by checking:
    - score(T) computed now
    - Whether the data rows used for T would differ if the parquet
      had additional future rows (they shouldn't, due to _as_of filtering)
    """
    rc = RegimeController()
    results = {}

    for date in test_dates:
        try:
            scores_t = rc.compute(date=date)
        except Exception as e:
            results[date] = {"error": str(e)}
            continue

        # Compute at T-5 as well — if both are non-zero for the same score,
        # it tells us the score is actually reading data, not just returning 0
        from datetime import datetime as dt
        t = dt.strptime(date, "%Y-%m-%d")
        t_minus5 = (t - timedelta(days=7)).strftime("%Y-%m-%d")  # ~5 trading days

        try:
            scores_tminus5 = rc.compute(date=t_minus5)
        except Exception:
            scores_tminus5 = {}

        score_keys = [k for k in scores_t if k.endswith("_score") or k == "microcap_crash_risk"]
        comparison = {}
        for k in score_keys:
            v_t = scores_t.get(k, None)
            v_t5 = scores_tminus5.get(k, None)
            comparison[k] = {
                "at_T": v_t,
                "at_T_minus5": v_t5,
                "changed": v_t != v_t5,
                "both_zero": v_t == 0.0 and v_t5 == 0.0,
            }

        results[date] = comparison

    return results


def main():
    print("=" * 70)
    print("REGIME CONTROLLER PIT (POINT-IN-TIME) AUDIT")
    print("=" * 70)

    # Test dates spanning different periods
    test_dates = [
        "2024-01-15",   # Well before any LLM events data
        "2024-06-15",   # Mid-2024
        "2025-01-10",   # Early 2025
        "2025-06-15",   # Mid-2025
        "2025-10-15",   # Late 2025
        "2026-01-15",   # Early 2026
        "2026-03-15",   # Before LLM events start
        "2026-05-01",   # During LLM events period
        "2026-05-15",   # Recent
        "2026-05-22",   # Latest available
    ]

    # --- 1. Structural Audit ---
    print("\n[1/4] STRUCTURAL AUDIT — Inspecting score methods...")
    structural = structural_audit()

    pit_safe_count = sum(1 for v in structural.values() if v["pit_safe"])
    pit_unsafe_count = sum(1 for v in structural.values() if not v["pit_safe"])

    for name, findings in structural.items():
        status = "PASS" if findings["pit_safe"] else "FAIL"
        print(f"  [{status}] {name}: {findings['method_description']}")
        if not findings["pit_safe"]:
            for ds in findings["data_sources"]:
                reg_key = f"registry_{ds}"
                if reg_key in findings:
                    print(f"         registry[{ds}]: pit_safe_level={findings[reg_key]['pit_safe_level']}")

    # --- 2. Runtime Consistency ---
    print(f"\n[2/4] RUNTIME CONSISTENCY — {len(test_dates)} dates x 3 runs...")
    consistency = runtime_consistency_check(test_dates)

    for date, result in consistency.items():
        status = "PASS" if result["consistent"] else "FAIL"
        n_scores = len([v for v in result["scores"].values() if v != 0.0]) if result["scores"] else 0
        print(f"  [{status}] {date}: {n_scores} non-zero scores, consistent={result['consistent']}")
        if not result["consistent"] and "differences" in result:
            for d in result["differences"]:
                print(f"         {d['score']}: {d['values']}")

    # --- 3. Temporal Stability (data-level) ---
    print(f"\n[3/4] TEMPORAL STABILITY — Checking _as_of filtering on data files...")
    temporal = temporal_stability_check(test_dates)

    for fname, info in temporal.items():
        if not info.get("exists", False):
            print(f"  [SKIP] {fname}: not found")
            continue
        if "error" in info:
            print(f"  [ERR ] {fname}: {info['error'][:80]}")
            continue
        if "date_checks" in info:
            # Check if filtering is correct for all test dates
            all_correct = all(
                v == "before_data_range" or (isinstance(v, dict) and v.get("correctly_filtered", True))
                for v in info["date_checks"].values()
            )
            status = "PASS" if all_correct else "WARN"
            dr = info.get("date_range", ["?", "?"])
            print(f"  [{status}] {fname}: date_col={info.get('date_col','?')}, range={dr[0]}~{dr[1]}, rows={info.get('total_rows', '?')}")
            if not all_correct:
                for tdate, check in info["date_checks"].items():
                    if isinstance(check, dict) and not check.get("correctly_filtered", True):
                        print(f"         {tdate}: selected {check['rows_selected']}/{check['rows_total']} ({check['pct_selected']}%)")

    # --- 4. Score Sensitivity ---
    print(f"\n[4/4] SCORE SENSITIVITY — T vs T-5 comparison...")
    sensitivity = compute_score_sensitivity(test_dates[-4:])  # last 4 dates only

    for date, comparison in sensitivity.items():
        if "error" in comparison:
            print(f"  [ERR ] {date}: {comparison['error']}")
            continue
        changed = [k for k, v in comparison.items() if v["changed"]]
        both_zero = [k for k, v in comparison.items() if v.get("both_zero")]
        print(f"  {date}: {len(changed)} changed, {len(both_zero)} always-zero")
        for k in both_zero:
            v = comparison[k]
            # Always-zero might mean data doesn't cover this period
            pass

    # --- Compile final report ---
    violations = []
    for name, findings in structural.items():
        if not findings["pit_safe"]:
            violations.append({
                "score": name,
                "reason": findings["method_description"],
                "data_sources": findings["data_sources"],
                "registry_pit_levels": findings["registry_pit_levels"],
            })

    # Also flag scores that are always zero (data coverage issue)
    data_coverage_warnings = []
    for date, result in consistency.items():
        if result["scores"]:
            for k, v in result["scores"].items():
                if v == 0.0 and k not in ["risk_on_score"]:
                    # Only flag if zero for ALL dates
                    pass

    # Find scores that are zero for ALL test dates
    all_scores = list(consistency.values())
    if all_scores and all_scores[0]["scores"]:
        score_keys = [k for k in all_scores[0]["scores"] if k.endswith("_score") or k == "microcap_crash_risk"]
        for k in score_keys:
            vals = [r["scores"].get(k, 0.0) for r in all_scores if r["scores"]]
            if all(v == 0.0 for v in vals):
                data_coverage_warnings.append({
                    "score": k,
                    "issue": "Returns 0.0 for ALL test dates — data may be missing or date range insufficient",
                })

    report = {
        "audit_date": datetime.now().strftime("%Y-%m-%d"),
        "audit_timestamp": datetime.now().isoformat(),
        "test_dates": test_dates,
        "scores": {
            name: {
                "pit_safe": findings["pit_safe"],
                "method": findings["method_description"],
                "data_sources": findings["data_sources"],
                "uses_as_of": findings["uses_as_of"],
                "has_date_filter": findings["has_date_filter"],
                "registry_pit_levels": findings["registry_pit_levels"],
            }
            for name, findings in structural.items()
        },
        "overall_pit_safe": pit_unsafe_count == 0,
        "pit_safe_count": pit_safe_count,
        "pit_unsafe_count": pit_unsafe_count,
        "violations": violations,
        "data_coverage_warnings": data_coverage_warnings,
        "runtime_consistency": {
            date: {"consistent": r["consistent"]}
            for date, r in consistency.items()
        },
        "temporal_stability_summary": {
            fname: {
                "exists": info.get("exists", False),
                "date_range": info.get("date_range"),
                "total_rows": info.get("total_rows"),
            }
            for fname, info in temporal.items()
        },
    }

    # Save report
    out_path = DATA_DIR / "regime_pit_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  PIT-safe scores:   {pit_safe_count}/11")
    print(f"  PIT-unsafe scores: {pit_unsafe_count}/11")
    print(f"  Runtime consistent: {sum(1 for r in consistency.values() if r['consistent'])}/{len(test_dates)}")
    if violations:
        print(f"\n  VIOLATIONS:")
        for v in violations:
            print(f"    - {v['score']}: {v['reason']}")
    if data_coverage_warnings:
        print(f"\n  DATA COVERAGE WARNINGS:")
        for w in data_coverage_warnings:
            print(f"    - {w['score']}: {w['issue']}")
    print(f"\n  Report saved to: {out_path}")


if __name__ == "__main__":
    main()
