#!/usr/bin/env python3
"""Migrate existing model results into the ExperimentArtifact system.

Reads JSON results from various training/backtest scripts and creates
standardized ExperimentArtifact entries so we can use compare_experiments()
to build a unified comparison table.

Sources:
    - data/storage/phase4/institutional_gate_xgb_174.json  (signal metrics)
    - data/storage/phase4_backtest_xgb_174_top20.json       (backtest)
    - data/storage/phase4/model_registry.json                (registry metadata)
    - data/storage/phase4k/optimizer_comparison.json         (opt100to10 12-split)
    - data/storage/phase4k/opt100to10_24split_gate.json      (opt100to10 24-split)
    - data/storage/model_suite_results.json                  (alstm, transformer)
    - data/storage/lgb_rolling_results.json                  (lightgbm)
    - data/storage/phase4/track_b_upgraded.json              (buffered partial)
    - data/storage/phase4/stress_test.json                   (stress test baseline)
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tracker.artifact_contract import ExperimentArtifact, compare_experiments

STORAGE = PROJECT_ROOT / "data" / "storage"


def _load_json(path: Path) -> dict | None:
    """Load JSON, return None if missing or corrupt."""
    if not path.exists():
        print(f"  [SKIP] {path.name} not found")
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"  [SKIP] {path.name} corrupt: {e}")
        return None


def migrate_xgb_174_champion():
    """XGB-174 champion: institutional gate signal + top20 backtest."""
    eid = "xgb_174_champion"
    print(f"\n=== Migrating {eid} ===")

    # --- Signal metrics from institutional gate ---
    gate = _load_json(STORAGE / "phase4" / "institutional_gate_xgb_174.json")
    backtest_raw = _load_json(STORAGE / "phase4_backtest_xgb_174_top20.json")
    registry = _load_json(STORAGE / "phase4" / "model_registry.json")
    stress = _load_json(STORAGE / "phase4" / "stress_test.json")

    # Config from registry + gate
    config_extra = {}
    if registry and "xgb_174" in registry.get("models", {}):
        reg = registry["models"]["xgb_174"]
        config_extra["execution"] = reg.get("execution", {})
        config_extra["n_features"] = reg.get("n_features", 174)
        config_extra["status"] = reg.get("status", "champion")

    if gate:
        config_extra["n_splits"] = gate.get("config", {}).get("n_splits", 24)
        config_extra["train_days"] = gate.get("config", {}).get("train_days", 750)
        config_extra["xgb_params"] = gate.get("config", {}).get("xgb_params", {})
        config_extra["cost_model"] = gate.get("config", {}).get("cost_model", {})

    art = ExperimentArtifact.create(
        experiment_id=eid,
        model_name="xgb_174",
        feature_set="FS-174",
        description="Champion model: XGB 174 features, 24-split rolling, institutional gate passed",
        **config_extra,
    )

    # Signal-level metrics
    if gate:
        sig = gate.get("signal_aggregate", {})
        spreads = gate.get("multi_layer_spreads", {})
        port = gate.get("portfolio", {})
        metrics = {
            "ic_mean": sig.get("ic_mean"),
            "ic_std": sig.get("ic_std"),
            "icir": sig.get("icir"),
            "rank_ic_mean": sig.get("rank_ic_mean"),
            "rank_ic_std": sig.get("rank_ic_std"),
            "rank_icir": sig.get("rank_icir"),
            "spread_top20": spreads.get("top20", {}).get("daily_spread_mean"),
            "spread_top50": spreads.get("top50", {}).get("daily_spread_mean"),
            "spread_top100": spreads.get("top100", {}).get("daily_spread_mean"),
            "n_days": port.get("n_days"),
            "coverage": None,  # not available
        }
        # Add portfolio-level IR
        metrics["excess_ir"] = port.get("information_ratio")
        metrics["ann_excess_return"] = port.get("ann_excess_return")
        metrics["cost_drag_annual"] = port.get("cost_drag_annual")
        art.save_metrics(metrics)
        print(f"  metrics: rank_ic={metrics['rank_ic_mean']:.4f}, icir={metrics['icir']:.3f}")

    # Backtest from top20 paper OMS
    if backtest_raw:
        ca = backtest_raw.get("cost_adjusted", {})
        cost = backtest_raw.get("cost", {})
        bt = {
            "sharpe": ca.get("sharpe"),
            "annual_return": ca.get("annual_return"),
            "annual_vol": ca.get("annual_vol"),
            "max_drawdown": ca.get("max_drawdown"),
            "calmar": ca.get("calmar"),
            "win_rate": ca.get("win_rate"),
            "avg_turnover": cost.get("avg_turnover"),
            "cost_drag": cost.get("cost_to_return_ratio"),
            "n_days": backtest_raw.get("n_days"),
            "avg_holdings": backtest_raw.get("avg_holdings"),
            "test_period": backtest_raw.get("test_period"),
        }
        art.save_backtest(bt)
        print(f"  backtest: sharpe={bt['sharpe']:.3f}, annual={bt['annual_return']:.1%}")

    # Stress test if available
    if stress:
        bl = stress.get("baseline", {})
        art.save_factor_health({
            "source": "stress_test",
            "baseline_sharpe": bl.get("sharpe_ratio"),
            "baseline_annual_return": bl.get("annual_return"),
            "baseline_max_drawdown": bl.get("max_drawdown"),
            "stress_periods": {k: v.get("sharpe_ratio") for k, v in stress.get("stress_periods", {}).items()},
        })
        print(f"  stress_test baseline sharpe: {bl.get('sharpe_ratio'):.3f}")

    report = art.validate()
    print(f"  valid={report['complete']}, missing={report['missing_required']}, warnings={report['warnings']}")
    return eid


def migrate_xgb_174_opt100to10():
    """XGB-174 opt_top100_to10: optimizer v2 portfolio construction."""
    eid = "xgb_174_opt100to10"
    print(f"\n=== Migrating {eid} ===")

    # Signal metrics are same as champion (same model, different portfolio construction)
    gate = _load_json(STORAGE / "phase4" / "institutional_gate_xgb_174.json")
    opt_12 = _load_json(STORAGE / "phase4k" / "optimizer_comparison.json")
    opt_24 = _load_json(STORAGE / "phase4k" / "opt100to10_24split_gate.json")
    registry = _load_json(STORAGE / "phase4" / "model_registry.json")

    config_extra = {}
    if registry and "xgb_174_opt100to10" in registry.get("models", {}):
        reg = registry["models"]["xgb_174_opt100to10"]
        config_extra["execution"] = reg.get("execution", {})
        config_extra["n_features"] = reg.get("n_features", 174)
        config_extra["status"] = reg.get("role", "shadow")

    art = ExperimentArtifact.create(
        experiment_id=eid,
        model_name="xgb_174",
        feature_set="FS-174",
        description="Optimizer v2: top100->10 alpha-proportional weighting, max_turnover=0.1",
        **config_extra,
    )

    # Signal metrics (same underlying model)
    if gate:
        sig = gate.get("signal_aggregate", {})
        spreads = gate.get("multi_layer_spreads", {})
        metrics = {
            "rank_ic_mean": sig.get("rank_ic_mean"),
            "rank_ic_std": sig.get("rank_ic_std"),
            "rank_icir": sig.get("rank_icir"),
            "ic_mean": sig.get("ic_mean"),
            "icir": sig.get("icir"),
            "spread_top100": spreads.get("top100", {}).get("daily_spread_mean"),
        }
        art.save_metrics(metrics)
        print(f"  metrics: rank_ic={metrics['rank_ic_mean']:.4f}")

    # Backtest from optimizer comparison (12-split) and 24-split
    bt = {}
    if opt_12:
        s = opt_12.get("summary", {}).get("opt_top100_to10", {})
        bt["sharpe"] = s.get("avg_sharpe")
        bt["annual_return"] = s.get("avg_annual", 0) / 100.0 if s.get("avg_annual") else None
        bt["max_drawdown"] = -s.get("avg_maxdd", 0) / 100.0 if s.get("avg_maxdd") else None
        bt["avg_turnover"] = s.get("avg_turnover", 0) / 100.0 if s.get("avg_turnover") else None
        bt["cost_drag"] = s.get("avg_cost_drag", 0) / 100.0 if s.get("avg_cost_drag") else None
        bt["n_splits_12"] = s.get("n_splits")
        bt["positive_split_pct"] = s.get("positive_split_pct")

    if opt_24:
        s24 = opt_24.get("summary", {}).get("opt_top100_to10", {})
        bt["sharpe_24split"] = s24.get("avg_sharpe")
        bt["annual_return_24split"] = s24.get("avg_annual", 0) / 100.0 if s24.get("avg_annual") else None
        bt["med_sharpe_24split"] = s24.get("med_sharpe")
        bt["positive_pct_24split"] = s24.get("positive_pct")
        bt["avg_maxdd_24split"] = -s24.get("avg_maxdd", 0) / 100.0 if s24.get("avg_maxdd") else None

    if bt:
        art.save_backtest(bt)
        print(f"  backtest(12-split): sharpe={bt.get('sharpe')}, annual={bt.get('annual_return')}")
        if bt.get("sharpe_24split"):
            print(f"  backtest(24-split): sharpe={bt.get('sharpe_24split')}, annual={bt.get('annual_return_24split')}")

    report = art.validate()
    print(f"  valid={report['complete']}, missing={report['missing_required']}, warnings={report['warnings']}")
    return eid


def migrate_xgb_174_buffered():
    """XGB-174 buffered_partial execution from track_b_upgraded."""
    eid = "xgb_174_buffered_partial"
    print(f"\n=== Migrating {eid} ===")

    track_b = _load_json(STORAGE / "phase4" / "track_b_upgraded.json")
    gate = _load_json(STORAGE / "phase4" / "institutional_gate_xgb_174.json")
    if not track_b:
        print("  [SKIP] no track_b data")
        return None

    art = ExperimentArtifact.create(
        experiment_id=eid,
        model_name="xgb_174",
        feature_set="FS-174",
        description="Buffered partial execution: open-to-open + IPO filter (60d), 12-split",
    )

    # Signal metrics (same model)
    if gate:
        sig = gate.get("signal_aggregate", {})
        art.save_metrics({
            "rank_ic_mean": sig.get("rank_ic_mean"),
            "rank_ic_std": sig.get("rank_ic_std"),
            "rank_icir": sig.get("rank_icir"),
            "ic_mean": sig.get("ic_mean"),
            "icir": sig.get("icir"),
        })

    # Backtest from track_b buffered_partial
    bp = track_b.get("summary", {}).get("buffered_partial", {})
    bt = {
        "sharpe": bp.get("avg_sharpe"),
        "annual_return": bp.get("avg_annual", 0) / 100.0 if bp.get("avg_annual") else None,
        "max_drawdown": -bp.get("avg_maxdd", 0) / 100.0 if bp.get("avg_maxdd") else None,
        "n_splits": track_b.get("n_splits"),
        "positive_splits": bp.get("positive_splits"),
        "execution_price": track_b.get("execution_price"),
    }
    art.save_backtest(bt)
    print(f"  backtest: sharpe={bt['sharpe']:.3f}, annual={bt['annual_return']:.1%}")

    report = art.validate()
    print(f"  valid={report['complete']}, missing={report['missing_required']}, warnings={report['warnings']}")
    return eid


def migrate_model_suite():
    """Migrate alstm + transformer from model_suite_results.json."""
    eids = []
    suite = _load_json(STORAGE / "model_suite_results.json")
    if not suite:
        return eids

    for m in suite.get("models", []):
        model_name = m.get("model", "unknown")
        eid = f"suite_{model_name}"
        print(f"\n=== Migrating {eid} ===")

        art = ExperimentArtifact.create(
            experiment_id=eid,
            model_name=model_name,
            feature_set="FS-174",
            description=f"Model suite: {model_name}, {m.get('n_dates')} test dates, device={m.get('device')}",
            device=m.get("device"),
            n_samples=m.get("n_samples"),
            train_time_s=m.get("train_time_s"),
        )

        metrics = {
            "ic_mean": m.get("ic_mean"),
            "ic_std": m.get("ic_std"),
            "icir": m.get("icir"),
            "rank_ic_mean": m.get("rank_ic_mean"),
            "rank_ic_pos_ratio": m.get("rank_ic_pos_ratio"),
            "spread_top20": m.get("top20_bot20_spread"),
            "n_days": m.get("n_dates"),
        }
        # Compute rank_ic_std from rank_ic_mean and any available info
        # Not directly available, set to None
        metrics["rank_ic_std"] = None
        art.save_metrics(metrics)
        print(f"  metrics: rank_ic={metrics['rank_ic_mean']:.4f}, icir={metrics['icir']:.3f}")

        report = art.validate()
        print(f"  valid={report['complete']}, missing={report['missing_required']}, warnings={report['warnings']}")
        eids.append(eid)

    return eids


def migrate_lgb_rolling():
    """Migrate LightGBM rolling results."""
    eid = "lgb_rolling_6split"
    print(f"\n=== Migrating {eid} ===")

    lgb = _load_json(STORAGE / "lgb_rolling_results.json")
    if not lgb:
        return None

    agg = lgb.get("aggregate", {})

    art = ExperimentArtifact.create(
        experiment_id=eid,
        model_name="lightgbm",
        feature_set="FS-174",
        description=f"LightGBM rolling: {lgb.get('n_splits')} splits, {lgb.get('test_days_per_split')} test days each",
        n_splits=lgb.get("n_splits"),
        label_expression=lgb.get("label_expression"),
    )

    metrics = {
        "ic_mean": agg.get("ic_mean"),
        "ic_std": agg.get("ic_std"),
        "rank_ic_mean": agg.get("rank_ic_mean"),
        "rank_ic_pos_ratio": agg.get("rank_ic_pos_splits"),
        "spread_top20": agg.get("top20_spread_mean"),
        "spread_pos_ratio": agg.get("spread_pos_splits"),
        "rank_ic_std": None,  # not directly available
    }
    art.save_metrics(metrics)
    print(f"  metrics: rank_ic={metrics['rank_ic_mean']:.4f}, ic={metrics['ic_mean']:.4f}")

    report = art.validate()
    print(f"  valid={report['complete']}, missing={report['missing_required']}, warnings={report['warnings']}")
    return eid


def migrate_xgb_205_research():
    """Migrate XGB-205 (downgraded to research_only) from registry."""
    eid = "xgb_205_research"
    print(f"\n=== Migrating {eid} ===")

    registry = _load_json(STORAGE / "phase4" / "model_registry.json")
    if not registry or "xgb_205" not in registry.get("models", {}):
        print("  [SKIP] no xgb_205 in registry")
        return None

    reg = registry["models"]["xgb_205"]

    art = ExperimentArtifact.create(
        experiment_id=eid,
        model_name="xgb_205",
        feature_set="FS-205",
        description=f"Research only: {reg.get('downgrade_reason', 'unknown')}",
        status=reg.get("status"),
        downgraded_at=reg.get("downgraded_at"),
    )

    # Minimal metrics (no detailed results available)
    art.save_metrics({
        "rank_ic_mean": None,
        "rank_ic_std": None,
        "note": "Downgraded to research_only; regime negative control failed",
    })

    report = art.validate()
    print(f"  valid={report['complete']}, missing={report['missing_required']}, warnings={report['warnings']}")
    return eid


def main():
    print("=" * 70)
    print("Migrating existing results to ExperimentArtifact system")
    print("=" * 70)

    experiment_ids = []

    # 1. XGB-174 Champion
    eid = migrate_xgb_174_champion()
    if eid:
        experiment_ids.append(eid)

    # 2. XGB-174 opt100to10
    eid = migrate_xgb_174_opt100to10()
    if eid:
        experiment_ids.append(eid)

    # 3. XGB-174 buffered partial
    eid = migrate_xgb_174_buffered()
    if eid:
        experiment_ids.append(eid)

    # 4. Model suite (alstm, transformer)
    eids = migrate_model_suite()
    experiment_ids.extend(eids)

    # 5. LightGBM
    eid = migrate_lgb_rolling()
    if eid:
        experiment_ids.append(eid)

    # 6. XGB-205 research
    eid = migrate_xgb_205_research()
    if eid:
        experiment_ids.append(eid)

    # --- Comparison Table ---
    print("\n")
    print("=" * 70)
    print("UNIFIED COMPARISON TABLE")
    print("=" * 70)

    df = compare_experiments(experiment_ids)
    if df.empty:
        print("No experiments to compare.")
        return

    # Format for display
    display_cols = [
        "experiment_id", "model_name", "feature_set",
        "rank_ic", "rank_icir", "spread_top20", "spread_top100",
        "sharpe", "annual_return", "max_drawdown", "avg_turnover",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    df_display = df[display_cols].copy()

    # Format numeric columns
    fmt = {
        "rank_ic": "{:.4f}",
        "rank_icir": "{:.3f}",
        "spread_top20": "{:.4f}",
        "spread_top100": "{:.4f}",
        "sharpe": "{:.3f}",
        "annual_return": "{:.1%}",
        "max_drawdown": "{:.1%}",
        "avg_turnover": "{:.4f}",
    }
    for col, f in fmt.items():
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f.format(x) if x is not None and x == x else "-"
            )

    print(df_display.to_string(index=False))
    print(f"\nTotal experiments: {len(df)}")
    from tracker.artifact_contract import EXPERIMENTS_DIR
    print(f"Artifacts stored in: {EXPERIMENTS_DIR}")

    # Also list all known experiments
    all_ids = ExperimentArtifact.list_all()
    print(f"\nAll registered experiments: {all_ids}")


if __name__ == "__main__":
    main()
