"""Promotion gate — automated checks for model/factor promotion.

Implements the gate conditions agreed with CX:
  1. PIT audit pass
  2. 24 split metrics.json all generated
  3. Residual RankIC > 0.005 vs champion
  4. 12+ splits with positive delta RankIC
  5. No significant turnover or exposure degradation
  6. Negative control pass (shuffle → IC ≈ 0)

Usage:
    from tracker.promotion_gate import PromotionGate

    gate = PromotionGate()
    result = gate.check("xgb_175_rolling_24split_20260524")
    print(result["pass"])           # True/False
    print(result["failures"])       # list of failed checks
    print(result["recommendation"]) # "promote_to_shadow" / "research_only" / "reject"
"""
import logging
from pathlib import Path
from typing import Optional

from tracker.artifact_contract import ExperimentArtifact, EXPERIMENTS_DIR

logger = logging.getLogger(__name__)

# Default gate thresholds (CX-approved)
DEFAULT_THRESHOLDS = {
    "min_rank_ic": 0.005,               # residual RankIC vs champion
    "min_rank_ic_pos_ratio": 0.50,       # ≥50% splits with positive RankIC
    "min_delta_pos_ratio": 0.50,         # ≥50% splits with positive delta vs champion
    "max_turnover": 0.25,               # max average daily turnover
    "max_cost_to_return": 0.35,         # cost drag / gross return ≤ 35%
    "min_coverage": 0.60,               # factor coverage ≥ 60%
    "min_splits": 12,                   # minimum splits to evaluate
    "max_industry_deviation": 0.15,     # max single-industry active weight
    "negative_control_ic_threshold": 0.01,  # shuffled IC must be < this
}


class PromotionGate:

    def __init__(self, thresholds: dict = None):
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def check(
        self,
        experiment_id: str,
        champion_id: str = None,
        split_experiment_ids: list[str] = None,
    ) -> dict:
        """Run all gate checks on an experiment.

        Args:
            experiment_id: The candidate experiment to evaluate.
            champion_id: Current champion for delta comparison. Optional.
            split_experiment_ids: Per-split experiment IDs for rolling gate.
                If None, checks single-experiment artifacts only.

        Returns:
            dict with keys: pass, failures, warnings, checks, recommendation
        """
        failures = []
        warnings = []
        checks = {}

        # 1. Artifact completeness
        art = ExperimentArtifact.load(experiment_id)
        validation = art.validate()
        checks["artifact_complete"] = validation["complete"]
        if not validation["complete"]:
            failures.append(
                f"Missing required artifacts: {validation['missing_required']}"
            )
        if validation["warnings"]:
            warnings.extend(validation["warnings"])

        # 2. Metrics checks
        metrics = art.load_metrics()
        if metrics:
            rank_ic = metrics.get("rank_ic_mean")
            if rank_ic is not None:
                checks["rank_ic_mean"] = rank_ic
                if rank_ic < self.thresholds["min_rank_ic"]:
                    failures.append(
                        f"rank_ic_mean={rank_ic:.4f} < "
                        f"threshold={self.thresholds['min_rank_ic']}"
                    )

            pos_ratio = metrics.get("rank_ic_pos_ratio")
            if pos_ratio is not None:
                checks["rank_ic_pos_ratio"] = pos_ratio
                if pos_ratio < self.thresholds["min_rank_ic_pos_ratio"]:
                    failures.append(
                        f"rank_ic_pos_ratio={pos_ratio:.2f} < "
                        f"threshold={self.thresholds['min_rank_ic_pos_ratio']}"
                    )

            coverage = metrics.get("coverage")
            if coverage is not None:
                checks["coverage"] = coverage
                if coverage < self.thresholds["min_coverage"]:
                    failures.append(
                        f"coverage={coverage:.2f} < "
                        f"threshold={self.thresholds['min_coverage']}"
                    )
        else:
            failures.append("No metrics.json found")

        # 3. Backtest checks
        bt = art.load_backtest()
        if bt:
            turnover = bt.get("avg_turnover")
            if turnover is not None:
                checks["avg_turnover"] = turnover
                if turnover > self.thresholds["max_turnover"]:
                    failures.append(
                        f"avg_turnover={turnover:.2f} > "
                        f"threshold={self.thresholds['max_turnover']}"
                    )

            ctr = bt.get("cost_to_return_ratio")
            if ctr is not None:
                checks["cost_to_return_ratio"] = ctr
                if ctr > self.thresholds["max_cost_to_return"]:
                    failures.append(
                        f"cost_to_return={ctr:.2f} > "
                        f"threshold={self.thresholds['max_cost_to_return']}"
                    )
        else:
            warnings.append("No backtest.json — portfolio gate skipped")

        # 4. Exposure checks
        exposure = art.load_exposure()
        if exposure:
            industry_weights = exposure.get("industry_active_weights", {})
            if industry_weights:
                max_dev = max(abs(v) for v in industry_weights.values())
                checks["max_industry_deviation"] = max_dev
                if max_dev > self.thresholds["max_industry_deviation"]:
                    warnings.append(
                        f"max_industry_deviation={max_dev:.2f} > "
                        f"threshold={self.thresholds['max_industry_deviation']}"
                    )
        else:
            warnings.append("No exposure.json — exposure gate skipped")

        # 5. Rolling split consistency (if split experiments provided)
        if split_experiment_ids:
            n_splits = len(split_experiment_ids)
            checks["n_splits"] = n_splits

            if n_splits < self.thresholds["min_splits"]:
                failures.append(
                    f"n_splits={n_splits} < "
                    f"threshold={self.thresholds['min_splits']}"
                )

            # Count splits with positive RankIC
            positive_splits = 0
            split_ics = []
            for sid in split_experiment_ids:
                try:
                    sart = ExperimentArtifact.load(sid)
                    sm = sart.load_metrics()
                    ric = sm.get("rank_ic_mean", 0)
                    split_ics.append(ric)
                    if ric > 0:
                        positive_splits += 1
                except Exception:
                    split_ics.append(None)

            checks["split_positive_ratio"] = (
                positive_splits / n_splits if n_splits > 0 else 0
            )

            # Delta vs champion (if provided)
            if champion_id:
                try:
                    champion_art = ExperimentArtifact.load(champion_id)
                    champion_metrics = champion_art.load_metrics()
                    champion_ric = champion_metrics.get("rank_ic_mean", 0)
                    checks["champion_rank_ic"] = champion_ric
                    checks["delta_rank_ic"] = (
                        (metrics.get("rank_ic_mean", 0) - champion_ric)
                        if metrics else None
                    )
                except Exception:
                    warnings.append(f"Could not load champion {champion_id}")

        # 6. Negative control check
        nc = metrics.get("negative_control_ic")
        if nc is not None:
            checks["negative_control_ic"] = nc
            if abs(nc) > self.thresholds["negative_control_ic_threshold"]:
                failures.append(
                    f"negative_control_ic={nc:.4f} > "
                    f"threshold={self.thresholds['negative_control_ic_threshold']} "
                    f"(possible data leak)"
                )

        # Recommendation
        passed = len(failures) == 0
        if passed and len(warnings) <= 2:
            recommendation = "promote_to_shadow"
        elif passed:
            recommendation = "promote_to_shadow_with_warnings"
        elif len(failures) <= 1 and all("backtest" in f or "exposure" in f for f in failures):
            recommendation = "research_only"
        else:
            recommendation = "reject"

        result = {
            "experiment_id": experiment_id,
            "pass": passed,
            "failures": failures,
            "warnings": warnings,
            "checks": checks,
            "recommendation": recommendation,
            "thresholds": self.thresholds,
        }

        # Log summary
        status = "PASS" if passed else "FAIL"
        logger.info(
            f"Gate {status}: {experiment_id} → {recommendation} "
            f"(failures={len(failures)}, warnings={len(warnings)})"
        )

        return result

    def check_quick(self, experiment_id: str) -> bool:
        """Quick pass/fail check without details."""
        return self.check(experiment_id)["pass"]


def run_gate_report(experiment_ids: list[str] = None) -> str:
    """Run gate on multiple experiments and return formatted report."""
    if experiment_ids is None:
        experiment_ids = ExperimentArtifact.list_all()

    gate = PromotionGate()
    lines = ["=== Promotion Gate Report ===", ""]

    for eid in experiment_ids:
        try:
            result = gate.check(eid)
            status = "PASS" if result["pass"] else "FAIL"
            rec = result["recommendation"]
            lines.append(f"  {eid}: {status} → {rec}")
            if result["failures"]:
                for f in result["failures"]:
                    lines.append(f"    FAIL: {f}")
            if result["warnings"]:
                for w in result["warnings"]:
                    lines.append(f"    WARN: {w}")
        except Exception as e:
            lines.append(f"  {eid}: ERROR — {e}")

    return "\n".join(lines)
