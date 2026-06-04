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
    # --- Executable PnL criteria (WARNINGS for now) ---
    "min_cost_adjusted_spread": 0.001,  # must be positive after costs
    "max_turnover_increase": 0.20,      # overlay can't increase turnover >20%
    "max_single_industry_weight": 0.15, # industry concentration limit
}


class PromotionGate:

    def __init__(self, thresholds: dict = None):
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def check(
        self,
        experiment_id: str,
        champion_id: str = None,
        split_experiment_ids: list[str] = None,
        *,
        allow_no_splits: bool = False,
    ) -> dict:
        """Run all gate checks on an experiment.

        Args:
            experiment_id: The candidate experiment to evaluate.
            champion_id: Current champion for delta comparison. Optional.
            split_experiment_ids: Per-split experiment IDs for rolling gate.
                Pre-fix this was nullable and callers could just omit it,
                degrading the "24 split positive delta" requirement to a
                single-artifact aggregate check. 2026-06-04 cx round 6
                P1-6: now ``None`` is REJECTED by default; pass
                ``allow_no_splits=True`` to opt out explicitly (e.g.
                for diagnostic / single-artifact checks where 24-split
                isn't meaningful).
            allow_no_splits: explicit opt-out from the split requirement.

        Returns:
            dict with keys: pass, failures, warnings, checks, recommendation
        """
        if not split_experiment_ids and not allow_no_splits:
            return {
                "pass": False,
                "failures": [
                    "split_experiment_ids is required (cx round 6 P1-6). "
                    "Pass the 24-split list or allow_no_splits=True for "
                    "diagnostic single-artifact checks."
                ],
                "warnings": [],
                "checks": {},
                "recommendation": "reject",
            }
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

        # 2026-06-04 cx round 6 P1-3: PIT audit gate. Pre-fix the
        # module docstring promised "PIT audit pass" but the code did
        # NOT check any PIT artifact. A look-ahead candidate with good
        # IC/backtest could slip the gate. The artifact ``pit_audit.json``
        # is now mandatory: it must exist AND contain ``passed: true``.
        # Producers (e.g. shadow runners) record this after running a
        # source-time / as-of-replay audit.
        pit_audit = art.load_aux("pit_audit") if hasattr(art, "load_aux") else None
        if pit_audit is None:
            # Older artifacts pre-date load_aux helper; try direct read.
            try:
                import json as _json_pit
                from tracker.artifact_contract import EXPERIMENTS_DIR as _ED
                _pit_path = Path(_ED) / experiment_id / "pit_audit.json"
                if _pit_path.exists():
                    pit_audit = _json_pit.loads(_pit_path.read_text())
            except Exception:
                pit_audit = None
        checks["pit_audit"] = bool(pit_audit and pit_audit.get("passed"))
        if not checks["pit_audit"]:
            failures.append(
                "PIT audit missing or failed (pit_audit.json must have "
                "passed=true). The shadow runner is expected to produce "
                "this artifact after a source-time / as-of-replay audit."
            )

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
        # 2026-06-04 cx round 12 P1-4 + cx round 11 P2-6: the gate now
        # requires three-of-three criteria when the backtest exposes
        # them. The pre-fix gate could pass with only label-space
        # RankIC positive — a champion that beats noise on standardised
        # label but earns no real return could still be promoted.
        #   (a) rank_ic_mean > min_rank_ic              (existing)
        #   (b) top20_raw_return_spread > 0             (new, raw $)
        #   (c) cost_adjusted_spread > min_cost_adjusted_spread (P1-4)
        # If the backtest does not record raw return spread (legacy
        # artifact), criterion (b) is recorded as a "raw_return_unknown"
        # warning instead of a hard failure so older artifacts can still
        # be evaluated — but the recommendation tag is downgraded.
        raw_return_spread_seen = False
        raw_return_spread_positive = False
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

            # cx round 12 P1-4: raw forward-return spread check (NOT
            # the label-space spread which may be CSZScoreNorm'd).
            rrs = bt.get("top20_raw_return_spread")
            if rrs is not None:
                raw_return_spread_seen = True
                checks["top20_raw_return_spread"] = rrs
                if rrs <= 0:
                    failures.append(
                        f"top20_raw_return_spread={rrs:+.4f} ≤ 0 "
                        f"(model has no raw $ alpha at top-K)"
                    )
                else:
                    raw_return_spread_positive = True

            # cx round 11 P2-6: gate aggregator metrics for the
            # rolling-split case (computed below by the splits block;
            # surface a warning here if a single artifact's RankIC
            # alone is being relied on).
        else:
            failures.append("No backtest.json — portfolio-level validation required for shadow")

        # 3b. Executable PnL criteria.
        # 2026-06-04 cx round 6 P1-4: cost_adjusted_spread used to be
        # a WARNING — a model with no after-cost alpha could still
        # pass=True and earn promote_to_shadow. That defeats the
        # point of a promotion gate. Now: missing OR non-positive
        # cost_adjusted_spread is a HARD FAILURE. Producers must
        # compute and record this in backtest.json (see
        # PortfolioBacktest implementation).
        if bt:
            cost_adj_spread = bt.get("cost_adjusted_spread")
            if cost_adj_spread is None:
                failures.append(
                    "cost_adjusted_spread missing from backtest.json — "
                    "executable-after-cost criterion is mandatory."
                )
            else:
                checks["cost_adjusted_spread"] = cost_adj_spread
                if cost_adj_spread < self.thresholds["min_cost_adjusted_spread"]:
                    failures.append(
                        f"cost_adjusted_spread={cost_adj_spread:.4f} < "
                        f"threshold={self.thresholds['min_cost_adjusted_spread']} "
                        f"(no after-cost alpha — refusing to promote)"
                    )

            avg_turnover = bt.get("avg_turnover")
            if avg_turnover is not None:
                # Flag unreasonably high turnover as a warning
                if avg_turnover > self.thresholds["max_turnover_increase"] + self.thresholds["max_turnover"]:
                    warnings.append(
                        f"avg_turnover={avg_turnover:.2f} exceeds max_turnover + "
                        f"max_turnover_increase={self.thresholds['max_turnover'] + self.thresholds['max_turnover_increase']:.2f} "
                        f"(wildly high turnover)"
                    )

        # 3c. Industry concentration (WARNING)
        exposure_for_concentration = art.load_exposure()
        if exposure_for_concentration:
            ind_weights = exposure_for_concentration.get("industry_active_weights", {})
            if ind_weights:
                max_abs_weight = max(abs(v) for v in ind_weights.values())
                checks["max_single_industry_weight"] = max_abs_weight
                if max_abs_weight > self.thresholds["max_single_industry_weight"]:
                    warnings.append(
                        f"max_single_industry_weight={max_abs_weight:.2f} > "
                        f"threshold={self.thresholds['max_single_industry_weight']} "
                        f"(industry concentration risk)"
                    )

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
            failures.append("No exposure.json — exposure validation required for shadow")

        # 5. Rolling split consistency (if split experiments provided)
        if split_experiment_ids:
            n_splits = len(split_experiment_ids)
            checks["n_splits"] = n_splits

            if n_splits < self.thresholds["min_splits"]:
                failures.append(
                    f"n_splits={n_splits} < "
                    f"threshold={self.thresholds['min_splits']}"
                )

            # Count splits with positive RankIC and missing metrics
            positive_splits = 0
            missing_splits = 0
            split_ics = []
            for sid in split_experiment_ids:
                try:
                    sart = ExperimentArtifact.load(sid)
                    sm = sart.load_metrics()
                    ric = sm.get("rank_ic_mean")
                    if ric is None:
                        missing_splits += 1
                        split_ics.append(None)
                    else:
                        split_ics.append(ric)
                        if ric > 0:
                            positive_splits += 1
                except Exception:
                    missing_splits += 1
                    split_ics.append(None)

            valid_splits = n_splits - missing_splits
            split_pos_ratio = (
                positive_splits / valid_splits if valid_splits > 0 else 0
            )
            checks["split_positive_ratio"] = split_pos_ratio
            checks["missing_splits"] = missing_splits

            # cx round 11 P2-6: multi-metric aggregator. A mean-driven
            # gate is gameable by a few outlier-Sharpe splits that pull
            # the average up. Record median, worst-quartile mean, and
            # max single-split contribution so the gate report exposes
            # those tails. Hard-fail when worst-quartile mean is itself
            # non-positive (= the bottom 25% of splits actively lose).
            import statistics as _stats
            valid_ics = [ic for ic in split_ics if ic is not None]
            if valid_ics:
                median_ic = float(_stats.median(valid_ics))
                checks["median_rank_ic"] = round(median_ic, 6)
                sorted_ics = sorted(valid_ics)
                q1_size = max(int(len(sorted_ics) * 0.25), 1)
                worst_quartile = sorted_ics[:q1_size]
                worst_q_mean = (
                    float(_stats.mean(worst_quartile)) if worst_quartile else 0.0
                )
                checks["worst_quartile_mean_rank_ic"] = round(worst_q_mean, 6)
                total_abs = sum(abs(ic) for ic in valid_ics) or 1.0
                max_single_contribution = max(
                    abs(ic) for ic in valid_ics
                ) / total_abs
                checks["max_single_split_contribution"] = round(
                    max_single_contribution, 4
                )
                if worst_q_mean <= 0:
                    failures.append(
                        f"worst_quartile_rank_ic_mean={worst_q_mean:+.4f} ≤ 0 "
                        f"(bottom 25% of splits actively lose — mean is "
                        f"propped by extremes)"
                    )
                if max_single_contribution > 0.40:
                    failures.append(
                        f"max_single_split_contribution={max_single_contribution:.2%} > 40% "
                        f"(one split dominates the aggregate — distribution "
                        f"too thin for promotion)"
                    )

            # HARD FAIL: split positive ratio below threshold
            if split_pos_ratio < self.thresholds["min_rank_ic_pos_ratio"]:
                failures.append(
                    f"split_positive_ratio={split_pos_ratio:.2f} < "
                    f"threshold={self.thresholds['min_rank_ic_pos_ratio']} "
                    f"({positive_splits}/{valid_splits} splits positive)"
                )

            # HARD FAIL: too many missing split metrics
            if missing_splits > n_splits * 0.25:
                failures.append(
                    f"missing_splits={missing_splits}/{n_splits} "
                    f"(>25% splits lack metrics)"
                )

            # Delta vs champion (if provided)
            if champion_id:
                try:
                    champion_art = ExperimentArtifact.load(champion_id)
                    champion_metrics = champion_art.load_metrics()
                    champion_ric = champion_metrics.get("rank_ic_mean", 0)
                    checks["champion_rank_ic"] = champion_ric

                    # Per-split delta vs champion
                    delta_positive = 0
                    delta_valid = 0
                    for ric in split_ics:
                        if ric is not None:
                            delta_valid += 1
                            if ric > champion_ric:
                                delta_positive += 1

                    delta_pos_ratio = (
                        delta_positive / delta_valid if delta_valid > 0 else 0
                    )
                    checks["delta_positive_ratio"] = delta_pos_ratio
                    checks["delta_rank_ic"] = (
                        (metrics.get("rank_ic_mean", 0) - champion_ric)
                        if metrics else None
                    )

                    # HARD FAIL: delta positive ratio below threshold
                    if delta_pos_ratio < self.thresholds["min_delta_pos_ratio"]:
                        failures.append(
                            f"delta_positive_ratio={delta_pos_ratio:.2f} < "
                            f"threshold={self.thresholds['min_delta_pos_ratio']} "
                            f"({delta_positive}/{delta_valid} splits beat champion)"
                        )
                except Exception:
                    warnings.append(f"Could not load champion {champion_id}")

        # 6. Negative control check
        # 2026-06-04 cx round 6 P1-5: pre-fix this unconditionally
        # called ``metrics.get(...)`` even when ``metrics`` was None
        # (no metrics.json branch above just appends a failure). That
        # crashed with AttributeError, masking the real "no metrics"
        # failure with an exception. Guard explicitly.
        if metrics is not None:
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
