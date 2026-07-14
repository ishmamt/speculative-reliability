"""Compute the four spec-defined reliability metrics (Section 10) over logged trajectories
and write a console table + results/reports/summary.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score

from src.config import load_config
from src.logging_utils import list_logged_instance_ids, read_instance_log


def actions_equal(a: dict, b: dict) -> bool:
    return a["tool"] == b["tool"] and a["target"] == b["target"] and a["patch"] == b["patch"]


def match_rate(all_steps: list[dict]) -> float:
    """Fraction of branched steps (>1 candidate) where an alternative/Speculator candidate matches the realized action."""
    branched = [s for s in all_steps if len(s["candidates"]) > 1]
    if not branched:
        return float("nan")
    matched = 0
    for step in branched:
        realized = step["actor_action"]
        others = step["candidates"][1:]
        if any(actions_equal(realized, c["action"]) for c in others):
            matched += 1
    return matched / len(branched)


def confidence_separation(trajectories: list[tuple[list[dict], dict]]) -> tuple[float, float, float]:
    """Mean actor_confidence for resolved vs unresolved trajectories, + AUROC (target=resolved)."""
    resolved_confidences, unresolved_confidences = [], []
    scores, labels = [], []
    for steps, summary in trajectories:
        if not steps:
            continue
        mean_conf = float(np.mean([s["actor_confidence"] for s in steps]))
        resolved = bool(summary["resolved"])
        (resolved_confidences if resolved else unresolved_confidences).append(mean_conf)
        scores.append(mean_conf)
        labels.append(int(resolved))

    mean_resolved = float(np.mean(resolved_confidences)) if resolved_confidences else float("nan")
    mean_unresolved = float(np.mean(unresolved_confidences)) if unresolved_confidences else float("nan")
    auroc = float("nan")
    if len(set(labels)) == 2:
        auroc = roc_auc_score(labels, scores)
    return mean_resolved, mean_unresolved, auroc


def sandbox_predictive_accuracy(trajectories: list[tuple[list[dict], dict]]) -> float:
    """Agreement rate between candidate sandbox pass/fail and the trajectory's final resolved status."""
    agreements, total = 0, 0
    for steps, summary in trajectories:
        resolved = bool(summary["resolved"])
        for step in steps:
            for cand in step["candidates"]:
                if cand["sandbox_result"] == "not_applicable":
                    continue
                predicted_resolved = cand["sandbox_result"] == "pass"
                agreements += int(predicted_resolved == resolved)
                total += 1
    return agreements / total if total else float("nan")


def retrospective_catch_rate(trajectories: list[tuple[list[dict], dict]]) -> float:
    """Among unresolved trajectories, fraction with >=1 'flag' label before trajectory end (retrospective, non-causal)."""
    unresolved = [(steps, summary) for steps, summary in trajectories if not summary["resolved"]]
    if not unresolved:
        return float("nan")
    caught = sum(1 for steps, _ in unresolved if any(s["gate_label"] == "flag" for s in steps))
    return caught / len(unresolved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    instance_ids = list_logged_instance_ids(cfg.logging.log_dir)
    if not instance_ids:
        raise RuntimeError(f"No logs found in {cfg.logging.log_dir}; run scripts/run_experiment.py first.")

    trajectories = []
    total_extra_sandbox_calls = 0
    total_extra_wall_clock_ms = 0.0
    for instance_id in instance_ids:
        steps, summary = read_instance_log(cfg.logging.log_dir, instance_id)
        if summary is None:
            continue
        trajectories.append((steps, summary))
        total_extra_sandbox_calls += summary["total_extra_sandbox_calls"]
        total_extra_wall_clock_ms += summary["total_extra_wall_clock_ms"]

    all_steps = [s for steps, _ in trajectories for s in steps]

    m1 = match_rate(all_steps)
    mean_resolved, mean_unresolved, auroc = confidence_separation(trajectories)
    m3 = sandbox_predictive_accuracy(trajectories)
    m4 = retrospective_catch_rate(trajectories)

    lines = [
        "# Reliability-Speculation Results Summary",
        "",
        f"Mode: `{cfg.mode}` | Trajectories: {len(trajectories)}",
        "",
        "## Metrics",
        "",
        "| # | Metric | Value |",
        "|---|--------|-------|",
        f"| 1 | Match rate (branched steps) | {m1:.4f} |",
        f"| 2a | Mean actor_confidence, resolved trajectories | {mean_resolved:.4f} |",
        f"| 2b | Mean actor_confidence, unresolved trajectories | {mean_unresolved:.4f} |",
        f"| 2c | AUROC (confidence -> resolved) | {auroc:.4f} |",
        f"| 3 | Sandbox predictive accuracy | {m3:.4f} |",
        f"| 4 | Retrospective catch rate (unresolved only) | {m4:.4f} |",
        "",
        "Metric 4 is **retrospective and non-causal**: the decision gate is observational-only "
        "in this version and never altered Actor behavior, so a 'flag' label before trajectory "
        "end reflects what the gate *would* have caught, not a prevented failure.",
        "",
        "## Overhead (not one of the four metrics)",
        "",
        "| Total extra sandbox calls | Total extra wall-clock ms |",
        "|---|---|",
        f"| {total_extra_sandbox_calls} | {total_extra_wall_clock_ms:.1f} |",
    ]
    report = "\n".join(lines)

    print(report)

    output_path = Path(cfg.output.results_dir) / "summary.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
