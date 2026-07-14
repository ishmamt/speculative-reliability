"""Run the full v0/v1 pipeline over the prepared instance subset, logging every step
and reporting v1's measured speedup against the theoretical formula (spec Section 6.7).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.actor import load_model
from src.config import load_config
from src.dataset import load_instances_from_manifest
from src.pipeline import V1RateSample, run_trajectory


def report_v1_speedup(all_samples: list[V1RateSample]) -> None:
    """Report measured E[Ts]/E[Tseq] against the theoretical formula:
    (1-p)/(1+p) * alpha/(alpha+beta) as T->inf, from "Speculative Actions"
    (arXiv:2510.04371, Proposition 1). v0 reports no speedup by construction (spec Section 5.7).
    """
    if not all_samples:
        print("No v1 steps recorded; skipping speedup report.")
        return

    p = sum(s.matched for s in all_samples) / len(all_samples)
    total_actor_s = sum(s.actor_wall_s for s in all_samples)
    total_actor_tokens = sum(s.actor_tokens for s in all_samples)
    total_spec_s = sum(s.speculator_wall_s for s in all_samples)
    total_spec_tokens = sum(s.speculator_tokens for s in all_samples)

    beta = total_actor_tokens / total_actor_s if total_actor_s > 0 else 0.0
    alpha = total_spec_tokens / total_spec_s if total_spec_s > 0 else 0.0

    theoretical_ratio = ((1 - p) / (1 + p)) * (alpha / (alpha + beta)) if (alpha + beta) > 0 else float("nan")

    speculative_wall_s = sum(max(s.actor_wall_s, s.speculator_wall_s) for s in all_samples)
    sequential_wall_s = sum(s.actor_wall_s + s.speculator_wall_s for s in all_samples)
    measured_ratio = speculative_wall_s / sequential_wall_s if sequential_wall_s > 0 else float("nan")

    print("\n--- v1 speedup report (spec Section 6.6-6.7) ---")
    print(f"p (Speculator top-1 match rate): {p:.4f}")
    print(f"alpha (Speculator tokens/s): {alpha:.2f}")
    print(f"beta (Actor tokens/s): {beta:.2f}")
    print(f"Measured E[Ts]/E[Tseq]:     {measured_ratio:.4f}")
    print(f"Theoretical (1-p)/(1+p)*alpha/(alpha+beta): {theoretical_ratio:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N instances (smoke-testing).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.signal.confidence_threshold is None:
        raise RuntimeError(
            "signal.confidence_threshold is unset. Run scripts/calibrate_threshold.py "
            "and set the threshold in the config before running the full experiment."
        )

    instances = load_instances_from_manifest(cfg.dataset.manifest_path)
    if args.limit is not None:
        instances = instances[: args.limit]

    print(f"Loading Actor model {cfg.models.actor}...")
    actor_model, actor_tokenizer = load_model(cfg.models.actor)

    speculator_model, speculator_tokenizer = None, None
    if cfg.mode == "v1":
        print(f"Loading Speculator model {cfg.models.speculator}...")
        speculator_model, speculator_tokenizer = load_model(cfg.models.speculator)

    all_v1_samples: list[V1RateSample] = []
    resolved_count = 0
    for i, instance in enumerate(instances):
        print(f"[{i + 1}/{len(instances)}] {instance['instance_id']}")
        summary, v1_samples = run_trajectory(
            instance, cfg, actor_model, actor_tokenizer, speculator_model, speculator_tokenizer
        )
        all_v1_samples.extend(v1_samples)
        resolved_count += int(summary.resolved)
        print(f"  resolved={summary.resolved} steps={summary.total_steps}")

    print(f"\nResolved {resolved_count}/{len(instances)} instances (mode={cfg.mode}).")

    if cfg.mode == "v1":
        report_v1_speedup(all_v1_samples)
    else:
        print("v0 mode: no latency/speedup claim by construction (spec Section 5.7).")


if __name__ == "__main__":
    main()
