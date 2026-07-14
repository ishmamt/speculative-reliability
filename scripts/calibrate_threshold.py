"""Run the Actor on `calibration.num_instances` instances and print a confidence
percentile table, to manually set `signal.confidence_threshold` before a full run
(spec Section 12). Must run before `run_experiment.py` on the full subset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.actor import generate_action, load_model
from src.config import load_config
from src.dataset import load_instances_from_manifest
from src.pipeline import build_state_description
from src.signals import compute_confidence

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    instances = load_instances_from_manifest(cfg.dataset.manifest_path)[: cfg.calibration.num_instances]
    if not instances:
        raise RuntimeError("No instances in manifest; run scripts/prepare_dataset.py first.")

    print(f"Loading Actor model {cfg.models.actor}...")
    actor_model, actor_tokenizer = load_model(cfg.models.actor)

    confidences: list[float] = []
    for instance in instances:
        state_description = build_state_description(instance, history=[])
        result = generate_action(actor_model, actor_tokenizer, state_description, temperature=0.0)
        mean_logprob, _ = compute_confidence(result.token_logprobs)
        confidences.append(mean_logprob)
        print(f"{instance['instance_id']}: mean_logprob={mean_logprob:.4f} tool={result.action.tool}")

    arr = np.array(confidences)
    print("\nConfidence (mean token logprob) percentile table:")
    print(f"{'percentile':>10} | {'mean_logprob':>12}")
    for p in PERCENTILES:
        print(f"{p:>10} | {np.percentile(arr, p):>12.4f}")

    print(
        "\nSet `signal.confidence_threshold` in the config to a percentile value above "
        "that separates low- from high-confidence actions, then run scripts/run_experiment.py."
    )


if __name__ == "__main__":
    main()
