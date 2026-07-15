"""Summarize logged trajectories: action/gate-label distributions, whether any sandbox
check ever passes, and whether/how each trajectory reached submit_patch.

Use this to tell apart "the model genuinely can't solve these" (varied real attempts, no
loops, sandbox occasionally passing on well-formed edits, submit_patch sometimes reached)
from "something is still off in the pipeline" (uniform failure, degenerate loops, patches
never varying, submit_patch never reached). Reads existing logs only — no model reload.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.logging_utils import list_logged_instance_ids, read_instance_log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    instance_ids = list_logged_instance_ids(cfg.logging.log_dir)
    if not instance_ids:
        raise RuntimeError(f"No logs found in {cfg.logging.log_dir}")

    for instance_id in instance_ids:
        steps, summary = read_instance_log(cfg.logging.log_dir, instance_id)
        print(f"\n=== {instance_id} ===")
        if summary is None:
            print("  (no summary — errored or still in progress)")
            continue
        print(f"  resolved={summary['resolved']} steps={summary['total_steps']}")

        action_counts = Counter(s["action_type"] for s in steps)
        gate_counts = Counter(s["gate_label"] for s in steps)
        print(f"  action_type: {dict(action_counts)}")
        print(f"  gate_label: {dict(gate_counts)}")

        all_candidate_results = Counter(c["sandbox_result"] for s in steps for c in s["candidates"])
        print(f"  all candidate sandbox_results (actor + alternatives, every step): {dict(all_candidate_results)}")

        submit_steps = [s for s in steps if s["action_type"] == "submit_patch"]
        if submit_steps:
            print(f"  submit_patch reached at step {submit_steps[0]['step_index']}")
        else:
            print("  submit_patch never reached (hit pipeline.max_steps)")

        edit_steps = [s for s in steps if s["action_type"] == "edit_file" and s["actor_action"]["patch"]]
        if edit_steps:
            last = edit_steps[-1]
            print(
                f"  last non-empty edit_file: step {last['step_index']}, "
                f"patch length {len(last['actor_action']['patch'])}, "
                f"sandbox={last['candidates'][0]['sandbox_result']}"
            )
        else:
            print("  no non-empty edit_file attempts at all")


if __name__ == "__main__":
    main()
