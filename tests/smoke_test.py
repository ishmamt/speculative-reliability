"""Structural smoke test: config loading, confidence/gate logic, and log round-trip.

Model-loading (actor.py, speculator.py) and sandboxing (sandbox.py) require a GPU +
Linux/Docker environment with the `swebench` harness installed and are exercised by
`scripts/run_experiment.py --limit 2`, not here — this test covers everything that can
run on a bare Python install (spec Section 14, step 5).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SignalConfig, load_config
from src.gate import gate_label
from src.logging_utils import (
    CandidateRecord,
    InstanceSummary,
    StepRecord,
    read_instance_log,
    write_step_record,
    write_summary_record,
)
from src.signals import branching_width, compute_confidence


def test_config_round_trip() -> None:
    cfg = load_config(str(Path(__file__).resolve().parent.parent / "configs" / "default.yaml"))
    assert cfg.mode in ("v0", "v1")
    assert cfg.dataset.subset_size == 50
    assert cfg.signal.k_uncertain == 3


def test_confidence_and_branching_width() -> None:
    mean_logprob, exp_logprob = compute_confidence([-0.1, -0.2, -0.05])
    assert -0.2 <= mean_logprob <= -0.05
    assert 0.0 < exp_logprob < 1.0

    cfg = SignalConfig(confidence_threshold=-0.15, k_default=1, k_uncertain=3, alt_temperature=0.8)
    assert branching_width(-0.1, cfg) == 1
    assert branching_width(-0.5, cfg) == 3


def test_gate_label_table() -> None:
    assert gate_label(-0.1, -0.15, "pass") == "confident_good"
    assert gate_label(-0.1, -0.15, "fail") == "confident_but_wrong"
    assert gate_label(-0.5, -0.15, "pass") == "uncertain_but_fine"
    assert gate_label(-0.5, -0.15, "fail") == "flag"
    assert gate_label(-0.1, -0.15, "not_applicable") == "confident_good"


def test_logging_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        record = StepRecord(
            instance_id="smoke__1",
            step_index=0,
            action_type="edit_file",
            actor_action={"tool": "edit_file", "target": "a.py", "patch": "diff"},
            actor_confidence=-0.1,
            candidates=[
                CandidateRecord(source="actor", action={"tool": "edit_file", "target": "a.py", "patch": "diff"}, confidence=-0.1, sandbox_result="pass")
            ],
            gate_label="confident_good",
            wall_clock_ms=42.0,
            extra_model_calls=0,
        )
        write_step_record(tmp_dir, record)
        write_summary_record(tmp_dir, InstanceSummary("smoke__1", True, 1, 0, 0.0))

        steps, summary = read_instance_log(tmp_dir, "smoke__1")
        assert len(steps) == 1
        assert steps[0]["actor_confidence"] == -0.1
        assert summary is not None
        assert summary["resolved"] is True


if __name__ == "__main__":
    test_config_round_trip()
    test_confidence_and_branching_width()
    test_gate_label_table()
    test_logging_round_trip()
    print("All smoke tests passed.")
