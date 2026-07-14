"""JSONL logging for per-step records and per-instance summaries (spec Section 9)."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class CandidateRecord:
    source: str  # "actor" | "speculator"
    action: dict[str, Any]
    confidence: float
    sandbox_result: str  # "pass" | "fail" | "not_applicable"


@dataclasses.dataclass
class StepRecord:
    instance_id: str
    step_index: int
    action_type: str
    actor_action: dict[str, Any]
    actor_confidence: float
    candidates: list[CandidateRecord]
    gate_label: str
    wall_clock_ms: float
    extra_model_calls: int


@dataclasses.dataclass
class InstanceSummary:
    instance_id: str
    resolved: bool
    total_steps: int
    total_extra_sandbox_calls: int
    total_extra_wall_clock_ms: float


def _step_log_path(log_dir: str, instance_id: str) -> Path:
    return Path(log_dir) / f"{instance_id}.jsonl"


def clear_instance_log(log_dir: str, instance_id: str) -> None:
    """Remove any existing log file for `instance_id` so a fresh trajectory run doesn't
    append onto stale records from a previous run of the same instance."""
    path = _step_log_path(log_dir, instance_id)
    path.unlink(missing_ok=True)


def write_step_record(log_dir: str, record: StepRecord) -> None:
    """Append one step record as a JSONL line to results/logs/{instance_id}.jsonl."""
    path = _step_log_path(log_dir, record.instance_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(dataclasses.asdict(record)) + "\n")


def write_summary_record(log_dir: str, summary: InstanceSummary) -> None:
    """Append the per-instance summary line to results/logs/{instance_id}.jsonl (spec Section 9)."""
    path = _step_log_path(log_dir, summary.instance_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"summary": True, **dataclasses.asdict(summary)}) + "\n")


def read_instance_log(log_dir: str, instance_id: str) -> tuple[list[dict], dict | None]:
    """Read back an instance's step records and its summary record for analysis."""
    path = _step_log_path(log_dir, instance_id)
    steps: list[dict] = []
    summary: dict | None = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("summary"):
            summary = obj
        else:
            steps.append(obj)
    return steps, summary


def list_logged_instance_ids(log_dir: str) -> list[str]:
    """List instance IDs that have a log file in `log_dir`."""
    return sorted(p.stem for p in Path(log_dir).glob("*.jsonl"))
