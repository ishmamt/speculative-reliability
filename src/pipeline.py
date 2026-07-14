"""Full per-instance trajectory loop: Actor generation, speculative branching,
sandbox verification, gating, and logging (spec Sections 5-9).

Each trajectory gets one persistent `git worktree` (separate from the ephemeral,
throwaway worktrees `sandbox.verify_patch` uses per candidate). Only the Actor's
*realized* `edit_file` action is committed into it, advancing the trajectory's
base ref for the next step's candidate checks — alternatives are verified against
that state and discarded, never committed, so the gate stays observational-only
(spec Section 8). At `submit_patch`, resolved/unresolved is read straight off the
FAIL_TO_PASS + PASS_TO_PASS subset run against the worktree's accumulated edits —
SWE-bench's resolved criterion *is* exactly that, so no separate full-suite eval
path is needed (spec Section 3).

Each history entry the Actor sees also carries a real observation (sandbox detail
for edit_file, actual file content for read_file, real test status for run_tests)
read from the trajectory's own worktree — without this, a blind, single-shot agent
has no way to tell it already succeeded/failed and tends to loop or stall.
"""
from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.actor import Action, generate_action
from src.config import Config
from src.gate import gate_label
from src.logging_utils import (
    CandidateRecord,
    InstanceSummary,
    StepRecord,
    clear_instance_log,
    write_step_record,
    write_summary_record,
)
from src.sandbox import (
    SandboxResult,
    commit_patch,
    create_worktree,
    ensure_repo_cloned,
    remove_worktree,
    run_test_subset,
    run_test_subset_detailed,
    verify_patch,
    verify_patch_detailed,
)
from src.signals import branching_width, compute_confidence
from src.speculator import resample_alternatives_v0, run_parallel_v1

_INVALID_OBSERVATION = "your previous response could not be parsed as valid JSON; respond with a single bare JSON object matching the required schema"


def build_state_description(instance: dict[str, Any], history: list[tuple[Action, str]]) -> str:
    """Render the task issue + prior action history into the prompt fed to Actor/Speculator.

    Each history line carries a real observation of what actually happened (sandbox
    detail, file content, or test status) — see module docstring.
    """
    lines = [
        f"Repository: {instance['repo']}",
        f"Base commit: {instance['base_commit']}",
        f"Issue:\n{instance['problem_statement']}",
    ]
    for i, (action, observation) in enumerate(history):
        line = f"Step {i}: {action.tool} target={action.target!r}"
        if observation:
            line += f"\n  observation: {observation}"
        lines.append(line)
    lines.append("Produce the next action as JSON.")
    return "\n".join(lines)


def _action_dict(action: Action) -> dict[str, str]:
    return {"tool": action.tool, "target": action.target, "patch": action.patch}


def _read_file_observation(worktree_path: Path, target: str, max_chars: int) -> str:
    """Read `target`'s current content from the trajectory worktree, truncated to `max_chars`."""
    if not target:
        return "(no target given)"
    file_path = worktree_path / target
    if not file_path.is_file():
        return f"(file not found: {target})"
    try:
        content = file_path.read_text(errors="replace")
    except OSError as exc:
        return f"(could not read file: {exc})"
    if len(content) > max_chars:
        remaining = len(content) - max_chars
        content = content[:max_chars] + f"\n... (truncated, {remaining} more chars)"
    return content


def _edit_observation(sandbox_result: SandboxResult, detail: str) -> str:
    if sandbox_result == "pass":
        return "sandbox=pass"
    if sandbox_result == "fail":
        return f"sandbox=fail; {detail}" if detail else "sandbox=fail"
    return "sandbox=not_applicable (empty or unparsed patch)"


@dataclasses.dataclass
class V1RateSample:
    """One step's raw timing/match data, for the Section 6.6-6.7 p/alpha/beta aggregates."""

    matched: bool
    actor_wall_s: float
    speculator_wall_s: float
    actor_tokens: int
    speculator_tokens: int


def run_step_v0(
    instance: dict[str, Any],
    step_index: int,
    history: list[tuple[Action, str]],
    base_ref: str,
    actor_model: PreTrainedModel,
    actor_tokenizer: PreTrainedTokenizerBase,
    cfg: Config,
) -> tuple[Action, StepRecord, int, float, str]:
    """Run one v0 (sequential) step.

    Returns (realized_action, record, extra_sandbox_calls, extra_wall_clock_ms, edit_detail)
    — `edit_detail` is the sandbox failure detail for an edit_file real action (empty
    otherwise), used by the caller to build the next step's observation.
    """
    state_description = build_state_description(instance, history)
    step_start = time.monotonic()

    real = generate_action(actor_model, actor_tokenizer, state_description, temperature=0.0)
    confidence, _ = compute_confidence(real.token_logprobs)
    k = branching_width(confidence, cfg.signal)

    extra_start = time.monotonic()
    alternatives = resample_alternatives_v0(actor_model, actor_tokenizer, state_description, k, cfg.signal.alt_temperature)
    extra_gen_s = time.monotonic() - extra_start

    actor_sandbox_result: SandboxResult = "not_applicable"
    edit_detail = ""
    if real.action.tool == "edit_file":
        actor_sandbox_result, edit_detail = verify_patch_detailed(instance["instance_id"], real.action.patch, cfg.sandbox, base_ref)

    extra_sandbox_start = time.monotonic()
    candidate_records = [
        CandidateRecord(
            source="actor",
            action=_action_dict(real.action),
            confidence=confidence,
            sandbox_result=actor_sandbox_result,
        )
    ]
    extra_sandbox_calls = 0
    for alt in alternatives:
        alt_confidence, _ = compute_confidence(alt.token_logprobs)
        alt_sandbox_result: SandboxResult = "not_applicable"
        if alt.action.tool == "edit_file":
            alt_sandbox_result = verify_patch(instance["instance_id"], alt.action.patch, cfg.sandbox, base_ref)
            extra_sandbox_calls += 1
        candidate_records.append(
            CandidateRecord(
                source="actor",
                action=_action_dict(alt.action),
                confidence=alt_confidence,
                sandbox_result=alt_sandbox_result,
            )
        )
    extra_sandbox_s = time.monotonic() - extra_sandbox_start

    label = gate_label(confidence, cfg.signal.confidence_threshold, actor_sandbox_result)
    wall_clock_ms = (time.monotonic() - step_start) * 1000
    extra_wall_clock_ms = (extra_gen_s + extra_sandbox_s) * 1000

    record = StepRecord(
        instance_id=instance["instance_id"],
        step_index=step_index,
        action_type=real.action.tool,
        actor_action=_action_dict(real.action),
        actor_confidence=confidence,
        candidates=candidate_records,
        gate_label=label,
        wall_clock_ms=wall_clock_ms,
        extra_model_calls=len(alternatives),
    )
    return real.action, record, extra_sandbox_calls, extra_wall_clock_ms, edit_detail


def run_step_v1(
    instance: dict[str, Any],
    step_index: int,
    history: list[tuple[Action, str]],
    base_ref: str,
    actor_model: PreTrainedModel,
    actor_tokenizer: PreTrainedTokenizerBase,
    speculator_model: PreTrainedModel,
    speculator_tokenizer: PreTrainedTokenizerBase,
    cfg: Config,
) -> tuple[Action, StepRecord, int, float, V1RateSample, str]:
    """Run one v1 (parallel) step.

    Returns (realized_action, record, extra_sandbox_calls, extra_wall_clock_ms, rate_sample,
    edit_detail). `edit_detail` is populated on both paths: on a Speculator cache hit it comes
    from the Speculator candidate's own (already-computed) detailed verification, so reusing
    it costs no extra sandbox call; on a mismatch the real action gets its own verification call.
    """
    state_description = build_state_description(instance, history)
    step_start = time.monotonic()

    result = run_parallel_v1(
        instance["instance_id"],
        actor_model,
        actor_tokenizer,
        speculator_model,
        speculator_tokenizer,
        state_description,
        cfg,
        base_ref,
    )

    actor_confidence, _ = compute_confidence(result.actor_result.token_logprobs)
    baseline_needs_sandbox = result.actor_result.action.tool == "edit_file"

    extra_sandbox_calls = sum(1 for sc in result.speculator_candidates if sc.action.tool == "edit_file")
    edit_detail = ""

    if result.match:
        actor_sandbox_result = result.speculator_sandbox_results[0]
        edit_detail = result.speculator_sandbox_details[0]
        if baseline_needs_sandbox:
            extra_sandbox_calls -= 1  # the matched speculator call substitutes for, not adds to, baseline's call
    else:
        actor_sandbox_result = "not_applicable"
        if baseline_needs_sandbox:
            actor_sandbox_result, edit_detail = verify_patch_detailed(
                instance["instance_id"], result.actor_result.action.patch, cfg.sandbox, base_ref
            )

    candidate_records = [
        CandidateRecord(
            source="actor",
            action=_action_dict(result.actor_result.action),
            confidence=actor_confidence,
            sandbox_result=actor_sandbox_result,
        )
    ]
    for sc, sr in zip(result.speculator_candidates, result.speculator_sandbox_results):
        sc_confidence, _ = compute_confidence(sc.token_logprobs)
        candidate_records.append(
            CandidateRecord(source="speculator", action=_action_dict(sc.action), confidence=sc_confidence, sandbox_result=sr)
        )

    label = gate_label(actor_confidence, cfg.signal.confidence_threshold, actor_sandbox_result)
    wall_clock_ms = (time.monotonic() - step_start) * 1000
    extra_wall_clock_ms = max(0.0, wall_clock_ms - result.actor_wall_s * 1000)

    record = StepRecord(
        instance_id=instance["instance_id"],
        step_index=step_index,
        action_type=result.actor_result.action.tool,
        actor_action=_action_dict(result.actor_result.action),
        actor_confidence=actor_confidence,
        candidates=candidate_records,
        gate_label=label,
        wall_clock_ms=wall_clock_ms,
        extra_model_calls=len(result.speculator_candidates),
    )
    speculator_tokens = sum(len(sc.token_logprobs) for sc in result.speculator_candidates)
    rate_sample = V1RateSample(
        matched=result.match,
        actor_wall_s=result.actor_wall_s,
        speculator_wall_s=result.speculator_wall_s,
        actor_tokens=len(result.actor_result.token_logprobs),
        speculator_tokens=speculator_tokens,
    )
    return (
        result.actor_result.action,
        record,
        max(extra_sandbox_calls, 0),
        extra_wall_clock_ms,
        rate_sample,
        edit_detail,
    )


def run_trajectory(
    instance: dict[str, Any],
    cfg: Config,
    actor_model: PreTrainedModel,
    actor_tokenizer: PreTrainedTokenizerBase,
    speculator_model: PreTrainedModel | None = None,
    speculator_tokenizer: PreTrainedTokenizerBase | None = None,
) -> tuple[InstanceSummary, list[V1RateSample]]:
    """Drive one instance's trajectory to completion (submit_patch or max_steps), logging every step.

    Returns (summary, v1_rate_samples); the second element is empty in v0 mode.
    """
    clear_instance_log(cfg.logging.log_dir, instance["instance_id"])

    repo_path = ensure_repo_cloned(instance, cfg.sandbox.worktree_base_dir)
    trajectory_worktree = create_worktree(repo_path, instance["base_commit"], cfg.sandbox.worktree_base_dir)
    base_ref = instance["base_commit"]

    history: list[tuple[Action, str]] = []
    resolved = False
    total_extra_sandbox_calls = 0
    total_extra_wall_clock_ms = 0.0
    v1_samples: list[V1RateSample] = []

    try:
        for step_index in range(cfg.pipeline.max_steps):
            if cfg.mode == "v0":
                action, record, extra_calls, extra_ms, edit_detail = run_step_v0(
                    instance, step_index, history, base_ref, actor_model, actor_tokenizer, cfg
                )
            else:
                assert speculator_model is not None and speculator_tokenizer is not None
                action, record, extra_calls, extra_ms, rate_sample, edit_detail = run_step_v1(
                    instance, step_index, history, base_ref,
                    actor_model, actor_tokenizer, speculator_model, speculator_tokenizer, cfg,
                )
                v1_samples.append(rate_sample)

            write_step_record(cfg.logging.log_dir, record)
            total_extra_sandbox_calls += extra_calls
            total_extra_wall_clock_ms += extra_ms

            if action.tool == "edit_file":
                observation = _edit_observation(record.candidates[0].sandbox_result, edit_detail)
            elif action.tool == "read_file":
                observation = _read_file_observation(trajectory_worktree, action.target, cfg.pipeline.max_observation_chars)
            elif action.tool == "run_tests":
                test_status, test_detail = run_test_subset_detailed(trajectory_worktree, instance, cfg.sandbox.test_timeout_seconds)
                observation = f"tests={test_status}" + (f"; {test_detail}" if test_detail else "")
            elif action.tool == "invalid":
                observation = _INVALID_OBSERVATION
            else:
                observation = ""
            history.append((action, observation))

            if action.tool == "edit_file":
                try:
                    base_ref = commit_patch(trajectory_worktree, action.patch, f"step {step_index}: {action.target}")
                except ValueError:
                    pass  # patch didn't apply; trajectory state (and base_ref) unchanged

            if action.tool == "submit_patch":
                resolved = run_test_subset(trajectory_worktree, instance, cfg.sandbox.test_timeout_seconds) == "pass"
                break
    finally:
        remove_worktree(repo_path, trajectory_worktree)

    summary = InstanceSummary(
        instance_id=instance["instance_id"],
        resolved=resolved,
        total_steps=len(history),
        total_extra_sandbox_calls=total_extra_sandbox_calls,
        total_extra_wall_clock_ms=total_extra_wall_clock_ms,
    )
    write_summary_record(cfg.logging.log_dir, summary)
    return summary, v1_samples
