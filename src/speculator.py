"""Speculator: v0 resamples the Actor at higher temperature; v1 runs a separate,
smaller model concurrently with the Actor, sandboxing its candidates while the
Actor is still generating (spec Sections 5-6).
"""
from __future__ import annotations

import dataclasses
import time
from concurrent.futures import ThreadPoolExecutor

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.actor import Action, GeneratedAction, generate_action
from src.config import Config
from src.sandbox import SandboxResult, verify_patch
from src.signals import branching_width, compute_confidence


def resample_alternatives_v0(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    state_description: str,
    k: int,
    alt_temperature: float,
) -> list[GeneratedAction]:
    """Sample k-1 additional actions from the Actor at `signal.alt_temperature` (spec Section 5, step 4)."""
    return [
        generate_action(model, tokenizer, state_description, temperature=alt_temperature)
        for _ in range(max(k - 1, 0))
    ]


def actions_match(a: Action, b: Action) -> bool:
    """Exact-match on tool + target + args, per spec Section 6, step 4."""
    return a.tool == b.tool and a.target == b.target and a.patch == b.patch


def _timed_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    state_description: str,
    temperature: float,
) -> tuple[GeneratedAction, float]:
    """Generate one action, returning (result, wall_clock_seconds) for alpha/beta rate tracking."""
    start = time.monotonic()
    result = generate_action(model, tokenizer, state_description, temperature=temperature)
    elapsed = time.monotonic() - start
    return result, elapsed


@dataclasses.dataclass
class V1StepResult:
    actor_result: GeneratedAction
    actor_wall_s: float
    speculator_candidates: list[GeneratedAction]
    speculator_sandbox_results: list[SandboxResult]
    speculator_wall_s: float
    match: bool  # top Speculator candidate matches the realized Actor action


def run_parallel_v1(
    instance_id: str,
    actor_model: PreTrainedModel,
    actor_tokenizer: PreTrainedTokenizerBase,
    speculator_model: PreTrainedModel,
    speculator_tokenizer: PreTrainedTokenizerBase,
    state_description: str,
    cfg: Config,
    base_ref: str,
) -> V1StepResult:
    """Run Speculator generation concurrently with Actor generation, sandboxing Speculator
    candidates as soon as they're ready so that verification can overlap with the Actor's
    still-running generation (spec Section 6, steps 1-3).

    `base_ref` is the trajectory's current worktree head (base commit for step 0, or the
    commit left by the previous step's accepted edit), so candidates are verified against
    accumulated prior edits rather than a fresh checkout each time.
    """
    with ThreadPoolExecutor(max_workers=4) as pool:
        actor_future = pool.submit(_timed_generate, actor_model, actor_tokenizer, state_description, 0.0)

        spec_top, spec_wall_s = _timed_generate(speculator_model, speculator_tokenizer, state_description, 0.0)
        spec_confidence, _ = compute_confidence(spec_top.token_logprobs)
        k = branching_width(spec_confidence, cfg.signal)

        extra_gen_start = time.monotonic()
        extra_candidates = [
            generate_action(speculator_model, speculator_tokenizer, state_description, temperature=cfg.signal.alt_temperature)
            for _ in range(max(k - 1, 0))
        ]
        spec_wall_s += time.monotonic() - extra_gen_start

        speculator_candidates = [spec_top] + extra_candidates

        sandbox_futures = [
            pool.submit(verify_patch, instance_id, sc.action.patch, cfg.sandbox, base_ref)
            if sc.action.tool == "edit_file" else None
            for sc in speculator_candidates
        ]

        actor_result, actor_wall_s = actor_future.result()
        speculator_sandbox_results = [
            f.result() if f is not None else "not_applicable" for f in sandbox_futures
        ]

    match = actions_match(actor_result.action, speculator_candidates[0].action)

    return V1StepResult(
        actor_result=actor_result,
        actor_wall_s=actor_wall_s,
        speculator_candidates=speculator_candidates,
        speculator_sandbox_results=speculator_sandbox_results,
        speculator_wall_s=spec_wall_s,
        match=match,
    )
