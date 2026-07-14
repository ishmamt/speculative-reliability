"""Observational, 4-way decision gate (spec Section 8). Fixed lookup table, no learned model."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.sandbox import SandboxResult

GateLabel = Literal["confident_good", "confident_but_wrong", "uncertain_but_fine", "flag"]

_TABLE: dict[tuple[bool, bool], GateLabel] = {
    (True, True): "confident_good",
    (True, False): "confident_but_wrong",
    (False, True): "uncertain_but_fine",
    (False, False): "flag",
}


def gate_label(actor_confidence: float, confidence_threshold: float, actor_sandbox_result: SandboxResult) -> GateLabel:
    """Look up the gate label from the Actor's own real action's confidence + sandbox result.

    `not_applicable` sandbox results (non edit_file actions) are treated as passing —
    there is nothing to have gotten wrong.
    """
    is_confident = actor_confidence >= confidence_threshold
    is_pass = actor_sandbox_result in ("pass", "not_applicable")
    return _TABLE[(is_confident, is_pass)]
