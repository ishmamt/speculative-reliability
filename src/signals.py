"""Token-logprob confidence signal and branching-width decision."""
from __future__ import annotations

import math
from typing import Sequence

from src.config import SignalConfig


def compute_confidence(token_logprobs: Sequence[float]) -> tuple[float, float]:
    """Return (mean_logprob, exp(mean_logprob)) over a generated action's token span."""
    if not token_logprobs:
        raise ValueError("token_logprobs must be non-empty")
    mean_logprob = sum(token_logprobs) / len(token_logprobs)
    return mean_logprob, math.exp(mean_logprob)


def branching_width(confidence: float, signal_cfg: SignalConfig) -> int:
    """Return k: k_default if confidence >= threshold, else k_uncertain (spec Section 5, steps 3-4)."""
    if signal_cfg.confidence_threshold is None:
        raise ValueError(
            "signal.confidence_threshold is unset; run scripts/calibrate_threshold.py first"
        )
    if confidence >= signal_cfg.confidence_threshold:
        return signal_cfg.k_default
    return signal_cfg.k_uncertain
