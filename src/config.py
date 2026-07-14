"""Typed config loading for the reliability-speculation pipeline. No hardcoded parameters live outside this schema."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import yaml


@dataclasses.dataclass
class DatasetConfig:
    name: str
    subset_size: int
    seed: int
    manifest_path: str


@dataclasses.dataclass
class ModelsConfig:
    actor: str
    speculator: str


@dataclasses.dataclass
class SignalConfig:
    confidence_threshold: Optional[float]
    k_default: int
    k_uncertain: int
    alt_temperature: float


@dataclasses.dataclass
class SandboxConfig:
    test_timeout_seconds: int
    worktree_base_dir: str


@dataclasses.dataclass
class LoggingConfig:
    log_dir: str
    log_level: str


@dataclasses.dataclass
class OutputConfig:
    results_dir: str


@dataclasses.dataclass
class CalibrationConfig:
    num_instances: int


@dataclasses.dataclass
class PipelineConfig:
    max_steps: int


@dataclasses.dataclass
class Config:
    mode: str
    dataset: DatasetConfig
    models: ModelsConfig
    signal: SignalConfig
    sandbox: SandboxConfig
    logging: LoggingConfig
    output: OutputConfig
    calibration: CalibrationConfig
    pipeline: PipelineConfig


def load_config(path: str) -> Config:
    """Load and validate the YAML config into a typed Config object."""
    raw = yaml.safe_load(Path(path).read_text())

    mode = raw["mode"]
    if mode not in ("v0", "v1"):
        raise ValueError(f"config.mode must be 'v0' or 'v1', got {mode!r}")

    return Config(
        mode=mode,
        dataset=DatasetConfig(**raw["dataset"]),
        models=ModelsConfig(**raw["models"]),
        signal=SignalConfig(**raw["signal"]),
        sandbox=SandboxConfig(**raw["sandbox"]),
        logging=LoggingConfig(**raw["logging"]),
        output=OutputConfig(**raw["output"]),
        calibration=CalibrationConfig(**raw["calibration"]),
        pipeline=PipelineConfig(**raw["pipeline"]),
    )
