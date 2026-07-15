"""SWE-bench Lite dataset loading and deterministic subset selection.

Dataset access goes through the HuggingFace `datasets` library (the same
loading path used internally by the official `swebench` harness) — no
custom sandbox/eval infra is built here per spec Section 3.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset

from src.config import DatasetConfig

SWEBENCH_LITE_HF_NAME = "princeton-nlp/SWE-bench_Lite"
SWEBENCH_LITE_SPLIT = "test"


_JSON_ENCODED_LIST_FIELDS = ("FAIL_TO_PASS", "PASS_TO_PASS")


def _normalize_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """The raw HF dataset stores FAIL_TO_PASS/PASS_TO_PASS as JSON-encoded strings (e.g.
    '["test_a", "test_b"]'), not native lists — `list(a_json_string)` silently degenerates
    into a list of individual characters instead of raising, so this must be normalized
    once at load time rather than trusted as already-a-list by every consumer.
    """
    for field in _JSON_ENCODED_LIST_FIELDS:
        if isinstance(instance.get(field), str):
            instance[field] = json.loads(instance[field])
    return instance


def load_swebench_lite() -> list[dict[str, Any]]:
    """Load the full SWE-bench Lite test split as a list of instance dicts."""
    ds = load_dataset(SWEBENCH_LITE_HF_NAME, split=SWEBENCH_LITE_SPLIT)
    return [_normalize_instance(dict(inst)) for inst in ds]


def filter_excluded_repos(instances: list[dict[str, Any]], exclude_repos: list[str]) -> list[dict[str, Any]]:
    """Drop instances from repos this project's git-worktree-based sandbox can't verify
    correctly — namely compiled C-extension repos (matplotlib, scikit-learn, astropy): their
    tests need a real build step this lightweight, non-Docker sandbox doesn't perform (spec
    Section 7; see README Limitations)."""
    excluded = set(exclude_repos)
    return [inst for inst in instances if inst["repo"] not in excluded]


def select_subset(instances: list[dict[str, Any]], cfg: DatasetConfig) -> list[dict[str, Any]]:
    """Deterministically sample `cfg.subset_size` instances using `cfg.seed`."""
    rng = random.Random(cfg.seed)
    if cfg.subset_size > len(instances):
        raise ValueError(
            f"dataset.subset_size ({cfg.subset_size}) exceeds available instances ({len(instances)})"
        )
    return rng.sample(instances, cfg.subset_size)


def save_manifest(instances: list[dict[str, Any]], manifest_path: str) -> None:
    """Persist selected instance IDs to `cfg.dataset.manifest_path`."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    instance_ids = [inst["instance_id"] for inst in instances]
    path.write_text(json.dumps(instance_ids, indent=2))


def load_manifest(manifest_path: str) -> list[str]:
    """Load previously persisted instance IDs from the manifest file."""
    return json.loads(Path(manifest_path).read_text())


def load_instances_from_manifest(manifest_path: str) -> list[dict[str, Any]]:
    """Load the full SWE-bench Lite split filtered down to the manifest's instance IDs, in manifest order."""
    instance_ids = load_manifest(manifest_path)
    all_instances = {inst["instance_id"]: inst for inst in load_swebench_lite()}
    return [all_instances[iid] for iid in instance_ids]
