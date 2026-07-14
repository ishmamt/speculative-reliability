"""Sandbox verification: apply a candidate patch in an isolated git worktree and run
the instance's FAIL_TO_PASS + PASS_TO_PASS test subset (spec Section 7).

Callable standalone with a raw patch string + instance ID — no model needs to be loaded.
Uses swebench harness utilities for test-directive resolution and log parsing per spec
Section 3 ("do not build custom sandbox/eval infra"), but drives execution via
`git worktree` against the container's existing repo checkout rather than spinning up a
new Docker container per candidate (spec Section 7, point 1).
"""
from __future__ import annotations

import shlex
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Literal

from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.python import get_test_directives
from swebench.harness.utils import get_modified_files

from src.config import SandboxConfig
from src.dataset import load_swebench_lite

SandboxResult = Literal["pass", "fail", "not_applicable"]

_INSTANCE_CACHE: dict[str, dict] | None = None


def _get_instance(instance_id: str) -> dict:
    """Look up a SWE-bench Lite instance dict by ID, loading the dataset once and caching it."""
    global _INSTANCE_CACHE
    if _INSTANCE_CACHE is None:
        _INSTANCE_CACHE = {inst["instance_id"]: inst for inst in load_swebench_lite()}
    return _INSTANCE_CACHE[instance_id]


def _repo_cache_dir(worktree_base_dir: str) -> Path:
    """Persistent clone directory, kept alongside the ephemeral worktree base dir."""
    return Path(worktree_base_dir) / "_repos"


def ensure_repo_cloned(instance: dict, worktree_base_dir: str) -> Path:
    """Clone the instance's repo into the cache dir if not already present; return its path."""
    repo = instance["repo"]
    repo_path = _repo_cache_dir(worktree_base_dir) / repo.replace("/", "__")
    if not repo_path.exists():
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo}.git", str(repo_path)],
            check=True,
        )
    return repo_path


def create_worktree(repo_path: Path, commit_ish: str, worktree_base_dir: str) -> Path:
    """`git worktree add` a fresh detached worktree at `commit_ish` (the instance's base
    commit for a step-0 check, or the trajectory's current head for later steps)."""
    worktree_path = Path(worktree_base_dir) / f"wt_{uuid.uuid4().hex[:12]}"
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree_path), commit_ish],
        cwd=str(repo_path),
        check=True,
    )
    return worktree_path


def commit_patch(worktree_path: Path, patch: str, message: str) -> str:
    """Apply `patch` and commit it in `worktree_path`; return the new commit hash.

    Used to advance a trajectory's persistent worktree so later steps' candidate
    verifications run against the accumulated state, not just the instance's base commit.
    Raises ValueError if there is nothing to commit (empty patch, failed apply, or a
    patch that applies but is a no-op) — callers should treat this as "state unchanged".
    """
    if not patch.strip() or not apply_patch(worktree_path, patch):
        raise ValueError("patch is empty or failed to apply")
    subprocess.run(["git", "add", "-A"], cwd=str(worktree_path), check=True)
    commit_result = subprocess.run(
        ["git", "-c", "user.email=agent@localhost", "-c", "user.name=agent", "commit", "-m", message],
        cwd=str(worktree_path),
        capture_output=True,
        text=True,
    )
    if commit_result.returncode != 0:
        raise ValueError(f"nothing to commit: {commit_result.stdout}{commit_result.stderr}")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(worktree_path), check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def apply_patch(worktree_path: Path, patch: str) -> bool:
    """Apply `patch` in the worktree via `git apply`. Return False if it fails to apply."""
    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=str(worktree_path),
        input=patch,
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def _apply_test_patch(worktree_path: Path, instance: dict) -> bool:
    """Reset the instance's official test files to HEAD and apply the golden `test_patch`,
    so FAIL_TO_PASS/PASS_TO_PASS tests reflect the instance's ground truth regardless of
    whether the candidate's own patch touched those files (mirrors the swebench harness's
    `make_eval_script_list_py`). Return False if the test_patch itself fails to apply.
    """
    test_patch = instance["test_patch"]
    test_files = get_modified_files(test_patch)
    if test_files:
        reset = subprocess.run(
            ["git", "checkout", "HEAD", "--", *test_files], cwd=str(worktree_path), capture_output=True, text=True
        )
        if reset.returncode != 0:
            return False
    return apply_patch(worktree_path, test_patch)


def run_test_subset(worktree_path: Path, instance: dict, timeout_seconds: int) -> SandboxResult:
    """Apply the instance's golden test_patch, then run only its FAIL_TO_PASS + PASS_TO_PASS
    tests via the repo/version-specific test command (spec Section 15: "pytest via SWE-bench
    harness" — non-pytest repos like django/django use their own runner, so the harness's own
    `MAP_REPO_VERSION_TO_SPECS[...]["test_cmd"]` is used rather than a hardcoded `pytest` call).
    """
    fail_to_pass = instance["FAIL_TO_PASS"]
    pass_to_pass = instance["PASS_TO_PASS"]
    test_names = list(fail_to_pass) + list(pass_to_pass)
    if not test_names:
        return "not_applicable"

    if not _apply_test_patch(worktree_path, instance):
        return "fail"

    specs = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]
    test_command = shlex.split(specs["test_cmd"]) + get_test_directives(instance)

    try:
        proc = subprocess.run(
            test_command,
            cwd=str(worktree_path),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return "fail"

    parser = MAP_REPO_TO_PARSER[instance["repo"]]
    statuses = parser(proc.stdout + proc.stderr)

    for test_name in test_names:
        if statuses.get(test_name) != "PASSED":
            return "fail"
    return "pass"


def remove_worktree(repo_path: Path, worktree_path: Path) -> None:
    """Remove the worktree and prune its git metadata (spec Section 7, point 5)."""
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree_path)],
        cwd=str(repo_path),
        check=False,
    )
    if worktree_path.exists():
        shutil.rmtree(worktree_path, ignore_errors=True)
    subprocess.run(["git", "worktree", "prune"], cwd=str(repo_path), check=False)


def verify_patch(instance_id: str, patch: str, sandbox_cfg: SandboxConfig, base_ref: str | None = None) -> SandboxResult:
    """End-to-end sandbox verification of a raw patch string against an instance ID.

    Callable independently of any loaded model: looks up the instance directly from
    the SWE-bench Lite dataset. Only `edit_file` candidates (non-empty patch) should
    reach this function — spec Section 3 exempts read-only/no-op actions.

    `base_ref` defaults to the instance's base commit (single-shot verification); pass
    a trajectory's current worktree head to verify a candidate against accumulated
    prior edits instead of a fresh checkout.
    """
    if not patch.strip():
        return "not_applicable"

    instance = _get_instance(instance_id)
    repo_path = ensure_repo_cloned(instance, sandbox_cfg.worktree_base_dir)
    worktree_path = create_worktree(repo_path, base_ref or instance["base_commit"], sandbox_cfg.worktree_base_dir)
    try:
        if not apply_patch(worktree_path, patch):
            return "fail"
        return run_test_subset(worktree_path, instance, sandbox_cfg.test_timeout_seconds)
    finally:
        remove_worktree(repo_path, worktree_path)
