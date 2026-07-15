"""Sandbox verification: apply a candidate patch in an isolated git worktree and run
the instance's FAIL_TO_PASS + PASS_TO_PASS test subset (spec Section 7).

Callable standalone with a raw patch string + instance ID — no model needs to be loaded.
Uses swebench harness utilities for test-directive resolution and log parsing per spec
Section 3 ("do not build custom sandbox/eval infra"), but drives execution via
`git worktree` against the container's existing repo checkout rather than spinning up a
new Docker container per candidate (spec Section 7, point 1).
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Literal

from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.python import get_test_directives
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import get_modified_files

from src.config import SandboxConfig
from src.dataset import load_swebench_lite

SandboxResult = Literal["pass", "fail", "not_applicable"]

_INSTANCE_CACHE: dict[str, dict] | None = None
_INSTALLED_REPO_VERSIONS: set[tuple[str, str]] = set()


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
    """Clone the instance's repo into the cache dir if not already present; return its path.

    Also ensures the repo's own runtime dependencies are installed at least once for this
    (repo, version) pair — see `_install_repo_dependencies` for why this is necessary.
    """
    repo = instance["repo"]
    repo_path = _repo_cache_dir(worktree_base_dir) / repo.replace("/", "__")
    if not repo_path.exists():
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo}.git", str(repo_path)],
            check=True,
        )

    version_key = (repo, str(instance.get("version", "")))
    if version_key not in _INSTALLED_REPO_VERSIONS:
        _install_repo_dependencies(instance, repo_path, worktree_base_dir)
        _INSTALLED_REPO_VERSIONS.add(version_key)

    return repo_path


def _install_repo_dependencies(instance: dict, repo_path: Path, worktree_base_dir: str) -> None:
    """Best-effort, once-per-(repo, version) install of the repo's own runtime dependencies
    (e.g. Django needs `asgiref`, `sqlparse` just to be importable) into the shared Python
    environment, at this instance's own base_commit so version-appropriate deps get pulled.

    This does NOT attempt to reproduce SWE-bench's exact per-instance pinned environment —
    that needs per-instance conda envs, which this git-worktree-based design intentionally
    skips (spec Section 7). It's a best-effort approximation using the repo's own setup
    metadata. A plain (non-editable) install is used deliberately: the throwaway worktree
    this installs from is removed immediately after, and `run_test_subset_detailed` already
    puts each test run's own worktree on PYTHONPATH ahead of site-packages — so the only
    thing this install needs to durably provide is third-party dependencies, not the
    package's own code. Failures are logged, not raised: some repos (e.g. matplotlib's C
    extensions) may still not fully install this way, and that's a known limitation.
    """
    install_worktree = create_worktree(repo_path, instance["base_commit"], worktree_base_dir)
    try:
        result = subprocess.run(
            ["pip", "install", str(install_worktree)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(
                f"warning: dependency install for {instance['repo']}@{instance.get('version')} "
                f"failed; sandbox test runs for this repo/version may fail as a result:\n"
                f"{result.stderr[-1000:]}"
            )
    except subprocess.TimeoutExpired:
        print(f"warning: dependency install for {instance['repo']}@{instance.get('version')} timed out")
    finally:
        remove_worktree(repo_path, install_worktree)


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


def _summarize_test_statuses(instance: dict, statuses: dict[str, str]) -> str:
    """Short human-readable summary of which FAIL_TO_PASS/PASS_TO_PASS tests are still
    failing or regressed, for feeding back to the Actor as an observation (not logged —
    the logged `sandbox_result` field stays the bare pass/fail/not_applicable literal
    per spec Section 9)."""
    still_failing = [t for t in instance["FAIL_TO_PASS"] if statuses.get(t) != "PASSED"]
    regressed = [t for t in instance["PASS_TO_PASS"] if statuses.get(t) != "PASSED"]
    parts = []
    if still_failing:
        parts.append(f"still failing: {', '.join(still_failing[:5])}")
    if regressed:
        parts.append(f"regressed (previously passing): {', '.join(regressed[:5])}")
    return "; ".join(parts)


def run_test_subset_detailed(worktree_path: Path, instance: dict, timeout_seconds: int) -> tuple[SandboxResult, str]:
    """Apply the instance's golden test_patch, then run only its FAIL_TO_PASS + PASS_TO_PASS
    tests via the repo/version-specific test command (spec Section 15: "pytest via SWE-bench
    harness" — non-pytest repos like django/django use their own runner, so the harness's own
    `MAP_REPO_VERSION_TO_SPECS[...]["test_cmd"]` is used rather than a hardcoded `pytest` call).

    Returns (status, detail) — `detail` names which tests are still failing/regressed, for
    observation-building; the logged JSONL schema only ever stores `status`.
    """
    fail_to_pass = instance["FAIL_TO_PASS"]
    pass_to_pass = instance["PASS_TO_PASS"]
    test_names = list(fail_to_pass) + list(pass_to_pass)
    if not test_names:
        return "not_applicable", ""

    if not _apply_test_patch(worktree_path, instance):
        return "fail", "golden test_patch failed to apply"

    specs = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]
    test_command = shlex.split(specs["test_cmd"]) + get_test_directives(instance)

    # Put the checked-out worktree on PYTHONPATH so `import <package>` resolves to the local,
    # patched checkout rather than failing outright or silently picking up an unrelated globally
    # installed version — otherwise a repo-relative runner script (e.g. Django's
    # ./tests/runtests.py) puts its own directory on sys.path[0], not the repo root.
    env = {**os.environ, "PYTHONPATH": str(worktree_path) + os.pathsep + os.environ.get("PYTHONPATH", "")}

    try:
        proc = subprocess.run(
            test_command,
            cwd=str(worktree_path),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return "fail", "test run timed out"
    except OSError as exc:
        # e.g. the repo's test_cmd binary (pytest, a runner script, ...) isn't on PATH or
        # isn't executable in this shared environment — an environment gap, not a real
        # test failure, but still something the instance can't be verified against.
        return "fail", f"test command could not be run: {exc}"

    parser = MAP_REPO_TO_PARSER[instance["repo"]]
    # This swebench version's parser functions all take (log, test_spec) even though the
    # test_spec argument goes unused in their bodies (confirmed for parse_log_django) — it's
    # a signature-compatibility requirement, not a semantic dependency on TestSpec's contents.
    statuses = parser(proc.stdout + proc.stderr, make_test_spec(instance))

    for test_name in test_names:
        if statuses.get(test_name) != "PASSED":
            return "fail", _summarize_test_statuses(instance, statuses)
    return "pass", ""


def run_test_subset(worktree_path: Path, instance: dict, timeout_seconds: int) -> SandboxResult:
    """Bare pass/fail/not_applicable variant of `run_test_subset_detailed`, for callers
    (e.g. candidate/alternative verification) that don't need the failure detail text."""
    status, _ = run_test_subset_detailed(worktree_path, instance, timeout_seconds)
    return status


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


def verify_patch_detailed(
    instance_id: str, patch: str, sandbox_cfg: SandboxConfig, base_ref: str | None = None
) -> tuple[SandboxResult, str]:
    """`verify_patch` plus a failure-detail string, for the one real action per step that
    gets fed back to the Actor as an observation (spec's logged schema only ever stores
    the bare status; see `run_test_subset_detailed`).
    """
    if not patch.strip():
        return "not_applicable", ""

    instance = _get_instance(instance_id)
    repo_path = ensure_repo_cloned(instance, sandbox_cfg.worktree_base_dir)
    worktree_path = create_worktree(repo_path, base_ref or instance["base_commit"], sandbox_cfg.worktree_base_dir)
    try:
        if not apply_patch(worktree_path, patch):
            return "fail", "patch failed to apply"
        return run_test_subset_detailed(worktree_path, instance, sandbox_cfg.test_timeout_seconds)
    finally:
        remove_worktree(repo_path, worktree_path)


def verify_patch(instance_id: str, patch: str, sandbox_cfg: SandboxConfig, base_ref: str | None = None) -> SandboxResult:
    """End-to-end sandbox verification of a raw patch string against an instance ID.

    Callable independently of any loaded model: looks up the instance directly from
    the SWE-bench Lite dataset. Only `edit_file` candidates (non-empty patch) should
    reach this function — spec Section 3 exempts read-only/no-op actions.

    `base_ref` defaults to the instance's base commit (single-shot verification); pass
    a trajectory's current worktree head to verify a candidate against accumulated
    prior edits instead of a fresh checkout.
    """
    status, _ = verify_patch_detailed(instance_id, patch, sandbox_cfg, base_ref)
    return status
