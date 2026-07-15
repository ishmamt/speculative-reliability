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
import re
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

_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def split_test_command(test_cmd: str) -> tuple[list[str], dict[str, str]]:
    """shlex-split a `specs["test_cmd"]` string, peeling off any leading shell-style
    `VAR=value` environment-variable prefixes (e.g. sympy's test_cmd is
    'PYTHONWARNINGS=ignore::UserWarning,ignore::SyntaxWarning bin/test -C --verbose') into a
    separate env dict. These prefixes are valid shell syntax the official harness runs
    through bash, but we invoke the command directly via subprocess (no shell), where the
    first token would otherwise be treated as the program name itself.
    """
    tokens = shlex.split(test_cmd)
    env_overrides: dict[str, str] = {}
    while tokens and _ENV_ASSIGNMENT_RE.match(tokens[0]):
        key, _, value = tokens[0].partition("=")
        env_overrides[key] = value
        tokens = tokens[1:]
    return tokens, env_overrides

_INSTANCE_CACHE: dict[str, dict] | None = None
_INSTALLED_REPO_VERSIONS: dict[tuple[str, str], Path] = {}


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
        _INSTALLED_REPO_VERSIONS[version_key] = _install_repo_dependencies(instance, repo_path, worktree_base_dir)

    return repo_path


def _deps_dir_for(worktree_base_dir: str, repo: str, version: str) -> Path:
    """Isolated install target for one (repo, version)'s dependencies, kept out of the
    shared environment's site-packages — see `_install_repo_dependencies`."""
    return Path(worktree_base_dir) / "_deps" / f"{repo.replace('/', '__')}__{version or 'unversioned'}"


def _install_repo_dependencies(instance: dict, repo_path: Path, worktree_base_dir: str) -> Path:
    """Best-effort, once-per-(repo, version) install of the repo's own runtime dependencies
    (e.g. Django needs `asgiref`, `sqlparse` just to be importable), at this instance's own
    base_commit so version-appropriate deps get pulled. Returns the isolated directory they
    were installed into.

    Critical: this must NOT install into the shared environment's site-packages. A repo under
    test can share its package name with a real dependency of this project's own tooling —
    `sympy` is both a SWE-bench Lite repo *and* a dependency `torch` needs for symbolic shape
    tracing, and `requests` is both a repo *and* a dependency of `huggingface_hub`/`datasets`.
    A global `pip install` of the repo's own (old, base_commit-era) code would silently
    replace the working version our own scripts need, breaking every future script run in
    this environment — not just sandbox verification for that repo. Installing into an
    isolated `--target` directory and adding *only that directory* to the test subprocess's
    own PYTHONPATH (see `run_test_subset_detailed`) avoids this entirely.

    This does NOT attempt to reproduce SWE-bench's exact per-instance pinned environment —
    that needs per-instance conda envs, which this git-worktree-based design intentionally
    skips (spec Section 7). It's a best-effort approximation using the repo's own setup
    metadata. Failures are logged, not raised: some repos (e.g. matplotlib's C extensions)
    may still not fully install this way, and that's a known limitation.
    """
    deps_dir = _deps_dir_for(worktree_base_dir, instance["repo"], str(instance.get("version", "")))
    deps_dir.mkdir(parents=True, exist_ok=True)
    install_worktree = create_worktree(repo_path, instance["base_commit"], worktree_base_dir)
    try:
        result = subprocess.run(
            ["pip", "install", "--target", str(deps_dir), str(install_worktree)],
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
    return deps_dir


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


_DUPLICATED_METHOD_KEY_RE = re.compile(r"^(?P<method>\S+) \((?P<path>.+)\)$")


def _normalize_parser_keys(statuses: dict[str, str]) -> dict[str, str]:
    """This swebench version's Django parser emits keys like
    'test_str (auth_tests.test_models.GroupTests.test_str)' — the method name duplicated at
    the end of the parenthetical — instead of the canonical SWE-bench FAIL_TO_PASS/PASS_TO_PASS
    form 'test_str (auth_tests.test_models.GroupTests)'. Strip the duplication so lookups
    against the canonical names succeed; keys that don't fit the pattern (e.g. docstring-only
    identifiers) pass through unchanged.
    """
    normalized: dict[str, str] = {}
    for key, value in statuses.items():
        match = _DUPLICATED_METHOD_KEY_RE.match(key)
        if match:
            method, path = match.group("method"), match.group("path")
            suffix = f".{method}"
            if path.endswith(suffix):
                key = f"{method} ({path[: -len(suffix)]})"
        normalized[key] = value
    return normalized


_TEST_ID_ONLY_RE = re.compile(r"^(?P<method>\S+) \((?P<path>[\w.]+)\)$")
_PASS_SUFFIXES = ("... ok", "... OK", "...  OK")


def _scan_docstring_test_ids(log: str) -> dict[str, str]:
    """Companion scan for a gap in this swebench version's Django parser: when a test has a
    docstring, Python's unittest prints the test ID alone on one line, then the docstring
    text + ' ... ok' on the *next* line — and the parser only registers status under that
    docstring text, never correlated back to the canonical 'method (Class)' name printed on
    the preceding line. This reads the same raw log and fills in that correlation.

    Conservative by construction: only a passing result is ever recorded here (a lookup miss
    still correctly defaults to "not passed" elsewhere, so this can't manufacture a false pass
    for a genuine failure — it can only recover passes the upstream parser already dropped).
    """
    results: dict[str, str] = {}
    pending: str | None = None
    for raw_line in log.split("\n"):
        line = raw_line.strip()
        if pending is None:
            match = _TEST_ID_ONLY_RE.match(line)
            if match and " ... " not in line:
                method, path = match.group("method"), match.group("path")
                suffix = f".{method}"
                if path.endswith(suffix):
                    path = path[: -len(suffix)]
                pending = f"{method} ({path})"
            continue
        if line.endswith(_PASS_SUFFIXES):
            results[pending] = "PASSED"
        pending = None
    return results


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
    test_tokens, cmd_env_overrides = split_test_command(specs["test_cmd"])
    test_command = test_tokens + get_test_directives(instance)

    # Put the checked-out worktree on PYTHONPATH so `import <package>` resolves to the local,
    # patched checkout rather than failing outright or silently picking up an unrelated globally
    # installed version — otherwise a repo-relative runner script (e.g. Django's
    # ./tests/runtests.py) puts its own directory on sys.path[0], not the repo root. The
    # isolated deps dir from _install_repo_dependencies goes on PYTHONPATH too — but only for
    # this subprocess's own env, never the shared environment (see that function's docstring).
    version_key = (instance["repo"], str(instance.get("version", "")))
    deps_dir = _INSTALLED_REPO_VERSIONS.get(version_key)
    pythonpath_entries = [str(worktree_path)]
    if deps_dir is not None:
        pythonpath_entries.append(str(deps_dir))
    pythonpath_entries.append(os.environ.get("PYTHONPATH", ""))
    env = {**os.environ, **cmd_env_overrides}
    env["PYTHONPATH"] = os.pathsep.join(p for p in pythonpath_entries if p)

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

    raw_log = proc.stdout + proc.stderr
    parser = MAP_REPO_TO_PARSER[instance["repo"]]
    # This swebench version's parser functions all take (log, test_spec) even though the
    # test_spec argument goes unused in their bodies (confirmed for parse_log_django) — it's
    # a signature-compatibility requirement, not a semantic dependency on TestSpec's contents.
    statuses = _normalize_parser_keys(parser(raw_log, make_test_spec(instance)))
    # Fill in passes the parser drops for docstring-bearing tests (see _scan_docstring_test_ids);
    # disjoint key space from the parser's own output, so a plain merge is safe.
    statuses = {**_scan_docstring_test_ids(raw_log), **statuses}

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
