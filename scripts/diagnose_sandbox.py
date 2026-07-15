"""Diagnose sandbox verification for one instance's golden (known-correct) patch.

Applies the instance's own reference `patch` (not a model-generated one) and runs it
through the same sandbox path as real candidates, printing both the parsed status/detail
and the raw test-command stdout/stderr. Use this to check whether the sandbox itself is
verified correct, independent of whether the Actor model can solve anything — a
known-correct patch should come back `status: pass`.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.python import get_test_directives
from swebench.harness.test_spec.test_spec import make_test_spec

from src.config import load_config
from src.dataset import load_instances_from_manifest
from src.sandbox import (
    _apply_test_patch,
    _INSTALLED_REPO_VERSIONS,
    _normalize_parser_keys,
    apply_patch,
    create_worktree,
    ensure_repo_cloned,
    remove_worktree,
    run_test_subset_detailed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--instance-id", default=None, help="Specific instance ID; defaults to --index into the manifest.")
    parser.add_argument("--index", type=int, default=0, help="Manifest index to use if --instance-id is not given.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    instances = load_instances_from_manifest(cfg.dataset.manifest_path)
    if args.instance_id:
        instance = next(i for i in instances if i["instance_id"] == args.instance_id)
    else:
        instance = instances[args.index]

    print(f"instance: {instance['instance_id']}  repo: {instance['repo']}  version: {instance.get('version')}")
    print(f"FAIL_TO_PASS: {instance['FAIL_TO_PASS']}")

    repo_path = ensure_repo_cloned(instance, cfg.sandbox.worktree_base_dir)
    wt = create_worktree(repo_path, instance["base_commit"], cfg.sandbox.worktree_base_dir)
    try:
        applied = apply_patch(wt, instance["patch"])
        print(f"golden patch applied: {applied}")

        test_patch_applied = _apply_test_patch(wt, instance)
        print(f"test_patch applied: {test_patch_applied}")

        specs = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]
        test_command = shlex.split(specs["test_cmd"]) + get_test_directives(instance)
        print(f"test_command: {test_command}")

        # ensure_repo_cloned() already ran the isolated (non-shared-env) dependency install;
        # mirror sandbox.py's PYTHONPATH construction so this raw run matches the real path.
        deps_dir = _INSTALLED_REPO_VERSIONS.get((instance["repo"], str(instance.get("version", ""))))
        pythonpath_entries = [str(wt)] + ([str(deps_dir)] if deps_dir else []) + [os.environ.get("PYTHONPATH", "")]
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in pythonpath_entries if p)}
        proc = subprocess.run(
            test_command, cwd=str(wt), capture_output=True, text=True, timeout=cfg.sandbox.test_timeout_seconds, env=env
        )
        print(f"returncode: {proc.returncode}")
        print("--- STDOUT (last 6000 chars) ---")
        print(proc.stdout[-6000:])
        print("--- STDERR (last 3000 chars) ---")
        print(proc.stderr[-3000:])

        parser_fn = MAP_REPO_TO_PARSER[instance["repo"]]
        statuses = _normalize_parser_keys(parser_fn(proc.stdout + proc.stderr, make_test_spec(instance)))
        print(f"\n--- parser found {len(statuses)} test statuses (normalized) ---")
        for key, value in sorted(statuses.items()):
            print(f"  {key!r}: {value}")
        print("\n--- FAIL_TO_PASS names we look up (must match a key above exactly) ---")
        for name in instance["FAIL_TO_PASS"]:
            print(f"  {name!r} -> {statuses.get(name, 'NOT FOUND')}")
    finally:
        remove_worktree(repo_path, wt)

    # Re-run through the real sandbox path (fresh worktree) for the parsed status/detail,
    # since the raw run above already consumed/mutated the worktree above.
    wt2 = create_worktree(repo_path, instance["base_commit"], cfg.sandbox.worktree_base_dir)
    try:
        apply_patch(wt2, instance["patch"])
        status, detail = run_test_subset_detailed(wt2, instance, cfg.sandbox.test_timeout_seconds)
        print(f"\nparsed status: {status}")
        print(f"parsed detail: {detail}")
    finally:
        remove_worktree(repo_path, wt2)


if __name__ == "__main__":
    main()
