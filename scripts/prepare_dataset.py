"""Select a deterministic SWE-bench Lite subset and persist it to the instance manifest."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.dataset import filter_excluded_repos, load_swebench_lite, save_manifest, select_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"Loading SWE-bench Lite ({cfg.dataset.name})...")
    all_instances = load_swebench_lite()
    print(f"Loaded {len(all_instances)} instances.")

    eligible = filter_excluded_repos(all_instances, cfg.dataset.exclude_repos)
    print(f"{len(eligible)} instances remain after excluding {cfg.dataset.exclude_repos}.")

    subset = select_subset(eligible, cfg.dataset)
    save_manifest(subset, cfg.dataset.manifest_path)

    print(f"Selected {len(subset)} instances (seed={cfg.dataset.seed}).")
    print(f"Manifest written to {cfg.dataset.manifest_path}")


if __name__ == "__main__":
    main()
