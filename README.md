# Reliability-Aware Speculative Execution for Coding Agents

Prototype system testing whether token-level confidence combined with sandboxed
dry-run verification of speculatively-branched actions predicts agent trajectory
failure on SWE-bench Lite. Two modes are supported: **v0** (sequential, single
model, no speedup) and **v1** (parallel speculator model, measurable speedup).
The reliability signal (confidence + sandbox outcome + gate label) is the primary
output in both modes; speedup is only measured in v1.

## Setup

- **Python**: 3.10+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- **Docker / SWE-bench harness**: dataset loading and sandbox verification go
  through the official `swebench` package. `sandbox.py` drives verification via
  `git worktree` against a repo checkout already present in the running
  container, rather than spinning up a fresh Docker container per candidate —
  it still assumes the same Linux container/image conventions the `swebench`
  harness expects. `swebench`'s harness module imports the Unix-only `resource`
  module, so this project **only runs on Linux** (native or WSL); it cannot be
  imported on native Windows.
- **GPU**: an accelerator capable of hosting `Qwen2.5-Coder-7B-Instruct` (Actor)
  is required for any real run; v1 additionally hosts `Qwen2.5-Coder-1.5B-Instruct`
  (Speculator) concurrently. Model serving is plain HuggingFace `transformers`
  (`generate(output_scores=True)` for per-token logprobs) — no `vllm` server is
  required, though the engineering constraints allow either.

## Config field reference

All parameters referenced anywhere in this project are config fields — nothing
is hardcoded in source. See `configs/default.yaml`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `mode` | str | `v0` | `v0` (sequential) or `v1` (parallel speculator) |
| `dataset.name` | str | `swebench_lite` | Dataset identifier (informational) |
| `dataset.subset_size` | int | `50` | Number of instances to sample |
| `dataset.seed` | int | `42` | RNG seed for subset sampling |
| `dataset.manifest_path` | str | `data/instance_manifest.json` | Where selected instance IDs are persisted |
| `dataset.exclude_repos` | list[str] | `[matplotlib/matplotlib, scikit-learn/scikit-learn, astropy/astropy]` | Repos dropped before sampling — these need compiled C extensions this sandbox can't build (see Limitations) |
| `models.actor` | str | `Qwen/Qwen2.5-Coder-7B-Instruct` | Actor HF model name |
| `models.speculator` | str | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Speculator HF model name (v1 only) |
| `signal.confidence_threshold` | float \| null | `null` | Mean-logprob gate threshold; set via `calibrate_threshold.py` before a full run |
| `signal.k_default` | int | `1` | Branching width when confident |
| `signal.k_uncertain` | int | `3` | Branching width when uncertain |
| `signal.alt_temperature` | float | `0.8` | Sampling temperature for alternative candidates |
| `sandbox.test_timeout_seconds` | int | `120` | Per-candidate test subset timeout |
| `sandbox.worktree_base_dir` | str | `/tmp/sbx_worktrees` | Base dir for ephemeral `git worktree` checkouts (and the persistent repo clone cache, in a `_repos/` subdir) |
| `logging.log_dir` | str | `results/logs` | Per-instance JSONL log directory |
| `logging.log_level` | str | `INFO` | Log verbosity |
| `output.results_dir` | str | `results/reports` | Where `analyze_results.py` writes `summary.md` |
| `calibration.num_instances` | int | `5` | Instances used by `calibrate_threshold.py` |
| `pipeline.max_steps` | int | `30` | Safety cap on trajectory length (real trajectories end at `submit_patch`) |

## Run order

```bash
# 1. Select and persist the instance subset
python scripts/prepare_dataset.py --config configs/default.yaml

# 2. Run the Actor on a handful of instances to see the confidence distribution
python scripts/calibrate_threshold.py --config configs/default.yaml

# 3. Edit configs/default.yaml: set signal.confidence_threshold from the printed table

# 4. Smoke-test the pipeline on 2 instances before committing to a full run
python scripts/run_experiment.py --config configs/default.yaml --limit 2

# 5. Run the full subset
python scripts/run_experiment.py --config configs/default.yaml

# 6. Compute the four reliability metrics
python scripts/analyze_results.py --config configs/default.yaml
```

## Log schema

One JSONL file per instance at `results/logs/{instance_id}.jsonl`, one line per
step plus a trailing summary line. See spec Section 9 / `src/logging_utils.py`
for the exact schema (`StepRecord`, `CandidateRecord`, `InstanceSummary`).

## Metrics reference

Computed by `scripts/analyze_results.py` from the logged trajectories:

1. **Match rate** — of steps where more than one candidate was sandboxed
   (branching occurred), the fraction where an alternative/Speculator candidate
   exactly matched the realized action. High values mean the confidence gate is
   branching in cases where the branch turns out to reproduce what the Actor
   would have done anyway.
2. **Confidence separation** — mean `actor_confidence` for resolved vs.
   unresolved trajectories, plus a single AUROC scoring mean trajectory
   confidence against the resolved/unresolved label. Answers: does the model's
   own token-level confidence carry information about whether the trajectory
   will ultimately succeed?
3. **Sandbox predictive accuracy** — agreement rate between a candidate's
   sandbox pass/fail and the trajectory's eventual resolved/unresolved status.
   Answers: does a single sandboxed dry-run predict the final outcome?
4. **Retrospective catch rate** — among unresolved trajectories, the fraction
   that had at least one `flag` gate label before the trajectory ended. This is
   explicitly **retrospective and non-causal**: the gate never altered Actor
   behavior in this version, so this measures what the gate *would* have
   caught, not what it *did* prevent.

A separate (non-metric) table reports total extra sandbox calls and total extra
wall-clock ms, summed over the subset.

## Limitations

- The decision gate (Section 8) is observational-only — it never alters Actor
  behavior in this version, so metric 4 above is retrospective, not a causal
  claim about prevented failures.
- No synthetic fault injection: only natural model failures are observed, at
  small n (`dataset.subset_size` defaults to 50).
- Single signal type: only token log-prob confidence is used. No other
  uncertainty signal (ensembling, self-consistency, etc.) is explored.
- Depth-1 speculation only: no multi-step speculation trees are built.
- No per-instance environment provisioning: sandbox verification runs against a
  shared Python environment with a best-effort, once-per-`(repo, version)`
  dependency install (`sandbox.py::_install_repo_dependencies`), not the
  official harness's per-instance conda/Docker environments. This works for
  pure-Python repos (confirmed against Django's golden patches) but not repos
  needing compiled C extensions (matplotlib, scikit-learn, astropy), which are
  excluded from sampling via `dataset.exclude_repos` rather than silently
  producing unreliable `fail` results.

## v0 -> v1 delta

Setting `mode: v1` changes:

- The Speculator becomes a separate, smaller model (`models.speculator`)
  running **concurrently** with the Actor, rather than the Actor resampled at
  higher temperature.
- Branching width `k` is gated on the **Speculator's** confidence, not the
  Actor's.
- Speculator candidates are sandboxed while the Actor is still generating
  (`src/speculator.py::run_parallel_v1` kicks off sandbox verification futures
  as soon as each Speculator candidate is ready, before blocking on the Actor's
  result).
- When the Actor's realized action exact-matches the Speculator's top
  candidate, the sandbox result is reused (cache hit) instead of re-verifying.
  Mismatched candidates' sandbox results are *not* discarded — they're logged
  as reliability evidence (this deviates from the source papers, which discard
  on mismatch).
- Latency is measured and reported: `scripts/run_experiment.py` reports the
  measured `E[Ts]/E[Tseq]` against the theoretical
  `(1-p)/(1+p) · α/(α+β)` as T→∞, from *Speculative Actions*
  (arXiv:2510.04371, Proposition 1), where `p` is the empirical Speculator
  top-1 match rate and `α`/`β` are the Speculator/Actor token generation
  rates. v0 reports no speedup, by construction (Section 5, step 7).
