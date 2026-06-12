# tests/data — generated data and run artifacts

This directory is the repository-local home for **data this project
generates** for debugging, testing, and benchmarking. It never holds
models or datasets downloaded from Hugging Face (those stay in the HF
cache); it holds only artifacts our own scripts produce.

Conventions:

- **Small generated fixtures** (text fixtures, small tensors) may be
  committed directly here.
- **Large generated artifacts are never committed** (see `.gitignore`):
  they are produced by a script in `scripts/` that (a) defaults its output
  here and (b) **reuses an existing file instead of regenerating** when one
  is already present. Delete the file (or pass the script's `--force`) to
  regenerate.
- **Run artifacts** (teed transcripts, profiler output, IR dumps,
  disposable probes) from measurement sessions go under `runs/<label>/`
  (gitignored). Milestone memos cite these paths; the artifacts themselves
  are session-local — every decision-carrying one must have a regeneration
  recipe in the memo or in `scripts/README.md`.
- **`/tmp` is never a deposit location.** Background-task buffers land
  there transiently and the mount is `noexec` and wiped; nothing written
  to `/tmp` may be cited or relied on. Anything worth keeping is teed or
  written here at launch time. `scripts/` tools follow the same rule:
  their file outputs default to subdirectories of this directory
  (`bundles/`, `ir_dump/`, …), overridable by CLI flags.

Current contents of record:

| Path | Producer | Regeneration |
|---|---|---|
| `bundles/swimir_de_steps0.pt` | `scripts/capture_index_distributions.py` | `--train-steps 0` (reuses the file if present; `--force` to regenerate) |
| `bundles/swimir_de_steps150.pt` | same | `--train-steps 150` |

Caveat from the M11/M13 evidence rules: regenerated bundles contain
*different records* (training nondeterminism), so timings against a
regenerated bundle are a new baseline, not comparable to transcripts taken
against the old file. The two bundles above are the M11 captures of record,
relocated here from the original session directory at the scripts/data
self-containment refactor — keep them if cross-referencing the M11/M13
memo numbers matters.
