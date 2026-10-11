# Revisions and artifacts

| Item | Revision |
|---|---|
| Round 1 engine | commit `65c13a7e` (the last code commit of PR #367 before its documentation commits; the squash commit on main is `9278347c`, which adds review fixes) |
| Round 2 engine | `main` at `9278347c` + the head of PR #369 (`a97eca66`) + the head of PR #372 (`d8546fdd`), merged with the changelog conflict resolved; built from a temporary branch since deleted. The merged PRs add review fixes (input validation for the fused QK-norm, a split-heuristic test and cap, version bumps) |
| llama.cpp | `d81235049`, CMake + Ninja, `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86` |
| Harness | Strata `bench/bench_vs_llama.py` at `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, unmodified |
| Machine | RunPod secure cloud A40 48 GB, driver 580.159.04, container `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Models | Granite 3B A800M Q4_K_M (bartowski), OLMoE-1B-7B-0924-Instruct Q4_K_M (allenai, SHA-256 `8c310f14…2f3f`, the same file as the one used in the runs), Qwen3-30B-A3B Q4_K_M (unsloth), Mixtral-8x7B-Instruct-v0.1 Q4_K_M (mradermacher, 28,448,468,384 bytes, published SHA-256 `9f8d778c…1d44`). `bootstrap.sh` checks all four by size and SHA-256. The Mixtral file used in the runs was size-checked only (its hash was not recomputed on the pod) |

Contents: `round1/` and `round2/` hold the harness summaries (`<model>.summary.md`) and the raw per-trial files (`raw-<model>/*.jsonl`);
`pod_up.py` and `bootstrap.sh` bring a pod up from this folder (it copies itself, `scripts/`, `cfg/` and `prompts/` to the pod); `scripts/` are the
pod scripts that produced the runs (`make_report.py` builds the report's tables from the summaries; `libcuptitrace.so` is built from
`cupti_trace.c`). The scripts here are the corrected versions: the originals' `run-cell.sh` swallowed the harness exit status, which let a failed
Mixtral cell read as passed during the campaign; it now fails the cell when a config has no trials or the log has errors. `MANIFEST.sha256` lists the SHA-256 of every file here. Kernel traces (CUPTI CSVs, 64 MB) are not archived.
