# Benchmarks

Use [scoreboards](scoreboards/README.md) for measurements and [profiling methodology](../docs/Research/PROFILING_METHODOLOGY.md) before adding a benchmark. Model correctness belongs in the status/parity docs.

From the repository root:

```bash
python3 -m venv benchmarks/python-baseline/.venv
benchmarks/python-baseline/.venv/bin/pip install -r benchmarks/python-baseline/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
bash benchmarks/run_benchmarks.sh --smoke --py-venv benchmarks/python-baseline/.venv
```

The pinned requirements describe this baseline environment, not the minimum supported runtime. Review them before installing on a different CUDA platform. Remove --smoke for the full workload; --skip-python permits C# iteration. Inspect script options before composing new runs.

- run_benchmarks.sh fingerprints hardware/software/digests, runs C# and Python workloads, and analyzes results.
- analyze.py joins comparable operation/shape/dtype rows and emits comparison CSV/Markdown.
- profile.sh wraps Nsight Systems; choose a current test filter from source.
- HartsyInference.GpuBenchmarks contains GPU microbenchmarks; HartsyInference.Benchmarks contains CPU benchmarks.
- python-baseline contains reference timing scripts and pinned dependencies; tests/python-reference contains correctness oracles.

Outputs under benchmarks/results are local/ignored artifacts, not a guaranteed committed archive. Preserve the exact run directory in an explicit durable artifact location and link it from the relevant scoreboard. Older missing local result links do not establish reproducibility.

Record cold/warm state, trials/dispersion, checkpoint/input hashes, GPU/driver/runtime, resolved settings, memory and a quality gate. Match saved inputs and execution precision with the reference; report skipped/failed workloads separately. Repeated trials and confidence intervals matter; a successful harness exit alone does not establish numerical parity or a speedup.
