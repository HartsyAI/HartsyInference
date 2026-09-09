# Local benchmark outputs

The harness writes run directories here. Outputs are ignored unless explicitly selected for archival; this README does not imply raw results are committed.

Keep hardware/software/digest metadata, raw measured trials, C#/reference logs and comparison outputs together. See [benchmark guide](../README.md) and [methodology](../../docs/Research/PROFILING_METHODOLOGY.md). Actual CSV schemas are defined by the scripts, not a copied table.

A speedup ratio above one favors HartsyInference in analyze.py; inspect matching inputs, sample counts, effect direction and quality before interpreting significance. Archive the complete run durably and link its provenance from the canonical scoreboard. Smoke runs are harness checks, not publication evidence.
