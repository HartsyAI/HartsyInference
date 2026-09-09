# HartsyInference — agent entry point

Pure C# inference libraries targeting net8.0/net10.0. Engine owns model resolution, loading,
caching, placement, and generation. CLI, HTTP API, and SwarmUI are consumers of that service layer.

Read once per task:

1. [Documentation rules](docs/README.md).
2. [Code style](docs/CODE_STYLE.md) — mandatory.
3. [Architecture and task routing](docs/Agents/AGENTS.md), then the one matching specialist file.

Load only task-relevant sections of [open work](docs/Checklists/ROADMAP.md),
[model status](docs/Checklists/MODEL_STATUS.md), and [research](docs/Research/).
Before debugging wrong output, crashes, or slowness, search
[Troubleshooting](docs/Checklists/TROUBLESHOOTING.md) for the affected component.
[Parity evidence](docs/Checklists/PARITY_VERIFICATION.md) distinguishes numerical verification from e2e output.
Do not read the whole documentation tree for an ordinary task.

Code lives in src/ (one folder per package), tests/, and benchmarks/.
CUDA sources/artifacts live in HartsyInference.Cuda/Kernels and Ptx; Vulkan in Shaders and Spirv.
