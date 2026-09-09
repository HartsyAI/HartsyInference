# Contributing

Use the .NET 10 SDK and git. GPU tests additionally require the relevant hardware, compiled kernels,
and sometimes checkpoints; libraries target net8.0/net10.0, tests net10.0.

```bash
git clone https://github.com/HartsyAI/HartsyInference.git
cd HartsyInference
dotnet build
dotnet test --filter "Category!=SyntheticSmoke&Category!=Integration&Category!=GpuIntegration&Category!=Slow&Network!=Real"
```

Read [AGENTS.md](AGENTS.md), [code style](docs/CODE_STYLE.md), and the matching
[specialist instructions](docs/Agents/AGENTS.md#task-routing). Test traits require explicit filtering;
there is no dedicated CPU/GPU CI workflow. Run real-weight/GPU checks deliberately for affected paths.

Keep Engine as the load/generate authority, preserve package/API boundaries, and validate silent numerical,
format, ownership, and concurrency behavior. Do not add a second orchestration layer in a consumer.

For PRs, describe the problem, resulting behavior, and verification. For bugs, include a minimal
reproduction plus OS, .NET, GPU/driver, checkpoint, and settings. Benchmark changes against a matched
external baseline; record measurements in [scoreboards](benchmarks/scoreboards/).
Kernel source changes must include rebuilt artifacts; see [KERNEL.md](docs/Agents/KERNEL.md).
Contributions are under the [MIT License](LICENSE).
