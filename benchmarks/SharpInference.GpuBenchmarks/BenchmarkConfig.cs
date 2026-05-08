using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Environments;
using BenchmarkDotNet.Exporters;
using BenchmarkDotNet.Exporters.Csv;
using BenchmarkDotNet.Exporters.Json;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Loggers;
using BenchmarkDotNet.Reports;

namespace SharpInference.GpuBenchmarks;

/// <summary>Common BenchmarkDotNet config for every GPU microbench. Pinned per the methodology in
/// <c>docs/Research/PROFILING_METHODOLOGY.md</c>: 1 warmup + 5 trials, JSON+Markdown+CSV exporters,
/// MemoryDiagnoser disabled (GPU memory is captured separately by the run script via <c>nvidia-smi</c>).
///
/// <para>Why these specific values: the master plan demands N=5 trials for Welch's t-test (df=4) at
/// α=0.01. BenchmarkDotNet's default warmup runs the kernel 5+ times before counting, which is overkill
/// for GPU work and lengthens runs; one warmup pass is enough to amortize JIT + first-launch cost.</para></summary>
public sealed class GpuBenchmarkConfig : ManualConfig
{
    /// <summary>Number of measurement trials per benchmark. Matches the rigor commitment.</summary>
    public const int Trials = 5;

    /// <summary>Number of warmup invocations.</summary>
    public const int WarmupCount = 1;

    public GpuBenchmarkConfig()
    {
        AddJob(Job.Default
            .WithStrategy(RunStrategy.Throughput)
            .WithWarmupCount(WarmupCount)
            .WithIterationCount(Trials)
            .WithLaunchCount(1)
            .WithUnrollFactor(1)
            .WithInvocationCount(1));

        AddExporter(MarkdownExporter.GitHub);
        AddExporter(JsonExporter.Full);
        AddExporter(CsvExporter.Default);

        AddLogger(ConsoleLogger.Default);
        AddColumnProvider(BenchmarkDotNet.Columns.DefaultColumnProviders.Instance);

        WithSummaryStyle(SummaryStyle.Default
            .WithMaxParameterColumnWidth(50));

        // No MemoryDiagnoser: GPU memory tracking happens via the run script's nvidia-smi poll.
        // Adding it here would only show managed allocations, which are mostly proxy objects.
    }
}
