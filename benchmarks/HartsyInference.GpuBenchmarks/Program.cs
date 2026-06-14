using System.Reflection;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

namespace HartsyInference.GpuBenchmarks;

/// <summary>Entry point for HartsyInference.GpuBenchmarks.
///
/// <para>BenchmarkDotNet is invoked through <see cref="BenchmarkSwitcher"/> so the user can filter
/// benchmarks via the command line: <c>dotnet run -c Release -- --filter '*MatMul*'</c>. Without a
/// filter, every benchmark in the assembly runs. Hardware/software fingerprints are NOT captured here
/// — they are captured by <c>benchmarks/run_benchmarks.sh</c> which orchestrates this binary together
/// with the Python baseline. Running this binary in isolation is fine for development but does not
/// produce the full result-directory layout described in <c>docs/Research/PROFILING_METHODOLOGY.md</c>.</para></summary>
public static class Program
{
    /// <summary>Runs whatever benchmarks BenchmarkSwitcher dispatches based on the CLI args.</summary>
    public static int Main(string[] args)
    {
        Assembly assembly = typeof(Program).Assembly;

        IConfig config = new GpuBenchmarkConfig();
        BenchmarkSwitcher switcher = BenchmarkSwitcher.FromAssembly(assembly);
        switcher.Run(args, config);
        return 0;
    }
}
