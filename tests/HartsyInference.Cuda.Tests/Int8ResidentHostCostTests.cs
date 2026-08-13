using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Times the HOST-side driver calls the resident-int8 chain makes per <c>Linear</c>, in isolation. The
/// chain runs ~2,700 times per LTX-2.5 step, so a per-call driver round trip that looks free in a microbenchmark
/// of one is worth milliseconds per step — and the only way to know which it is, is to time it alone rather than
/// infer it from a whole-chain number (the mistake that produced the "GEMM is at the hardware wall" claim).</summary>
/// <remarks>Reports rather than asserts a threshold: absolute driver-call latency is machine- and driver-specific,
/// so a hard bound here would be a flake. The number it prints is the input to the decision.</remarks>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed class Int8ResidentHostCostTests
{
    private readonly ITestOutputHelper _output;
    public Int8ResidentHostCostTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    [Fact]
    public void ReportPerCallDriverCosts()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend(0, PtxDir());

        const int Warmup = 200, Iterations = 20000;
        for (int i = 0; i < Warmup; i++) _ = CudaMemory.GetMemInfo();
        System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < Iterations; i++) _ = CudaMemory.GetMemInfo();
        sw.Stop();
        double memInfoUs = sw.Elapsed.TotalMilliseconds * 1000.0 / Iterations;

        // The four transients RunResidentInt8 allocates and frees per row chunk, at LTX-2.5-ish sizes.
        nuint[] sizes = [64 << 20, 20 << 20, 4096, 80 << 20];
        for (int i = 0; i < Warmup; i++)
        {
            ulong p = GpuTransferHelper.AllocateDevice(sizes[i % sizes.Length]);
            GpuTransferHelper.FreeDevice(p);
        }
        backend.Sync();
        const int AllocIterations = 4000;
        sw.Restart();
        for (int i = 0; i < AllocIterations; i++)
        {
            ulong p = GpuTransferHelper.AllocateDevice(sizes[i % sizes.Length]);
            GpuTransferHelper.FreeDevice(p);
        }
        sw.Stop();
        double allocFreeUs = sw.Elapsed.TotalMilliseconds * 1000.0 / AllocIterations;

        // ~2,700 resident-int8 Linear calls per LTX-2.5 step, one GetMemInfo each, ~5 alloc/free pairs each.
        _output.WriteLine($"cuMemGetInfo:            {memInfoUs:F3} us/call  -> {memInfoUs * 2700 / 1000.0:F1} ms/step at 2700 calls");
        _output.WriteLine($"pool alloc+free pair:    {allocFreeUs:F3} us/pair -> {allocFreeUs * 2700 * 5 / 1000.0:F1} ms/step at 5 pairs/call");
    }
}
