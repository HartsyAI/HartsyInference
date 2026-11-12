using System.Diagnostics;
using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Residency;

/// <summary>
/// A weight preload that asks for more than the card can hold must not leave it full. On Windows a process that holds more than about 8 GiB of a
/// 12 GiB card (measured on an RTX 3060: the driver's video-memory budget collapses from 11.3 to 5.4 GB and <c>cuMemGetInfo</c> reports 0 free
/// with 4 GB still physically free) goes on allocating out of system memory without an error, and the next forward ran 35 times slower or did not
/// finish (a 1.3B TRELLIS.2 flow: 5 s clean, over 200 s). The preload now stops when the device reports itself full, gives memory back and leaves
/// the rest of the weights on the host. Benchmark lane: it allocates about a card's worth of memory.
/// </summary>
[Trait("Category", "GpuBenchmark")]
public sealed class PreloadUnderPressureGpuTests
{
    private const long WeightBytes = 64L << 20;

    private readonly ITestOutputHelper _output;

    public PreloadUnderPressureGpuTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public unsafe void PreloadingMoreThanTheCardHolds_LeavesRoomAndTheNextForwardStaysFast()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        using CudaBackend backend = new(0, PtxDir());
        IBackend device = backend;
        (long freeBefore, long total) = device.GetVramInfo();
        int count = (int)(freeBefore / WeightBytes);

        // Distinct weights over one host buffer: the device holds `count` copies, the host one.
        nint host = Marshal.AllocHGlobal((nint)WeightBytes);
        new Span<byte>((void*)host, (int)WeightBytes).Clear();
        List<Tensor> weights = [];
        try
        {
            for (int i = 0; i < count; i++)
                weights.Add(new Tensor((void*)host, new TensorShape(WeightBytes / sizeof(float)), DType.F32));

            Stopwatch preload = Stopwatch.StartNew();
            device.PreloadWeights(weights);
            device.Sync();
            long freeAfter = device.GetVramInfo().FreeBytes;
            _output.WriteLine($"asked for {(long)count * WeightBytes >> 20} MB of weights on a card with {freeBefore >> 20} MB free of {total >> 20}; "
                + $"after the preload {freeAfter >> 20} MB free, {preload.Elapsed.TotalSeconds:F1} s");

            // The room left must hold the activations of a large op, which is what stalls on an overcommitted card. The bound follows from the
            // 2048 MB vram.preloadHeadroomMb default; the guard is Windows-only, so elsewhere the preload may legitimately fill the card.
            if (OperatingSystem.IsWindows())
            {
                Assert.True(freeAfter >= 1536L << 20, $"the preload left the device full: {freeAfter >> 20} MB free");
            }

            const int n = 12288;
            using Tensor a = Filled(n, 0.5f), b = Filled(n, 0.25f);
            using Tensor c = new(new TensorShape(n, n), DType.F32);
            Stopwatch op = Stopwatch.StartNew();
            device.MatMul(c, a, b);
            device.Sync();
            _output.WriteLine($"{n}x{n} matmul after the preload: {op.Elapsed.TotalSeconds:F1} s");
            Assert.True(op.Elapsed.TotalSeconds < 60, $"the matmul took {op.Elapsed.TotalSeconds:F0} s");
            Assert.Equal(n * 0.5f * 0.25f, c.AsReadOnlySpan<float>()[0], 1);
        }
        finally
        {
            device.FreeWeights(weights);
            foreach (Tensor weight in weights)
                weight.Dispose();
            Marshal.FreeHGlobal(host);
        }
    }

    private static Tensor Filled(int n, float value)
    {
        Tensor t = new(new TensorShape(n, n), DType.F32);
        t.AsSpan<float>().Fill(value);
        return t;
    }

    private static string PtxDir()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }

        throw new DirectoryNotFoundException("src/HartsyInference.Cuda/Ptx not found above the test output directory.");
    }
}
