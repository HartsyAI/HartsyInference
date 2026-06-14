using BenchmarkDotNet.Attributes;
using HartsyInference.Cuda;

namespace HartsyInference.GpuBenchmarks;

/// <summary>Microbenchmark for the GPU memory alloc/free fast path. The plan in
/// [`CUDA_PERFORMANCE_PLAN.md`](../../docs/Research/CUDA_PERFORMANCE_PLAN.md) hypothesizes (H6) that
/// memory pooling alone provides &lt; 1.1× speedup on a workload already using
/// <c>cuMemAllocAsync</c>. This benchmark is the data point for that hypothesis: it characterizes
/// the per-call cost of <c>cuMemAlloc</c> (sync) vs <c>cuMemAllocAsync</c> at common size classes.
/// Phase B4.4 will add a third entrant — a size-class pool — and re-run this comparison.</summary>
[Config(typeof(GpuBenchmarkConfig))]
public class MemoryAllocFreeBenchmarks
{
    private BenchmarkFixture? _fixture;

    [ParamsSource(nameof(SizeSource))]
    public int SizeIndex { get; set; }
    public IEnumerable<int> SizeSource => Enumerable.Range(0, _sizes.Length);

    /// <summary>Common diffusion-pipeline allocation sizes. Small to large; covers the typical hot
    /// path (per-op activation buffers) plus the large weight-cache case.</summary>
    private static readonly nuint[] _sizes =
    [
        4096,                          // tiny — single timestep embedding
        4 * 1024 * 1024,              // 4 MB — small activation
        64 * 1024 * 1024,             // 64 MB — DiT block hidden state
        256 * 1024 * 1024,            // 256 MB — VAE intermediate
        1024 * 1024 * 1024,           // 1 GB — large activation / im2col scratch
    ];

    [GlobalSetup]
    public void Setup() => _fixture = new BenchmarkFixture();

    [GlobalCleanup]
    public void Cleanup() => _fixture?.Dispose();

    /// <summary>Sync alloc + sync free. Conservative baseline; matches what bare
    /// <c>cuMemAlloc</c> + <c>cuMemFree</c> would do in a naive implementation.</summary>
    [Benchmark]
    public void AllocFree_Sync()
    {
        ulong p = CudaMemory.Allocate(_sizes[SizeIndex]);
        CudaMemory.Free(p);
    }

    /// <summary>Async alloc + async free on the backend's compute stream. Current production path.</summary>
    [Benchmark]
    public void AllocFree_Async()
    {
        nuint size = _sizes[SizeIndex];
        nint stream = _fixture!.Backend.Stream.Handle;
        ulong p = CudaMemory.AllocateAsync(size, stream);
        CudaMemory.FreeAsync(p, stream);
        _fixture.Sync();
    }
}
