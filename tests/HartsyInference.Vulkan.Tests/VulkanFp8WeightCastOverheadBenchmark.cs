using System.Diagnostics;
using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Investigates the gap between isolated coopmat2 GPU-only-time benchmarks (1.5-10 μs per GEMM at
/// comparable shapes) and the real Krea2 e2e run's measured Linear cost (~82-135 ms/call average) — a
/// 10-90x discrepancy the "keep tuning coopmat2" pass needs explained before any further kernel-level work
/// is worth attempting. Leading hypothesis: Krea2 runs with <see cref="VulkanBackend.CacheWeightCasts"/>
/// disabled (`[Krea2Recipe] CacheWeightCasts disabled (fp8-resident, transient per-GEMM dequant)` — a
/// deliberate VRAM tradeoff, same choice made on the CUDA backend, since caching every layer's FP8->F16
/// cast would roughly double the ~13 GB fp8 checkpoint's VRAM footprint), meaning EVERY Linear call pays a
/// fresh FP8->F16 cast dispatch for its weight instead of reusing a cached cast. CUDA likely doesn't pay an
/// equivalent cost: <c>CudaBackend.EnableNativeFp8Gemm</c> defaults ON for Ada+ GPUs (the 4090 qualifies),
/// meaning its GEMM kernel consumes FP8 weights DIRECTLY — no separate cast dispatch, no separate cast-
/// result VRAM allocation. Vulkan's coopmat2 kernel has no equivalent fused-FP8-load path, so it always
/// needs the separate cast. This benchmark isolates whether that gap alone explains the real-world Linear
/// cost, using MANY DISTINCT weight tensors (not one reused tensor) to mimic Krea2's 28 distinct DiT
/// blocks — a single reused tensor would let the allocator's pooling hide the "first use of a unique
/// weight" cost this benchmark exists to measure.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanFp8WeightCastOverheadBenchmark
{
    private readonly ITestOutputHelper _out;
    public VulkanFp8WeightCastOverheadBenchmark(ITestOutputHelper output) => _out = output;

    private static bool VulkanAvailable()
    {
        try { using VulkanInstance i = new(); return i.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    [Fact]
    public unsafe void Measure_ManyDistinctFp8Weights_CachedVsTransientCast()
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16 || !backend.Vk.HasCooperativeMatrix2)
        {
            _out.WriteLine("SKIPPED: no F16/coopmat2 support");
            return;
        }

        // Krea2's REAL DiT FFN shape (SwiGlu's up/gate projection): M=jointSeq (imgSeq=4096+txtSeq=13),
        // K=hidden, N=ffnInner — matches the exact shape that produced the 134,643,712-byte SwiGlu
        // allocation in the OOM root-cause investigation. Using the real scale matters: GEMM compute time
        // scales with M*K*N, and an earlier, smaller-shape version of this benchmark (M=512,N=8192) showed
        // only a 1.18x transient-vs-cached difference — nowhere near the ~10-90x gap this benchmark exists
        // to explain, raising the question of whether shape/compute-scale (not weight-cast overhead) is
        // the real remaining factor.
        const int M = 4109, K = 6144, N = 16384;   // K corrected to Krea2's real HiddenSize (48 heads x 128 headDim)
        // Fewer than Krea2's real 28 blocks — this synthetic test doesn't have the real run's other VRAM
        // consumers (the ~13 GB checkpoint, text encoder, VAE), so caching all 28 blocks' F16 casts
        // simultaneously hits the same real OOM this investigation already root-caused (confirmed: this
        // benchmark's shape hit the identical `size=134643712` allocation failure at NumDistinctBlocks=28).
        // 4 distinct blocks still captures "many distinct weights, real per-GEMM shape" without that risk.
        const int NumDistinctBlocks = 4;
        const int NumSimulatedSteps = 2;    // mimics a 2-step denoise loop reusing the SAME weights every step

        Tensor input = new(new TensorShape(M, K), DType.F16);
        Half* ip = (Half*)input.DataPointer;
        for (long i = 0; i < (long)M * K; i++) ip[i] = (Half)(((i * 13) % 17 - 8) * 0.01f);

        Tensor[] weightsFp8 = new Tensor[NumDistinctBlocks];
        Tensor output = new(new TensorShape(M, N), DType.F16);
        for (int b = 0; b < NumDistinctBlocks; b++)
        {
            Tensor wF32 = new(new TensorShape(N, K), DType.F32);
            float* wp = (float*)wF32.DataPointer;
            for (long i = 0; i < (long)N * K; i++) wp[i] = (((i * 7 + b * 31) % 13) - 6) * 0.01f;
            weightsFp8[b] = wF32.CastTo(DType.F8E4M3);
            weightsFp8[b].Fp8ScaleFactor = 1.0f;
            wF32.Dispose();
        }
        backend.PreloadWeights(weightsFp8);

        void RunAllBlocksOnce()
        {
            for (int b = 0; b < NumDistinctBlocks; b++)
                backend.Linear(output, input, weightsFp8[b], null);
        }

        // Warm up (pipeline builds), matching every other benchmark's convention.
        for (int i = 0; i < NumSimulatedSteps; i++) RunAllBlocksOnce();
        backend.Sync();

        // Measured: CacheWeightCasts=false (Krea2's real, deliberate config).
        backend.CacheWeightCasts = false;
        Stopwatch swTransient = Stopwatch.StartNew();
        for (int i = 0; i < NumSimulatedSteps; i++) RunAllBlocksOnce();
        backend.Sync();
        swTransient.Stop();

        // Measured: CacheWeightCasts=true (cast once per weight, reused every subsequent call/step).
        backend.CacheWeightCasts = true;
        for (int i = 0; i < NumSimulatedSteps; i++) RunAllBlocksOnce();   // warm the cache
        backend.Sync();
        Stopwatch swCached = Stopwatch.StartNew();
        for (int i = 0; i < NumSimulatedSteps; i++) RunAllBlocksOnce();
        backend.Sync();
        swCached.Stop();

        int callsPerRun = NumDistinctBlocks * NumSimulatedSteps;
        double transientMsPerCall = swTransient.Elapsed.TotalMilliseconds / callsPerRun;
        double cachedMsPerCall = swCached.Elapsed.TotalMilliseconds / callsPerRun;
        _out.WriteLine($"Shape (M={M},K={K},N={N}), {NumDistinctBlocks} distinct FP8 weights x {NumSimulatedSteps} simulated steps ({callsPerRun} calls/run):");
        _out.WriteLine($"  CacheWeightCasts=false (transient, Krea2's real config): {swTransient.Elapsed.TotalMilliseconds:F1} ms total, {transientMsPerCall:F3} ms/call");
        _out.WriteLine($"  CacheWeightCasts=true  (cast-once, reused):              {swCached.Elapsed.TotalMilliseconds:F1} ms total, {cachedMsPerCall:F3} ms/call");
        _out.WriteLine($"  Ratio (transient/cached): {transientMsPerCall / cachedMsPerCall:F2}x");

        input.Dispose(); output.Dispose();
        foreach (Tensor w in weightsFp8) w.Dispose();
    }

    /// <summary>Weight-cast caching is ruled out (previous test: 1.01x ratio at real shape) — this tests the
    /// NEXT hypothesis: real Krea2 blocks aren't independent back-to-back Linear calls, they're a genuine
    /// DEPENDENCY CHAIN (gate/up Linear -> Silu -> Mul -> down Linear, each op's output feeding the next
    /// op's input, exactly <c>Krea2Block.SwiGlu</c>'s structure), repeated per block. <c>Dispatch()</c>
    /// inserts an unconditional global <c>VkMemoryBarrier2</c> after EVERY dispatch (see ROADMAP.md's
    /// "per-dispatch barrier scoping" entry) — a genuine dependency chain means the GPU can't overlap
    /// dispatch N+1's execution with dispatch N's completion (real data dependency, not just barrier
    /// conservatism), unlike this file's other benchmark, which calls Linear independently on the SAME
    /// input/output every time (no forced serialization beyond whatever the barrier itself costs). If this
    /// chained version's ms/call is much closer to the real run's ~82-135 ms/call than the independent
    /// version's ~17 ms/call, dependency-chain serialization (not weight-cast, not raw GEMM throughput) is
    /// the dominant remaining real-world cost.</summary>
    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    public unsafe void Measure_SwiGluDependencyChain_RealShapes(int NumDistinctBlocks)
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16 || !backend.Vk.HasCooperativeMatrix2)
        {
            _out.WriteLine("SKIPPED: no F16/coopmat2 support");
            return;
        }

        const int M = 4109, Hidden = 6144, FfnInner = 16384;
        const int NumSimulatedSteps = 2;

        Tensor input = new(new TensorShape(M, Hidden), DType.F16);
        Half* ip = (Half*)input.DataPointer;
        for (long i = 0; i < (long)M * Hidden; i++) ip[i] = (Half)(((i * 13) % 17 - 8) * 0.01f);

        Tensor[] wGate = new Tensor[NumDistinctBlocks];
        Tensor[] wUp = new Tensor[NumDistinctBlocks];
        Tensor[] wDown = new Tensor[NumDistinctBlocks];
        for (int b = 0; b < NumDistinctBlocks; b++)
        {
            Tensor gF32 = new(new TensorShape(FfnInner, Hidden), DType.F32);
            Tensor uF32 = new(new TensorShape(FfnInner, Hidden), DType.F32);
            Tensor dF32 = new(new TensorShape(Hidden, FfnInner), DType.F32);
            float* gp = (float*)gF32.DataPointer; float* up = (float*)uF32.DataPointer;
            for (long i = 0; i < (long)FfnInner * Hidden; i++)
            {
                gp[i] = (((i * 7 + b * 31) % 13) - 6) * 0.01f;
                up[i] = (((i * 11 + b * 17) % 13) - 6) * 0.01f;
            }
            float* dp = (float*)dF32.DataPointer;
            for (long i = 0; i < (long)Hidden * FfnInner; i++) dp[i] = (((i * 5 + b * 23) % 13) - 6) * 0.01f;

            wGate[b] = gF32.CastTo(DType.F8E4M3); wGate[b].Fp8ScaleFactor = 1.0f; gF32.Dispose();
            wUp[b] = uF32.CastTo(DType.F8E4M3); wUp[b].Fp8ScaleFactor = 1.0f; uF32.Dispose();
            wDown[b] = dF32.CastTo(DType.F8E4M3); wDown[b].Fp8ScaleFactor = 1.0f; dF32.Dispose();
        }
        backend.PreloadWeights(wGate); backend.PreloadWeights(wUp); backend.PreloadWeights(wDown);
        backend.CacheWeightCasts = false;   // matches Krea2's real config — already shown not to matter much on its own

        // Fresh tensors per block, disposed immediately after use — matching Krea2Block.SwiGlu's REAL
        // pattern exactly (see that method: `Tensor g = new(...); ... g.Dispose();` for every intermediate,
        // never a reused tensor object across blocks/steps). An earlier version of this test reused g/u/
        // silu/gated/down across ALL blocks/steps, which turned out to trigger a real but SEPARATE bug
        // (VulkanGpuTransferHelper.CacheActivation overwrites a tensor's cached-buffer dictionary entry
        // without freeing the buffer it replaces, when the SAME tensor object is used as a non-in-place
        // op's output more than once) — that pattern doesn't occur in real Krea2Block code (which never
        // reuses a tensor object as repeated non-in-place output), so it was an artifact of the test's own
        // construction, not a real explanation for Krea2's e2e slowdown. Fixed here to test the real pattern.
        TensorShape ffShape = new(M, FfnInner);
        TensorShape hiddenShape = new(M, Hidden);
        void RunChainedSwiGluAllBlocks()
        {
            for (int b = 0; b < NumDistinctBlocks; b++)
            {
                Tensor g = new(ffShape, DType.F16);
                Tensor u = new(ffShape, DType.F16);
                backend.Linear(g, input, wGate[b], null);
                backend.Linear(u, input, wUp[b], null);
                Tensor silu = new(ffShape, DType.F16);
                backend.Silu(silu, g);
                g.Dispose();
                Tensor gated = new(ffShape, DType.F16);
                backend.Mul(gated, silu, u);
                silu.Dispose(); u.Dispose();
                Tensor down = new(hiddenShape, DType.F16);
                backend.Linear(down, gated, wDown[b], null);
                gated.Dispose();
                down.Dispose();
            }
        }

        (_, _, int blocksAfterWarmup1Step, _) = backend.MemoryStats;
        for (int i = 0; i < NumSimulatedSteps; i++) RunChainedSwiGluAllBlocks();
        backend.Sync();
        (long usedAfterWarmup, long reservedAfterWarmup, int blocksAfterWarmup, _) = backend.MemoryStats;

        Stopwatch sw = Stopwatch.StartNew();
        for (int i = 0; i < NumSimulatedSteps; i++) RunChainedSwiGluAllBlocks();
        backend.Sync();
        sw.Stop();
        (long usedAfterMeasured, long reservedAfterMeasured, int blocksAfterMeasured, _) = backend.MemoryStats;

        int linearCallsPerRun = NumDistinctBlocks * NumSimulatedSteps * 3;   // gate + up + down per block
        double msPerLinearCall = sw.Elapsed.TotalMilliseconds / linearCallsPerRun;
        _out.WriteLine($"SwiGlu dependency chain (M={M},Hidden={Hidden},FfnInner={FfnInner}), {NumDistinctBlocks} blocks x {NumSimulatedSteps} steps ({linearCallsPerRun} Linear calls/run):");
        _out.WriteLine($"  {sw.Elapsed.TotalMilliseconds:F1} ms total, {msPerLinearCall:F3} ms/Linear-call");
        _out.WriteLine($"  Allocator blocks: after 1st warmup pass={blocksAfterWarmup1Step}, after full warmup={blocksAfterWarmup}, after measured pass={blocksAfterMeasured}");
        _out.WriteLine($"  VRAM used: after warmup={usedAfterWarmup / (1024.0 * 1024):F1} MB, after measured={usedAfterMeasured / (1024.0 * 1024):F1} MB (reserved={reservedAfterMeasured / (1024.0 * 1024):F1} MB)");
        _out.WriteLine($"  Compare: independent back-to-back Linear (previous test) was ~17.3-17.5 ms/call at similar scale;");
        _out.WriteLine($"  real Krea2 e2e run measured ~82-135 ms/call.");

        input.Dispose();
        foreach (Tensor w in wGate) w.Dispose();
        foreach (Tensor w in wUp) w.Dispose();
        foreach (Tensor w in wDown) w.Dispose();
    }
}
