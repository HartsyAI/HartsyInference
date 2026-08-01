using System.Diagnostics;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>The next lever after <see cref="VulkanFp8WeightCastOverheadBenchmark"/> ruled out weight-cast
/// caching, dependency-chain barrier serialization, and allocator block-count scaling as explanations for
/// the real Krea2 e2e run's ~82-135 ms/Linear-call (vs. ~33 ms/call in the isolated SwiGlu-chain benchmark
/// at the same real shape). That benchmark only exercised the FFN half of a block; this one uses the REAL
/// <see cref="Krea2Block"/> class directly (not a re-implementation) — full attention (QKV projections,
/// per-head RMSNorm, GQA, RoPE, SDPA, sigmoid output gate, out-proj) + norm/modulate/gated-residual glue +
/// FFN, at Krea2's real config (hidden=6144, ffnInner=16384, 48 heads/12 kv-heads, headDim=128) and real
/// joint sequence length (imgSeq=4096 + txtSeq=13 = 4109, matching a real 64x64-patch generation) — to see
/// whether the FULL block structure (not just FFN) reproduces the real run's per-call cost, or whether the
/// mystery gap needs GPU-level profiling (Nsight Graphics, still unavailable here) to resolve.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanKrea2BlockFullScaleBenchmark
{
    private readonly ITestOutputHelper _out;
    public VulkanKrea2BlockFullScaleBenchmark(ITestOutputHelper output) => _out = output;

    private static bool VulkanAvailable()
    {
        try { using VulkanInstance i = new(); return i.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    /// <summary>Builds one block's worth of synthetic FP8 weights with the exact key names
    /// <see cref="Krea2Block.LoadWeights"/> / <see cref="Krea2Attention.LoadWeights"/> expect.</summary>
    private static unsafe void AddSyntheticBlockWeights(
        Dictionary<string, Tensor> w, string prefix, int hidden, int ffnInner, int numHeads, int numKvHeads, int headDim, int seed)
    {
        Tensor MakeFp8(int rows, int cols, int s)
        {
            Tensor f32 = new(new TensorShape(rows, cols), DType.F32);
            float* p = (float*)f32.DataPointer;
            long n = (long)rows * cols;
            for (long i = 0; i < n; i++) p[i] = (((i * 7 + s * 31) % 13) - 6) * 0.01f;
            Tensor fp8 = f32.CastTo(DType.F8E4M3);
            fp8.Fp8ScaleFactor = 1.0f;
            f32.Dispose();
            return fp8;
        }
        Tensor MakeF32Vec(int n, int s, float scale = 0.02f)
        {
            Tensor t = new(new TensorShape(n), DType.F32);
            float* p = (float*)t.DataPointer;
            for (int i = 0; i < n; i++) p[i] = (((i + s) % 13) - 6) * scale;
            return t;
        }

        int kvWidth = numKvHeads * headDim;
        w[$"{prefix}.scale_shift_table"] = MakeF32Vec(6 * hidden, seed + 1);
        w[$"{prefix}.norm1.weight"] = MakeF32Vec(hidden, seed + 2, 0.001f);
        w[$"{prefix}.norm2.weight"] = MakeF32Vec(hidden, seed + 3, 0.001f);
        w[$"{prefix}.attn.to_q.weight"] = MakeFp8(hidden, hidden, seed + 4);
        w[$"{prefix}.attn.to_k.weight"] = MakeFp8(kvWidth, hidden, seed + 5);
        w[$"{prefix}.attn.to_v.weight"] = MakeFp8(kvWidth, hidden, seed + 6);
        w[$"{prefix}.attn.to_gate.weight"] = MakeFp8(hidden, hidden, seed + 7);
        w[$"{prefix}.attn.to_out.0.weight"] = MakeFp8(hidden, hidden, seed + 8);
        w[$"{prefix}.attn.norm_q.weight"] = MakeF32Vec(headDim, seed + 9, 0.001f);
        w[$"{prefix}.attn.norm_k.weight"] = MakeF32Vec(headDim, seed + 10, 0.001f);
        w[$"{prefix}.ff.gate.weight"] = MakeFp8(ffnInner, hidden, seed + 11);
        w[$"{prefix}.ff.up.weight"] = MakeFp8(ffnInner, hidden, seed + 12);
        w[$"{prefix}.ff.down.weight"] = MakeFp8(hidden, ffnInner, seed + 13);
    }

    // Block count deliberately small: 2 vs 4 blocks already showed FLAT per-call cost (160-170ms both
    // ways — this benchmark's finding is about the full block's attention/glue-op allocation DIVERSITY,
    // not block count), and 4 blocks alone reserved ~19.8GB — too close to this box's real VRAM ceiling to
    // safely coexist with xUnit's default cross-class test parallelism (no [Collection] serialization
    // exists for GPU tests in this suite; 4 blocks running concurrently with other GPU tests caused real
    // OOM failures elsewhere in the suite, confirmed as contention, not a regression, by re-running those
    // tests in isolation). 2 blocks (~9.8GB) stays a safer margin below the ceiling.
    [Theory]
    [InlineData(2, true)]
    [InlineData(2, false)]
    public unsafe void Measure_FullKrea2Block_RealConfig_RealJointSeq(int NumDistinctBlocks, bool enableCoopMat2)
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16 || !backend.Vk.HasCooperativeMatrix2)
        {
            _out.WriteLine("SKIPPED: no F16/coopmat2 support");
            return;
        }
        backend.EnableCoopMat2 = enableCoopMat2;

        // Krea2's real config (Krea2Config.cs defaults) and real joint sequence length for a 1024x1024
        // (64x64-patch) generation — the exact M=4109 that produced the OOM allocation this whole
        // investigation traced back to a step-graph-capture VRAM peak.
        const int Hidden = 6144, FfnInner = 16384, NumHeads = 48, NumKvHeads = 12, HeadDim = 128;
        const float Eps = 1e-5f;
        const int ImgSeqH = 64, ImgSeqW = 64, TxtSeq = 13;
        const int JointSeq = ImgSeqH * ImgSeqW + TxtSeq;   // 4109
        const int Batch = 1;
        const int NumSimulatedSteps = 2;

        int[] axesDimRope = [32, 48, 48];
        const int RopeTheta = 1000;

        Krea2Block[] blocks = new Krea2Block[NumDistinctBlocks];
        for (int b = 0; b < NumDistinctBlocks; b++)
        {
            Dictionary<string, Tensor> w = new();
            AddSyntheticBlockWeights(w, "block", Hidden, FfnInner, NumHeads, NumKvHeads, HeadDim, seed: b * 1000);
            blocks[b] = new Krea2Block(Hidden, FfnInner, NumHeads, NumKvHeads, Eps);
            blocks[b].LoadWeights(w, "block");
            backend.PreloadWeights(w.Values);
        }
        backend.CacheWeightCasts = false;   // matches Krea2's real config

        FluxRope rope = new(axesDimRope, RopeTheta);
        Tensor posIds = FluxRope.BuildPositionIds(TxtSeq, ImgSeqH, ImgSeqW);
        rope.Precompute(posIds);
        posIds.Dispose();

        Tensor tembMod = new(new TensorShape(Batch, 6 * Hidden), DType.F32);
        float* tp = (float*)tembMod.DataPointer;
        for (long i = 0; i < (long)Batch * 6 * Hidden; i++) tp[i] = (((i * 3) % 13) - 6) * 0.01f;

        Tensor MakeJoint()
        {
            Tensor t = new(new TensorShape(Batch, JointSeq, Hidden), DType.F16);
            Half* p = (Half*)t.DataPointer;
            for (long i = 0; i < (long)Batch * JointSeq * Hidden; i++) p[i] = (Half)(((i * 13) % 17 - 8) * 0.01f);
            return t;
        }

        void RunAllBlocksOnce()
        {
            Tensor joint = MakeJoint();
            for (int b = 0; b < NumDistinctBlocks; b++)
            {
                Tensor next = blocks[b].Forward(backend, joint, tembMod, rope, Batch, JointSeq);
                joint.Dispose();
                joint = next;
            }
            joint.Dispose();
        }

        for (int i = 0; i < NumSimulatedSteps; i++) RunAllBlocksOnce();
        backend.Sync();
        (long usedAfterWarmup, long reservedAfterWarmup, int blocksAfterWarmup, _) = backend.MemoryStats;

        (long coopMat2Before, long coopMatBefore, long tiledBefore) = backend.GemmEngagementCounts;
        Stopwatch sw = Stopwatch.StartNew();
        for (int i = 0; i < NumSimulatedSteps; i++) RunAllBlocksOnce();
        backend.Sync();
        sw.Stop();
        (long coopMat2After, long coopMatAfter, long tiledAfter) = backend.GemmEngagementCounts;

        long coopMat2Calls = coopMat2After - coopMat2Before;
        long coopMatCalls = coopMatAfter - coopMatBefore;
        long tiledCalls = tiledAfter - tiledBefore;
        long totalGemmCalls = coopMat2Calls + coopMatCalls + tiledCalls;
        double msPerGemmCall = totalGemmCalls > 0 ? sw.Elapsed.TotalMilliseconds / totalGemmCalls : -1;
        _out.WriteLine($"Full Krea2Block (Hidden={Hidden},FfnInner={FfnInner},Heads={NumHeads}/{NumKvHeads},JointSeq={JointSeq}), {NumDistinctBlocks} blocks x {NumSimulatedSteps} steps:");
        _out.WriteLine($"  {sw.Elapsed.TotalMilliseconds:F1} ms total, {totalGemmCalls} GEMM calls, {msPerGemmCall:F3} ms/GEMM-call");
        _out.WriteLine($"  GEMM path split: coopmat2={coopMat2Calls}, coopmat={coopMatCalls}, tiled={tiledCalls}");
        _out.WriteLine($"  Allocator: after warmup blocks={blocksAfterWarmup}, used={usedAfterWarmup / (1024.0 * 1024):F1} MB, reserved={reservedAfterWarmup / (1024.0 * 1024):F1} MB");
        _out.WriteLine($"  Compare: isolated SwiGlu-only chain (previous benchmark) was ~32.7-33.1 ms/Linear-call;");
        _out.WriteLine($"  real Krea2 e2e run measured ~82-135 ms/Linear-call.");

        tembMod.Dispose();
    }
}
