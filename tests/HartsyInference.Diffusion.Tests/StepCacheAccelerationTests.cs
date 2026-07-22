using Xunit;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Tests;

/// <summary>CPU correctness tests for the device-resident step-acceleration primitives: the
/// <see cref="IBackend.RelativeL1Distance"/> gate metric, <see cref="DeviceFeatureCache"/> semantics,
/// <see cref="GuidanceInterval"/> parsing/gating, and the QwenImageTransformer First-Block-Cache wiring
/// (tiny synthetic config). The load-bearing gates: a MISS forward and the recompute-after-drift forward are
/// bit-identical to the uncached baseline (same computation), and a HIT forward reproduces it to float
/// rounding — block0 + fl(final − block0) is exact to one rounding per element (IEEE 754 does not make
/// a+(b−a)==b bitwise) — proving the residual store/apply path AND the anchor lifetime bookkeeping through
/// the real Forward code.</summary>
public sealed unsafe class StepCacheAccelerationTests
{
    private static Tensor MakeF32(TensorShape shape, int seed, float scale = 0.05f)
    {
        Tensor t = new Tensor(shape, DType.F32);
        Random rng = new Random(seed);
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale;
        return t;
    }

    private static float[] Snapshot(Tensor t)
    {
        long n = t.ElementCount;
        float[] result = new float[n];
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < n; i++) result[i] = p[i];
        return result;
    }

    // ── IBackend.RelativeL1Distance (default host implementation) ─────────

    [Fact]
    public void RelativeL1Distance_MatchesHandComputedValue()
    {
        using CpuBackend backend = new CpuBackend();
        using Tensor a = new Tensor(new TensorShape(1, 4), DType.F32);
        using Tensor b = new Tensor(new TensorShape(1, 4), DType.F32);
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        ap[0] = 1f; ap[1] = -2f; ap[2] = 3f; ap[3] = 0f;
        bp[0] = 1f; bp[1] = -1f; bp[2] = 2f; bp[3] = 1f;

        // Σ|a−b| = 0 + 1 + 1 + 1 = 3; Σ|b| = 1 + 1 + 2 + 1 = 5.
        float rel = ((IBackend)backend).RelativeL1Distance(a, b);
        Assert.Equal(0.6f, rel, 5);
    }

    [Fact]
    public void RelativeL1Distance_ZeroReference_ReturnsZero()
    {
        using CpuBackend backend = new CpuBackend();
        using Tensor a = MakeF32(new TensorShape(1, 8), seed: 1);
        using Tensor b = new Tensor(new TensorShape(1, 8), DType.F32);
        float* bp = (float*)b.DataPointer;
        for (int i = 0; i < 8; i++) bp[i] = 0f;

        Assert.Equal(0f, ((IBackend)backend).RelativeL1Distance(a, b));
    }

    [Fact]
    public void RelativeL1Distance_F16_MatchesF32Computation()
    {
        using CpuBackend backend = new CpuBackend();
        using Tensor a16 = new Tensor(new TensorShape(1, 4), DType.F16);
        using Tensor b16 = new Tensor(new TensorShape(1, 4), DType.F16);
        Half* ap = (Half*)a16.DataPointer;
        Half* bp = (Half*)b16.DataPointer;
        ap[0] = (Half)1f; ap[1] = (Half)(-2f); ap[2] = (Half)3f; ap[3] = (Half)0f;
        bp[0] = (Half)1f; bp[1] = (Half)(-1f); bp[2] = (Half)2f; bp[3] = (Half)1f;

        float rel = ((IBackend)backend).RelativeL1Distance(a16, b16);
        Assert.Equal(0.6f, rel, 3);
    }

    [Fact]
    public void RelativeL1Distance_ShapeMismatch_Throws()
    {
        using CpuBackend backend = new CpuBackend();
        using Tensor a = MakeF32(new TensorShape(1, 4), seed: 1);
        using Tensor b = MakeF32(new TensorShape(1, 8), seed: 2);
        Assert.Throws<ArgumentException>(() => ((IBackend)backend).RelativeL1Distance(a, b));
    }

    [Fact]
    public void CpuBackend_ReportsDeviceStepCacheGateSupport()
    {
        using CpuBackend backend = new CpuBackend();
        Assert.True(((IBackend)backend).SupportsDeviceStepCacheGate);
    }

    // ── DeviceFeatureCache semantics (CPU backend, F32) ───────────────────

    [Fact]
    public void DeviceFeatureCache_FirstStep_AlwaysComputes()
    {
        using CpuBackend backend = new CpuBackend();
        using DeviceFeatureCache cache = new DeviceFeatureCache(threshold: 0.5f);
        using Tensor indicator = MakeF32(new TensorShape(1, 16), seed: 3);
        Assert.True(cache.ShouldCompute(backend, indicator));
        Assert.Equal(1, cache.Computes);
    }

    [Fact]
    public void DeviceFeatureCache_StableIndicator_ReusesAndReconstructsExactly()
    {
        using CpuBackend backend = new CpuBackend();
        using DeviceFeatureCache cache = new DeviceFeatureCache(threshold: 0.5f, maxConsecutiveReuse: 10);
        using Tensor input = MakeF32(new TensorShape(1, 16), seed: 4);
        using Tensor output = MakeF32(new TensorShape(1, 16), seed: 5);
        using Tensor indicator = MakeF32(new TensorShape(1, 16), seed: 6);

        Assert.True(cache.ShouldCompute(backend, indicator));
        cache.StoreResidual(backend, input, output);

        // Identical indicator → zero drift → reuse.
        Assert.False(cache.ShouldCompute(backend, indicator));
        Assert.Equal(1, cache.Reuses);

        // Reconstruction is input + fl(output − input): exact to one rounding per element (the residual
        // subtraction rounds once, the add rounds once — IEEE 754 does NOT guarantee a+(b−a)==b bitwise).
        using Tensor reconstructed = cache.ApplyResidual(backend, input);
        float[] expected = Snapshot(output);
        float[] actual = Snapshot(reconstructed);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-6f,
                $"reconstruction[{i}]={actual[i]} vs output {expected[i]}");
    }

    [Fact]
    public void DeviceFeatureCache_LargeDrift_ForcesRecompute()
    {
        using CpuBackend backend = new CpuBackend();
        using DeviceFeatureCache cache = new DeviceFeatureCache(threshold: 0.1f, maxConsecutiveReuse: 10);
        using Tensor input = MakeF32(new TensorShape(1, 16), seed: 7);
        using Tensor output = MakeF32(new TensorShape(1, 16), seed: 8);
        using Tensor indicator = MakeF32(new TensorShape(1, 16), seed: 9);

        Assert.True(cache.ShouldCompute(backend, indicator));
        cache.StoreResidual(backend, input, output);

        using Tensor doubled = new Tensor(indicator.Shape, DType.F32);
        backend.Scale(doubled, indicator, 2.0f);
        Assert.True(cache.ShouldCompute(backend, doubled));
        Assert.Equal(2, cache.Computes);
    }

    [Fact]
    public void DeviceFeatureCache_ConsecutiveReuseCap_IsBounded()
    {
        using CpuBackend backend = new CpuBackend();
        using DeviceFeatureCache cache = new DeviceFeatureCache(threshold: 100f, maxConsecutiveReuse: 2);
        using Tensor input = MakeF32(new TensorShape(1, 8), seed: 10);
        using Tensor output = MakeF32(new TensorShape(1, 8), seed: 11);
        using Tensor indicator = MakeF32(new TensorShape(1, 8), seed: 12);

        Assert.True(cache.ShouldCompute(backend, indicator));
        cache.StoreResidual(backend, input, output);

        Assert.False(cache.ShouldCompute(backend, indicator));
        Assert.False(cache.ShouldCompute(backend, indicator));
        Assert.True(cache.ShouldCompute(backend, indicator));
        Assert.Equal(2, cache.Reuses);
        Assert.Equal(2, cache.Computes);
    }

    [Fact]
    public void DeviceFeatureCache_ShapeChange_ForcesRecompute()
    {
        using CpuBackend backend = new CpuBackend();
        using DeviceFeatureCache cache = new DeviceFeatureCache(threshold: 100f, maxConsecutiveReuse: 10);
        using Tensor input = MakeF32(new TensorShape(1, 8), seed: 13);
        using Tensor output = MakeF32(new TensorShape(1, 8), seed: 14);
        using Tensor indicator = MakeF32(new TensorShape(1, 8), seed: 15);

        Assert.True(cache.ShouldCompute(backend, indicator));
        cache.StoreResidual(backend, input, output);

        // A different indicator shape (resolution change mid-session) must never reuse a stale residual.
        using Tensor widened = MakeF32(new TensorShape(1, 16), seed: 16);
        Assert.True(cache.ShouldCompute(backend, widened));
    }

    // ── GuidanceInterval ──────────────────────────────────────────────────

    [Theory]
    [InlineData("0.15,0.9", 0.15f, 0.9f)]
    [InlineData("0, 1", 0f, 1f)]
    [InlineData(" 0.2 , 0.8 ", 0.2f, 0.8f)]
    public void GuidanceInterval_Parse_Valid(string spec, float start, float end)
    {
        GuidanceInterval interval = GuidanceInterval.Parse(spec);
        Assert.Equal(start, interval.Start);
        Assert.Equal(end, interval.End);
    }

    [Theory]
    [InlineData("")]
    [InlineData("0.5")]
    [InlineData("0.9,0.1")]
    [InlineData("-0.1,0.5")]
    [InlineData("0.1,1.5")]
    [InlineData("abc,0.5")]
    public void GuidanceInterval_Parse_Malformed_Throws(string spec)
    {
        Assert.Throws<ArgumentException>(() => GuidanceInterval.Parse(spec));
    }

    [Fact]
    public void GuidanceInterval_Applies_GatesOnBand()
    {
        GuidanceInterval interval = new GuidanceInterval(0.2f, 0.8f);
        Assert.False(interval.Applies(0.1f));
        Assert.True(interval.Applies(0.2f));
        Assert.True(interval.Applies(0.5f));
        Assert.True(interval.Applies(0.8f));
        Assert.False(interval.Applies(0.9f));
        Assert.False(interval.IsAlways);
        Assert.True(GuidanceInterval.Always.IsAlways);
        Assert.True(GuidanceInterval.Always.Applies(0f));
        Assert.True(GuidanceInterval.Always.Applies(1f));
    }

    [Fact]
    public void GuidanceInterval_FromEnvironment_UnsetIsAlways()
    {
        const string variable = "HARTSY_CFG_INTERVAL_TEST_UNSET";
        Environment.SetEnvironmentVariable(variable, null);
        Assert.True(GuidanceInterval.FromEnvironment(variable).IsAlways);

        Environment.SetEnvironmentVariable(variable, "0.3,0.7");
        try
        {
            GuidanceInterval interval = GuidanceInterval.FromEnvironment(variable);
            Assert.Equal(0.3f, interval.Start);
            Assert.Equal(0.7f, interval.End);
        }
        finally
        {
            Environment.SetEnvironmentVariable(variable, null);
        }
    }

    // ── QwenImageTransformer First-Block-Cache wiring (tiny synthetic config) ──

    private const int Hidden = 256;
    private const int HeadDim = 128;   // QwenImageRope's default axes_dim [16,56,56] sums to 128 — fixed.
    private const int Depth = 2;
    private const int PatchSize = 2;
    private const int InChannels = 4;
    private const int ContextDim = 32;
    private const int MlpDim = 256;    // MlpRatio 1.0 at Hidden=256.

    private static Dictionary<string, Tensor> BuildTinyWeights()
    {
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>();
        int seed = 100;
        void Add(string key, params int[] dims)
        {
            weights[key] = MakeF32(new TensorShape(dims.Select(d => (long)d).ToArray()), seed++);
        }

        int patchDim = PatchSize * PatchSize * InChannels;
        Add("img_in.weight", Hidden, patchDim);
        Add("img_in.bias", Hidden);
        Add("txt_norm.weight", ContextDim);
        Add("txt_in.weight", Hidden, ContextDim);
        Add("txt_in.bias", Hidden);
        Add("time_text_embed.timestep_embedder.linear_1.weight", Hidden, 256);
        Add("time_text_embed.timestep_embedder.linear_1.bias", Hidden);
        Add("time_text_embed.timestep_embedder.linear_2.weight", Hidden, Hidden);
        Add("time_text_embed.timestep_embedder.linear_2.bias", Hidden);

        for (int i = 0; i < Depth; i++)
        {
            string p = $"transformer_blocks.{i}";
            // 6 modulation params (shift/scale/gate for msa + mlp) — AdaLNModulation(hiddenSize, 6).
            Add($"{p}.img_mod.1.weight", 6 * Hidden, Hidden);
            Add($"{p}.img_mod.1.bias", 6 * Hidden);
            Add($"{p}.txt_mod.1.weight", 6 * Hidden, Hidden);
            Add($"{p}.txt_mod.1.bias", 6 * Hidden);
            Add($"{p}.attn.to_q.weight", Hidden, Hidden);
            Add($"{p}.attn.to_q.bias", Hidden);
            Add($"{p}.attn.to_k.weight", Hidden, Hidden);
            Add($"{p}.attn.to_k.bias", Hidden);
            Add($"{p}.attn.to_v.weight", Hidden, Hidden);
            Add($"{p}.attn.to_v.bias", Hidden);
            Add($"{p}.attn.to_out.0.weight", Hidden, Hidden);
            Add($"{p}.attn.to_out.0.bias", Hidden);
            Add($"{p}.attn.add_q_proj.weight", Hidden, Hidden);
            Add($"{p}.attn.add_q_proj.bias", Hidden);
            Add($"{p}.attn.add_k_proj.weight", Hidden, Hidden);
            Add($"{p}.attn.add_k_proj.bias", Hidden);
            Add($"{p}.attn.add_v_proj.weight", Hidden, Hidden);
            Add($"{p}.attn.add_v_proj.bias", Hidden);
            Add($"{p}.attn.to_add_out.weight", Hidden, Hidden);
            Add($"{p}.attn.to_add_out.bias", Hidden);
            Add($"{p}.attn.norm_q.weight", HeadDim);
            Add($"{p}.attn.norm_k.weight", HeadDim);
            Add($"{p}.attn.norm_added_q.weight", HeadDim);
            Add($"{p}.attn.norm_added_k.weight", HeadDim);
            Add($"{p}.img_mlp.net.0.proj.weight", MlpDim, Hidden);
            Add($"{p}.img_mlp.net.0.proj.bias", MlpDim);
            Add($"{p}.img_mlp.net.2.weight", Hidden, MlpDim);
            Add($"{p}.img_mlp.net.2.bias", Hidden);
            Add($"{p}.txt_mlp.net.0.proj.weight", MlpDim, Hidden);
            Add($"{p}.txt_mlp.net.0.proj.bias", MlpDim);
            Add($"{p}.txt_mlp.net.2.weight", Hidden, MlpDim);
            Add($"{p}.txt_mlp.net.2.bias", Hidden);
        }

        Add("norm_out.linear.weight", 2 * Hidden, Hidden);
        Add("norm_out.linear.bias", 2 * Hidden);
        Add("proj_out.weight", patchDim, Hidden);
        Add("proj_out.bias", patchDim);
        return weights;
    }

    private static QwenImageTransformer BuildTinyTransformer(Dictionary<string, Tensor> weights)
    {
        QwenImageTransformer transformer = new QwenImageTransformer(new QwenImageConfig
        {
            HiddenSize = Hidden,
            NumHeads = Hidden / HeadDim,
            HeadDim = HeadDim,
            Depth = Depth,
            PatchSize = PatchSize,
            InChannels = InChannels,
            ContextDim = ContextDim,
            MlpRatio = (float)MlpDim / Hidden,
        });
        transformer.LoadWeights(weights);
        return transformer;
    }

    [Fact]
    public void QwenTransformer_CacheHit_MatchesFullComputeToRounding()
    {
        using CpuBackend backend = new CpuBackend();
        Dictionary<string, Tensor> weights = BuildTinyWeights();
        using QwenImageTransformer transformer = BuildTinyTransformer(weights);

        const int hPacked = 2, wPacked = 2;
        int patchDim = PatchSize * PatchSize * InChannels;
        using Tensor latent = MakeF32(new TensorShape(1, hPacked * wPacked, patchDim), seed: 500, scale: 0.5f);
        using Tensor context = MakeF32(new TensorShape(1, 3, ContextDim), seed: 501, scale: 0.5f);

        // Uncached baseline at a fixed timestep.
        using Tensor baseline = transformer.Forward(backend, latent, context, timestep: 0.5f, hPacked, wPacked);

        // Cached: first call misses (full compute + residual store); second call with IDENTICAL inputs
        // produces an identical block-0 output → zero gate drift → HIT. The miss output is bit-identical to
        // the uncached baseline (same computation); the hit output is block0 + fl(final − block0), exact to
        // one float rounding per element, then amplified only by the final norm/projection.
        using DeviceFeatureCache cache = new DeviceFeatureCache(threshold: 0.5f, maxConsecutiveReuse: 4);
        using Tensor missOut = transformer.Forward(backend, latent, context, 0.5f, hPacked, wPacked, stepCache: cache);
        using Tensor hitOut = transformer.Forward(backend, latent, context, 0.5f, hPacked, wPacked, stepCache: cache);

        Assert.Equal(1, cache.Computes);
        Assert.Equal(1, cache.Reuses);
        Assert.Equal(Snapshot(baseline), Snapshot(missOut));
        float[] expected = Snapshot(baseline);
        float[] actual = Snapshot(hitOut);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-4f,
                $"hit[{i}]={actual[i]} vs baseline {expected[i]}");

        foreach (Tensor w in weights.Values) w.Dispose();
    }

    [Fact]
    public void QwenTransformer_NullCache_MatchesOriginalPath()
    {
        using CpuBackend backend = new CpuBackend();
        Dictionary<string, Tensor> weights = BuildTinyWeights();
        using QwenImageTransformer transformer = BuildTinyTransformer(weights);

        const int hPacked = 2, wPacked = 2;
        int patchDim = PatchSize * PatchSize * InChannels;
        using Tensor latent = MakeF32(new TensorShape(1, hPacked * wPacked, patchDim), seed: 600, scale: 0.5f);
        using Tensor context = MakeF32(new TensorShape(1, 3, ContextDim), seed: 601, scale: 0.5f);

        using Tensor a = transformer.Forward(backend, latent, context, 0.7f, hPacked, wPacked);
        using Tensor b = transformer.Forward(backend, latent, context, 0.7f, hPacked, wPacked, stepCache: null);
        Assert.Equal(Snapshot(a), Snapshot(b));

        foreach (Tensor w in weights.Values) w.Dispose();
    }

    [Fact]
    public void QwenTransformer_TimestepJump_ForcesRecompute()
    {
        using CpuBackend backend = new CpuBackend();
        Dictionary<string, Tensor> weights = BuildTinyWeights();
        using QwenImageTransformer transformer = BuildTinyTransformer(weights);

        const int hPacked = 2, wPacked = 2;
        int patchDim = PatchSize * PatchSize * InChannels;
        using Tensor latent = MakeF32(new TensorShape(1, hPacked * wPacked, patchDim), seed: 700, scale: 0.5f);
        using Tensor context = MakeF32(new TensorShape(1, 3, ContextDim), seed: 701, scale: 0.5f);

        // A tight threshold with a large timestep jump must recompute (the modulation shift changes
        // block-0's output), and the recompute must again match the uncached forward bit-exactly.
        using DeviceFeatureCache cache = new DeviceFeatureCache(threshold: 1e-6f, maxConsecutiveReuse: 8);
        using Tensor first = transformer.Forward(backend, latent, context, 1.0f, hPacked, wPacked, stepCache: cache);
        using Tensor second = transformer.Forward(backend, latent, context, 0.1f, hPacked, wPacked, stepCache: cache);
        using Tensor reference = transformer.Forward(backend, latent, context, 0.1f, hPacked, wPacked);

        Assert.Equal(2, cache.Computes);
        Assert.Equal(0, cache.Reuses);
        Assert.Equal(Snapshot(reference), Snapshot(second));

        foreach (Tensor w in weights.Values) w.Dispose();
    }
}
