using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Phase 8 (ROADMAP.md §1, DiT sharding): verifies <see cref="Krea2Transformer.ForwardSharded"/> — a
/// block-range split of the 28-block loop across two <see cref="CudaBackend"/>s — matches
/// <see cref="Krea2Transformer.Forward"/> (unsharded, one backend, whole DiT resident) bit-for-bit on the same
/// synthetic tiny config <see cref="Krea2TransformerTests"/> uses for its CPU structural smoke test. This is a
/// sequential pipeline split, not tensor-parallel — no latency win, VRAM pooling only — so with F16 mode and
/// step-graph both off (the default; sharding excludes both anyway), every op is deterministic F32 and the two
/// paths should produce IDENTICAL floats, not just close ones.</summary>
[Trait("Category", "Integration")]
[Collection("CudaSerial")]
public sealed class Krea2DitShardingTests
{
    private readonly ITestOutputHelper _output;
    public Krea2DitShardingTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    // 4 blocks (vs the CPU smoke test's 2) so the split point (2) has real work on both sides.
    private static Krea2Config TinyConfig => new()
    {
        VaeChannels = 4,
        PatchSize = 2,
        NumLayers = 4,
        AttentionHeadDim = 8,
        NumAttentionHeads = 2,
        NumKvHeads = 1,
        IntermediateSize = 32,
        TimestepEmbedDim = 16,
        TextHiddenDim = 12,
        NumTextLayers = 3,
        TextNumHeads = 2,
        TextNumKvHeads = 2,
        TextIntermediateSize = 24,
        NumLayerwiseTextBlocks = 1,
        NumRefinerTextBlocks = 1,
        AxesDimRope = [2, 4, 2],
        RopeTheta = 1000,
        NormEps = 1e-5f,
    };

    [Fact]
    public void ForwardSharded_MatchesUnsharded_BitParity()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        int deviceCount = CudaContext.GetDeviceCount();
        int secondOrdinal = deviceCount >= 2 ? 1 : 0;
        _output.WriteLine($"Devices: {deviceCount}; sharded backend B on ordinal {secondOrdinal}.");

        Krea2Config cfg = TinyConfig;
        const int splitBlock = 2;

        // ── Reference: one backend, whole DiT resident, unsharded Forward. ──
        Dictionary<string, Tensor> wRef = Krea2WeightBuilder.Build(cfg);
        using CudaBackend refBackend = new(deviceOrdinal: 0, PtxDir());
        using Krea2Transformer refTransformer = new(cfg);
        refTransformer.LoadWeights(wRef);
        refBackend.PreloadWeights(refTransformer.EnumerateWeights());

        int h = 8, wd = 8;
        using Tensor latent = Krea2WeightBuilder.Rand(new TensorShape(1, cfg.VaeChannels, h, wd), 100, 0.5f);
        using Tensor encoderHidden = Krea2WeightBuilder.Rand(new TensorShape(1, 5, cfg.NumTextLayers * cfg.TextHiddenDim), 200, 0.2f);

        using Tensor velocityRef = refTransformer.Forward(refBackend, latent, 0.7f, encoderHidden);
        float[] refValues = ToArray(velocityRef);
        refBackend.FreeWeights(refTransformer.EnumerateWeights());
        Krea2WeightBuilder.DisposeAll(wRef);

        // ── Sharded: SAME weight VALUES (independent load, same seeds), split across two backends. Backend A
        // gets shared weights + blocks[0,split); backend B gets ONLY blocks[split,BlockCount) — the asymmetric
        // preload that makes this VRAM-pooling rather than replication. ──
        Dictionary<string, Tensor> wSharded = Krea2WeightBuilder.Build(cfg);
        using CudaBackend backendA = new(deviceOrdinal: 0, PtxDir());
        using CudaBackend backendB = new(deviceOrdinal: secondOrdinal, PtxDir());
        using Krea2Transformer shardedTransformer = new(cfg);
        shardedTransformer.LoadWeights(wSharded);

        List<Tensor> aWeights = new(shardedTransformer.EnumerateSharedWeights());
        aWeights.AddRange(shardedTransformer.EnumerateBlockRangeWeights(0, splitBlock));
        backendA.PreloadWeights(aWeights);
        List<Tensor> bWeights = new(shardedTransformer.EnumerateBlockRangeWeights(splitBlock, cfg.NumLayers));
        backendB.PreloadWeights(bWeights);

        using Tensor velocitySharded = shardedTransformer.ForwardSharded(
            backendA, backendB, latent, 0.7f, encoderHidden, splitBlock);
        float[] shardedValues = ToArray(velocitySharded);

        backendA.FreeWeights(aWeights);
        backendB.FreeWeights(bWeights);
        Krea2WeightBuilder.DisposeAll(wSharded);

        Assert.Equal(refValues.Length, shardedValues.Length);
        int mismatches = 0;
        for (int i = 0; i < refValues.Length; i++)
        {
            if (refValues[i] != shardedValues[i])
            {
                if (mismatches < 5)
                    _output.WriteLine($"mismatch at {i}: ref={refValues[i]} sharded={shardedValues[i]}");
                mismatches++;
            }
        }
        Assert.True(mismatches == 0, $"{mismatches}/{refValues.Length} values differ between unsharded and sharded forward.");
    }

    private static unsafe float[] ToArray(Tensor t)
    {
        float[] arr = new float[t.ElementCount];
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) arr[i] = p[i];
        return arr;
    }
}

/// <summary>Weight-building helpers shared with <see cref="Krea2TransformerTests"/> (kept file-local there since
/// that test predates this one) — duplicated here as a small internal static class rather than exposing
/// internals across test files, since the two tests build weights identically but for different backends.</summary>
internal static class Krea2WeightBuilder
{
    public static Dictionary<string, Tensor> Build(Krea2Config c)
    {
        Dictionary<string, Tensor> w = new();
        int H = c.HiddenSize, td = c.TextHiddenDim, inCh = c.InChannels;
        int qd = c.NumAttentionHeads * c.AttentionHeadDim, kvd = c.NumKvHeads * c.AttentionHeadDim;
        int tqd = c.TextNumHeads * (td / c.TextNumHeads), tkvd = c.TextNumKvHeads * (td / c.TextNumHeads);
        int seed = 1;
        void Lin(string k, int o, int i) => w[k] = Rand(new TensorShape(o, i), seed++, 0.06f);
        void Vec(string k, int d, float center) => w[k] = Const(new TensorShape(d), center, seed++);

        Lin("img_in.weight", H, inCh); Vec("img_in.bias", H, 0f);
        Lin("time_embed.linear_1.weight", H, c.TimestepEmbedDim); Vec("time_embed.linear_1.bias", H, 0f);
        Lin("time_embed.linear_2.weight", H, H); Vec("time_embed.linear_2.bias", H, 0f);
        Lin("time_mod_proj.weight", 6 * H, H); Vec("time_mod_proj.bias", 6 * H, 0f);
        Vec("txt_in.norm.weight", td, 0f);
        Lin("txt_in.linear_1.weight", H, td); Vec("txt_in.linear_1.bias", H, 0f);
        Lin("txt_in.linear_2.weight", H, H); Vec("txt_in.linear_2.bias", H, 0f);
        w["final_layer.scale_shift_table"] = Rand(new TensorShape(2, H), seed++, 0.02f);
        Vec("final_layer.norm.weight", H, 0f);
        Lin("final_layer.linear.weight", inCh, H); Vec("final_layer.linear.bias", inCh, 0f);

        void FusionBlock(string p) => Block(p, td, td / c.TextNumHeads, tqd, tkvd, c.TextIntermediateSize);
        for (int i = 0; i < c.NumLayerwiseTextBlocks; i++) FusionBlock($"text_fusion.layerwise_blocks.{i}");
        for (int i = 0; i < c.NumRefinerTextBlocks; i++) FusionBlock($"text_fusion.refiner_blocks.{i}");
        Lin("text_fusion.projector.weight", 1, c.NumTextLayers);

        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"transformer_blocks.{i}";
            w[$"{p}.scale_shift_table"] = Rand(new TensorShape(6, H), seed++, 0.02f);
            Block(p, H, c.AttentionHeadDim, qd, kvd, c.IntermediateSize);
        }
        return w;

        void Block(string p, int dim, int headDim, int qDim, int kvDim, int inner)
        {
            Vec($"{p}.norm1.weight", dim, 0f);
            Vec($"{p}.norm2.weight", dim, 0f);
            Lin($"{p}.attn.to_q.weight", qDim, dim);
            Lin($"{p}.attn.to_k.weight", kvDim, dim);
            Lin($"{p}.attn.to_v.weight", kvDim, dim);
            Lin($"{p}.attn.to_gate.weight", dim, dim);
            Lin($"{p}.attn.to_out.0.weight", dim, dim);
            Vec($"{p}.attn.norm_q.weight", headDim, 0f);
            Vec($"{p}.attn.norm_k.weight", headDim, 0f);
            Lin($"{p}.ff.gate.weight", inner, dim);
            Lin($"{p}.ff.up.weight", inner, dim);
            Lin($"{p}.ff.down.weight", dim, inner);
        }
    }

    public static unsafe Tensor Rand(TensorShape s, int seed, float scale)
    {
        Tensor t = new(s, DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        long n = s.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    public static unsafe Tensor Const(TensorShape s, float center, int seed)
    {
        Tensor t = new(s, DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        long n = s.ElementCount;
        for (long i = 0; i < n; i++) p[i] = center + (float)((rng.NextDouble() * 2 - 1) * 0.02);
        return t;
    }

    public static void DisposeAll(Dictionary<string, Tensor> w)
    {
        foreach (Tensor t in w.Values) t.Dispose();
    }
}
