using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>DiT sharding (ROADMAP.md §1): verifies <see cref="Lumina2Transformer.ForwardSharded"/> — a
/// block-range split of the main <c>Lumina2Block</c> loop across two <see cref="CudaBackend"/>s — matches
/// <see cref="Lumina2Transformer.Forward"/> (unsharded, one backend, whole NextDiT resident) bit-for-bit on a
/// synthetic tiny config. Same-GPU split by design (see <c>Sd3DitShardingTests</c>' class doc for why
/// cross-architecture pairs are not expected to reproduce bit-exact): with every op deterministic F32, the two
/// paths should produce IDENTICAL floats. The context/noise-refiner prefix always runs on backend A — this test
/// also exercises that both refiner stacks (and the shared main-layer RoPE table, cross-backend via
/// <c>ZImageRope</c>'s auto-promoted lazy upload — see <see cref="Lumina2Transformer.ForwardSharded"/> doc)
/// hand off correctly to backend B's block range.</summary>
[Trait("Category", "Integration")]
[Collection("CudaSerial")]
public sealed class Lumina2DitShardingTests
{
    private readonly ITestOutputHelper _output;
    public Lumina2DitShardingTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    // 4 main layers (vs. the real 26) so the split point (2) has real work on both sides. 1 refiner layer each
    // (vs. the real 2) keeps the tiny config small; GQA (NumHeads=4, NumKvHeads=2) exercises the same repeat-KV
    // path the real 24:8 config uses.
    private static Lumina2Config TinyConfig => new()
    {
        HiddenSize = 32,
        NumHeads = 4,
        NumKvHeads = 2,
        HeadDim = 8,
        NumLayers = 4,
        NumRefinerLayers = 1,
        FfnDim = 24,
        InChannels = 4,
        PatchSize = 2,
        CapFeatDim = 12,
        AdaLNEmbedDim = 16,
        AxesDims = [2, 3, 3],
        RopeTheta = 10000f,
    };

    [Fact]
    public void ForwardSharded_MatchesUnsharded_BitParity()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        // Same-GPU by design — see class doc.
        const int secondOrdinal = 0;
        _output.WriteLine($"Devices: {CudaContext.GetDeviceCount()}; sharded backend B on ordinal {secondOrdinal} (same-GPU split).");

        Lumina2Config cfg = TinyConfig;
        const int splitBlock = 2;

        // ── Reference: one backend, whole NextDiT resident, unsharded Forward. ──
        Dictionary<string, Tensor> wRef = Lumina2WeightBuilder.Build(cfg);
        using CudaBackend refBackend = new(deviceOrdinal: 0, PtxDir());
        using Lumina2Transformer refTransformer = new(cfg);
        refTransformer.LoadWeights(wRef);
        refBackend.PreloadWeights(refTransformer.EnumerateWeights());

        int h = 8, wd = 8;
        using Tensor latent = Lumina2WeightBuilder.Rand(new TensorShape(1, cfg.InChannels, h, wd), 100, 0.5f);
        const int capLen = 5;
        using Tensor captionEmbeddings = Lumina2WeightBuilder.Rand(new TensorShape(1, capLen, cfg.CapFeatDim), 200, 0.2f);

        using Tensor velocityRef = refTransformer.Forward(refBackend, latent, captionEmbeddings, 0.3f);
        float[] refValues = ToArray(velocityRef);
        refBackend.FreeWeights(refTransformer.EnumerateWeights());
        Lumina2WeightBuilder.DisposeAll(wRef);

        // ── Sharded: SAME weight VALUES (independent load, same seeds), split across two backends. Backend A
        // gets shared weights (embedders + both refiner stacks) + main layers[0,split); backend B gets ONLY
        // main layers[split,NumLayers) — the asymmetric preload that makes this VRAM-pooling rather than
        // replication. ──
        Dictionary<string, Tensor> wSharded = Lumina2WeightBuilder.Build(cfg);
        using CudaBackend backendA = new(deviceOrdinal: 0, PtxDir());
        using CudaBackend backendB = new(deviceOrdinal: secondOrdinal, PtxDir());
        using Lumina2Transformer shardedTransformer = new(cfg);
        shardedTransformer.LoadWeights(wSharded);

        List<Tensor> aWeights = new(shardedTransformer.EnumerateSharedWeights());
        aWeights.AddRange(shardedTransformer.EnumerateBlockRangeWeights(0, splitBlock));
        backendA.PreloadWeights(aWeights);
        List<Tensor> bWeights = new(shardedTransformer.EnumerateBlockRangeWeights(splitBlock, cfg.NumLayers));
        backendB.PreloadWeights(bWeights);

        using Tensor velocitySharded = shardedTransformer.ForwardSharded(
            backendA, backendB, latent, captionEmbeddings, 0.3f, splitBlock);
        float[] shardedValues = ToArray(velocitySharded);

        backendA.FreeWeights(aWeights);
        backendB.FreeWeights(bWeights);
        Lumina2WeightBuilder.DisposeAll(wSharded);

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

/// <summary>Tiny synthetic Lumina-Image-2.0 weight builder shared by the sharding tests — mirrors
/// <c>Krea2WeightBuilder</c>'s shape.</summary>
internal static class Lumina2WeightBuilder
{
    public static Dictionary<string, Tensor> Build(Lumina2Config c)
    {
        Dictionary<string, Tensor> w = new();
        int hidden = c.HiddenSize;
        int seed = 1;
        void Lin(string k, int o, int i) => w[k] = Rand(new TensorShape(o, i), seed++, 0.06f);
        void Vec(string k, int d) => w[k] = Rand(new TensorShape(d), seed++, 0.02f);

        Lin("time_caption_embed.timestep_embedder.linear_1.weight", c.AdaLNEmbedDim, 256);
        Vec("time_caption_embed.timestep_embedder.linear_1.bias", c.AdaLNEmbedDim);
        Lin("time_caption_embed.timestep_embedder.linear_2.weight", c.AdaLNEmbedDim, c.AdaLNEmbedDim);
        Vec("time_caption_embed.timestep_embedder.linear_2.bias", c.AdaLNEmbedDim);

        Vec("time_caption_embed.caption_embedder.0.weight", c.CapFeatDim);
        Lin("time_caption_embed.caption_embedder.1.weight", hidden, c.CapFeatDim);
        Vec("time_caption_embed.caption_embedder.1.bias", hidden);

        Lin("x_embedder.weight", hidden, c.InChannels * c.PatchSize * c.PatchSize * c.FramePatchSize);
        Vec("x_embedder.bias", hidden);

        Lin("norm_out.linear_1.weight", hidden, c.AdaLNEmbedDim);
        Vec("norm_out.linear_1.bias", hidden);
        int patchOutDim = c.InChannels * c.PatchSize * c.PatchSize * c.FramePatchSize;
        Lin("norm_out.linear_2.weight", patchOutDim, hidden);
        Vec("norm_out.linear_2.bias", patchOutDim);

        int qDim = c.NumHeads * c.HeadDim, kvDim = c.NumKvHeads * c.HeadDim;
        void RefinerBlock(string p)
        {
            Vec($"{p}.norm1.weight", hidden);
            Vec($"{p}.norm2.weight", hidden);
            Lin($"{p}.attn.to_q.weight", qDim, hidden);
            Lin($"{p}.attn.to_k.weight", kvDim, hidden);
            Lin($"{p}.attn.to_v.weight", kvDim, hidden);
            Lin($"{p}.attn.to_out.0.weight", hidden, qDim);
            Vec($"{p}.attn.norm_q.weight", c.HeadDim);
            Vec($"{p}.attn.norm_k.weight", c.HeadDim);
            Vec($"{p}.ffn_norm1.weight", hidden);
            Vec($"{p}.ffn_norm2.weight", hidden);
            Lin($"{p}.feed_forward.linear_1.weight", c.FfnDim, hidden);
            Lin($"{p}.feed_forward.linear_2.weight", hidden, c.FfnDim);
            Lin($"{p}.feed_forward.linear_3.weight", c.FfnDim, hidden);
        }
        void MainBlock(string p)
        {
            Vec($"{p}.norm1.norm.weight", hidden);
            Lin($"{p}.norm1.linear.weight", 4 * hidden, c.AdaLNEmbedDim);
            Vec($"{p}.norm1.linear.bias", 4 * hidden);
            Lin($"{p}.attn.to_q.weight", qDim, hidden);
            Lin($"{p}.attn.to_k.weight", kvDim, hidden);
            Lin($"{p}.attn.to_v.weight", kvDim, hidden);
            Lin($"{p}.attn.to_out.0.weight", hidden, qDim);
            Vec($"{p}.attn.norm_q.weight", c.HeadDim);
            Vec($"{p}.attn.norm_k.weight", c.HeadDim);
            Vec($"{p}.norm2.weight", hidden);
            Vec($"{p}.ffn_norm1.weight", hidden);
            Vec($"{p}.ffn_norm2.weight", hidden);
            Lin($"{p}.feed_forward.linear_1.weight", c.FfnDim, hidden);
            Lin($"{p}.feed_forward.linear_2.weight", hidden, c.FfnDim);
            Lin($"{p}.feed_forward.linear_3.weight", c.FfnDim, hidden);
        }

        // context_refiner is Lumina2ContextRefinerBlock (no AdaLN, plain norm1.weight); noise_refiner is a
        // Lumina2Block just like the main layers (WITH AdaLN, norm1.norm.weight + norm1.linear.*) — see
        // Lumina2Transformer's constructor.
        for (int i = 0; i < c.NumRefinerLayers; i++) RefinerBlock($"context_refiner.{i}");
        for (int i = 0; i < c.NumRefinerLayers; i++) MainBlock($"noise_refiner.{i}");
        for (int i = 0; i < c.NumLayers; i++) MainBlock($"layers.{i}");

        return w;
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

    public static void DisposeAll(Dictionary<string, Tensor> w)
    {
        foreach (Tensor t in w.Values) t.Dispose();
    }
}
