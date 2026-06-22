using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.TextEncoders;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Structural smoke tests for the Qwen3-VL vision tower and the multimodal instruction encoder used by
/// Boogu-Image editing. Tiny synthetic configs + random weights on the CPU backend: they prove the image processor,
/// the ViT (patch embed, bilinear pos-embed, 2D RoPE, blocks, merger, deepstack mergers), and the M-RoPE + deepstack
/// LM path run end to end and produce finite outputs of the right shape. Numeric parity vs HF is a separate harness
/// (docs/Research/BOOGU_IMAGE.md §10).</summary>
public sealed class Qwen3VlVisionTowerTests
{
    private readonly ITestOutputHelper _output;
    public Qwen3VlVisionTowerTests(ITestOutputHelper output) => _output = output;

    private static Qwen3VlVisionConfig TinyVision => new()
    {
        Depth = 2,
        HiddenSize = 16,
        NumHeads = 2,
        IntermediateSize = 32,
        InChannels = 3,
        PatchSize = 16,
        SpatialMergeSize = 2,
        TemporalPatchSize = 2,
        OutHiddenSize = 24,
        NumPositionEmbeddings = 16, // side = 4
        DeepstackVisualIndexes = [1],
        NormEps = 1e-6f,
    };

    private static LlamaStyleEncoderConfig TinyLm => new()
    {
        HiddenSize = 24,
        NumLayers = 3,
        NumQueryHeads = 2,
        NumKvHeads = 1,
        HeadDim = 12,
        IntermediateSize = 32,
        VocabSize = 200,
        RmsNormEps = 1e-6f,
        RopeTheta = 5_000_000f,
        MaxPositionEmbeddings = 512,
        QkHeadNorm = true,
        AttentionBias = false,
        HasFinalNorm = true,
    };

    [Fact]
    public void ImageProcessor_ProducesPatchesAndGrid()
    {
        Qwen3VlVisionConfig cfg = TinyVision;
        Qwen3VlImageProcessor proc = new(cfg, maxPixels: 64 * 64);
        using Tensor rgb = Random3(3, 64, 64, 7);
        (Tensor pix, int t, int h, int w) = proc.Preprocess(rgb);
        Assert.Equal(1, t);
        Assert.Equal(4, h); // 64 / 16
        Assert.Equal(4, w);
        Assert.Equal(16, (int)pix.Shape[0]);              // h*w patches
        Assert.Equal(cfg.PatchEmbedInDim, (int)pix.Shape[1]); // 3*2*16*16 = 1536
        AssertFinite(pix);
        pix.Dispose();
    }

    [Fact]
    public void VisionEncoder_ProducesMergedTokensAndDeepstack()
    {
        Qwen3VlVisionConfig cfg = TinyVision;
        using CpuBackend backend = new();
        using Qwen3VlVisionEncoder vision = new(cfg);
        Dictionary<string, Tensor> w = BuildVisionWeights(cfg);
        vision.LoadWeights(w);

        Qwen3VlImageProcessor proc = new(cfg, maxPixels: 64 * 64);
        using Tensor rgb = Random3(3, 64, 64, 11);
        (Tensor pix, int t, int h, int ww) = proc.Preprocess(rgb);

        Qwen3VlVisionEncoder.VisionOutput vo = vision.Forward(backend, pix, t, h, ww);
        pix.Dispose();

        Assert.Equal(4, vo.NumMergedTokens);                       // 16 patches / 4
        Assert.Equal(4, (int)vo.MergedTokens.Shape[0]);
        Assert.Equal(cfg.OutHiddenSize, (int)vo.MergedTokens.Shape[1]);
        Assert.Single(vo.DeepstackFeatures);
        AssertFinite(vo.MergedTokens);
        AssertFinite(vo.DeepstackFeatures[0]);

        vo.MergedTokens.Dispose();
        foreach (Tensor d in vo.DeepstackFeatures) d.Dispose();
        DisposeAll(w);
        _output.WriteLine("Vision encoder produced finite merged tokens + deepstack.");
    }

    [Fact]
    public void MultimodalEncoder_TextPlusImage_ProducesFiniteHidden()
    {
        Qwen3VlVisionConfig vcfg = TinyVision;
        LlamaStyleEncoderConfig lcfg = TinyLm;
        using CpuBackend backend = new();

        using Qwen3VlVisionEncoder vision = new(vcfg);
        Dictionary<string, Tensor> vw = BuildVisionWeights(vcfg);
        vision.LoadWeights(vw);

        using LlamaStyleEncoder lm = new(lcfg);
        Dictionary<string, Tensor> lw = BuildLmWeights(lcfg);
        lm.LoadWeights(lw);

        Qwen3VlImageProcessor proc = new(vcfg, maxPixels: 64 * 64);
        Qwen3VlMultimodalEncoder mm = new(lm, vision, proc, vcfg, imageTokenId: 100,
            textHeadDim: lcfg.HeadDim, ropeTheta: lcfg.RopeTheta, mropeSection: [2, 2, 2]);

        using Tensor rgb = Random3(3, 64, 64, 23);
        // [text, text, <image>×4, text] — 4 placeholders == numMergedTokens.
        int[] tokens = [1, 2, 100, 100, 100, 100, 3];

        using Tensor hidden = mm.Encode(backend, tokens, [rgb]);
        Assert.Equal(3, hidden.Shape.Rank);
        Assert.Equal(tokens.Length, (int)hidden.Shape[1]);
        Assert.Equal(lcfg.HiddenSize, (int)hidden.Shape[2]);
        AssertFinite(hidden);

        DisposeAll(vw);
        DisposeAll(lw);
        _output.WriteLine($"Multimodal encoder hidden {hidden.Shape} finite.");
    }

    // ── synthetic weights ──

    internal static Dictionary<string, Tensor> BuildVisionWeights(Qwen3VlVisionConfig c)
    {
        Dictionary<string, Tensor> w = new();
        int H = c.HiddenSize, inter = c.IntermediateSize, merged = H * c.SpatialMergeSize * c.SpatialMergeSize;
        int seed = 1;
        void Lin(string k, int o, int i) => w[k] = Rand(new TensorShape(o, i), seed++, 0.08f);
        void Vec(string k, int d, float center) => w[k] = Const(new TensorShape(d), center, seed++);

        w["patch_embed.proj.weight"] = Rand(new TensorShape(H, c.PatchEmbedInDim), seed++, 0.02f);
        Vec("patch_embed.proj.bias", H, 0f);
        w["pos_embed.weight"] = Rand(new TensorShape(c.NumPositionEmbeddings, H), seed++, 0.02f);

        for (int i = 0; i < c.Depth; i++)
        {
            string p = $"blocks.{i}";
            Vec($"{p}.norm1.weight", H, 1f); Vec($"{p}.norm1.bias", H, 0f);
            Vec($"{p}.norm2.weight", H, 1f); Vec($"{p}.norm2.bias", H, 0f);
            Lin($"{p}.attn.qkv.weight", 3 * H, H); Vec($"{p}.attn.qkv.bias", 3 * H, 0f);
            Lin($"{p}.attn.proj.weight", H, H); Vec($"{p}.attn.proj.bias", H, 0f);
            Lin($"{p}.mlp.linear_fc1.weight", inter, H); Vec($"{p}.mlp.linear_fc1.bias", inter, 0f);
            Lin($"{p}.mlp.linear_fc2.weight", H, inter); Vec($"{p}.mlp.linear_fc2.bias", H, 0f);
        }

        Vec("merger.norm.weight", H, 1f); Vec("merger.norm.bias", H, 0f);
        Lin("merger.linear_fc1.weight", merged, merged); Vec("merger.linear_fc1.bias", merged, 0f);
        Lin("merger.linear_fc2.weight", c.OutHiddenSize, merged); Vec("merger.linear_fc2.bias", c.OutHiddenSize, 0f);

        for (int i = 0; i < c.DeepstackVisualIndexes.Length; i++)
        {
            string p = $"deepstack_merger_list.{i}";
            Vec($"{p}.norm.weight", merged, 1f); Vec($"{p}.norm.bias", merged, 0f);
            Lin($"{p}.linear_fc1.weight", merged, merged); Vec($"{p}.linear_fc1.bias", merged, 0f);
            Lin($"{p}.linear_fc2.weight", c.OutHiddenSize, merged); Vec($"{p}.linear_fc2.bias", c.OutHiddenSize, 0f);
        }
        return w;
    }

    internal static Dictionary<string, Tensor> BuildLmWeights(LlamaStyleEncoderConfig c)
    {
        Dictionary<string, Tensor> w = new();
        int H = c.HiddenSize, qd = c.NumQueryHeads * c.HeadDim, kvd = c.NumKvHeads * c.HeadDim, inter = c.IntermediateSize;
        int seed = 100;
        void Lin(string k, int o, int i) => w[k] = Rand(new TensorShape(o, i), seed++, 0.08f);
        void Vec(string k, int d, float center) => w[k] = Const(new TensorShape(d), center, seed++);

        w["model.embed_tokens.weight"] = Rand(new TensorShape(c.VocabSize, H), seed++, 0.05f);
        Vec("model.norm.weight", H, 1f);
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            Vec($"{p}.input_layernorm.weight", H, 1f);
            Vec($"{p}.post_attention_layernorm.weight", H, 1f);
            Lin($"{p}.self_attn.q_proj.weight", qd, H);
            Lin($"{p}.self_attn.k_proj.weight", kvd, H);
            Lin($"{p}.self_attn.v_proj.weight", kvd, H);
            Lin($"{p}.self_attn.o_proj.weight", H, qd);
            Vec($"{p}.self_attn.q_norm.weight", c.HeadDim, 1f);
            Vec($"{p}.self_attn.k_norm.weight", c.HeadDim, 1f);
            Lin($"{p}.mlp.gate_proj.weight", inter, H);
            Lin($"{p}.mlp.up_proj.weight", inter, H);
            Lin($"{p}.mlp.down_proj.weight", H, inter);
        }
        return w;
    }

    private static unsafe Tensor Random3(int c, int h, int w, int seed)
    {
        Tensor t = new(new TensorShape(c, h, w), DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)rng.NextDouble();
        return t;
    }

    private static unsafe Tensor Rand(TensorShape s, int seed, float scale)
    {
        Tensor t = new(s, DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        long n = s.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private static unsafe Tensor Const(TensorShape s, float center, int seed)
    {
        Tensor t = new(s, DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        long n = s.ElementCount;
        for (long i = 0; i < n; i++) p[i] = center + (float)((rng.NextDouble() * 2 - 1) * 0.02);
        return t;
    }

    private static unsafe void AssertFinite(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        for (long i = 0; i < n; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }

    private static void DisposeAll(Dictionary<string, Tensor> w)
    {
        foreach (Tensor t in w.Values) t.Dispose();
    }
}
