using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Hunyuan-Image 2.1's glyph stream, transcribed from ComfyUI (<c>comfy/text_encoders/hunyuan_image.py</c>,
/// <c>byt5_config_small_glyph.json</c>, <c>byt5_tokenizer/added_tokens.json</c> @6b904b5faf): which quoted text becomes
/// glyph text and in what order, how ByT5 tokenizes it, the encoder preset, and that the DiT reads the second stream.</summary>
public sealed class HunyuanImageGlyphTests
{
    [Theory]
    [InlineData("a cat on a sofa", null)]
    [InlineData("a sign saying \"OPEN\"", "Text \"OPEN\". ")]
    [InlineData("\"OPEN\" and \"CLOSED\" signs", "Text \"OPEN\". Text \"CLOSED\". ")]
    // ASCII spans first, then ‘…’, then “…”, whatever their order in the prompt.
    [InlineData("“你好” then ‘x’ then \"Hi\"", "Text \"Hi\". Text \"x\". Text \"你好\". ")]
    [InlineData("an empty \"\" quote", "Text \"\". ")]
    // Lazy: the first closing quote ends the span.
    [InlineData("\"a\" b \"c\"", "Text \"a\". Text \"c\". ")]
    // '.' stops at a newline, as in Python's re.
    [InlineData("\"line\nbreak\"", null)]
    public void GlyphText_FollowsComfysQuoteRules(string prompt, string? expected)
    {
        Assert.Equal(expected, HunyuanImageGlyphText.Extract(prompt));
    }

    [Fact]
    public void ByT5_BytesShiftByThreeAndEndWithEos()
    {
        Assert.Equal(new[] { 'H' + 3, 'i' + 3, ByT5Tokenizer.EosId }, ByT5Tokenizer.Encode("Hi"));
        Assert.Equal(new[] { ByT5Tokenizer.EosId }, ByT5Tokenizer.Encode(""));
        // "é" is 0xC3 0xA9; "你" is 0xE4 0xBD 0xA0.
        Assert.Equal(new[] { 0xC3 + 3, 0xA9 + 3, 1 }, ByT5Tokenizer.Encode("é"));
        Assert.Equal(new[] { 0xE4 + 3, 0xBD + 3, 0xA0 + 3, 1 }, ByT5Tokenizer.Encode("你"));
    }

    [Fact]
    public void ByT5_SentinelsAndSpecialsMatchAsWholeTokens()
    {
        Assert.Equal(new[] { 259, 1 }, ByT5Tokenizer.Encode("<extra_id_0>"));
        Assert.Equal(new[] { 'a' + 3, 383, 'b' + 3, 1 }, ByT5Tokenizer.Encode("a<extra_id_124>b"));
        Assert.Equal(new[] { 'x' + 3, 1, 'y' + 3, 1 }, ByT5Tokenizer.Encode("x</s>y"));
        // Out of range or malformed sentinels stay bytes.
        Assert.Equal("<extra_id_125>".Length + 1, ByT5Tokenizer.Encode("<extra_id_125>").Length);
        Assert.Equal("<extra_id_>".Length + 1, ByT5Tokenizer.Encode("<extra_id_>").Length);
    }

    [Fact]
    public void ByT5SmallGlyph_MatchesComfysConfig()
    {
        T5TextEncoderConfig c = T5TextEncoderConfig.ByT5SmallGlyph;

        Assert.Equal((1472, 3584, 64, 6, 12, 1510), (c.DModel, c.DFf, c.DKv, c.NumHeads, c.NumLayers, c.VocabSize));
        Assert.Equal((32, 128, 1e-6f), (c.RelativeAttentionNumBuckets, c.RelativeAttentionMaxDistance, c.LayerNormEpsilon));
        Assert.True(c.GatedFeedForward);
        Assert.False(c.UsePerLayerPositionBias);
    }

    [Fact]
    public void T5Encoder_RunsWithByT5sNarrowAttention()
    {
        // ByT5-small's attention is 6 heads × 64 = 384 wide inside a 1472-wide model; the tiny copy keeps that mismatch.
        T5TextEncoderConfig cfg = new() { DModel = 24, DFf = 40, DKv = 4, NumHeads = 2, NumLayers = 2, VocabSize = 300 };
        Dictionary<string, Tensor> w = new();
        int seed = 1;
        void Mat(string k, int o, int i) => w[k] = HunyuanImageWeightBuilder.Rand(new TensorShape(o, i), seed++, 0.2f);
        int inner = cfg.NumHeads * cfg.DKv;
        Mat("shared.weight", cfg.VocabSize, cfg.DModel);
        Mat("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", cfg.RelativeAttentionNumBuckets, cfg.NumHeads);
        for (int l = 0; l < cfg.NumLayers; l++)
        {
            string p = $"encoder.block.{l}.layer";
            Mat($"{p}.0.SelfAttention.q.weight", inner, cfg.DModel);
            Mat($"{p}.0.SelfAttention.k.weight", inner, cfg.DModel);
            Mat($"{p}.0.SelfAttention.v.weight", inner, cfg.DModel);
            Mat($"{p}.0.SelfAttention.o.weight", cfg.DModel, inner);
            Mat($"{p}.1.DenseReluDense.wi_0.weight", cfg.DFf, cfg.DModel);
            Mat($"{p}.1.DenseReluDense.wi_1.weight", cfg.DFf, cfg.DModel);
            Mat($"{p}.1.DenseReluDense.wo.weight", cfg.DModel, cfg.DFf);
            w[$"{p}.0.layer_norm.weight"] = Ones(cfg.DModel);
            w[$"{p}.1.layer_norm.weight"] = Ones(cfg.DModel);
        }
        w["encoder.final_layer_norm.weight"] = Ones(cfg.DModel);
        try
        {
            using CpuBackend backend = new CpuBackend();
            using T5TextEncoder encoder = new T5TextEncoder(cfg);
            encoder.LoadWeights(w);
            int[] ids = ByT5Tokenizer.Encode("Text \"Hi\". ");
            using Tensor hidden = encoder.Encode(backend, [ids]);

            Assert.Equal(new TensorShape(1, ids.Length, cfg.DModel), hidden.Shape);
            Assert.All(hidden.AsSpan<float>().ToArray(), v => Assert.True(float.IsFinite(v)));
        }
        finally
        {
            HunyuanImageWeightBuilder.DisposeAll(w);
        }
    }

    private static Tensor Ones(int n)
    {
        Tensor t = new Tensor(new TensorShape(n), DType.F32);
        t.AsSpan<float>().Fill(1f);
        return t;
    }

    [Fact]
    public void Dit_ReadsTheGlyphStream()
    {
        HunyuanImageConfig cfg = new()
        {
            HiddenSize = 128, NumHeads = 1, HeadDim = 128, NumDoubleBlocks = 1, NumSingleBlocks = 1, NumRefinerLayers = 1,
            PatchSize = 1, InChannels = 16, TextEmbedDim = 32, TextEmbedDim2 = 12, GuidanceEmbed = false, MlpRatio = 2.0f,
        };
        Dictionary<string, Tensor> weights = HunyuanImageWeightBuilder.Build(cfg);
        try
        {
            using CpuBackend backend = new CpuBackend();
            using HunyuanImageTransformer dit = new HunyuanImageTransformer(cfg);
            dit.LoadWeights(weights);
            using Tensor latent = HunyuanImageWeightBuilder.Rand(new TensorShape(1, 16, 16), 100, 0.5f);
            using Tensor text = HunyuanImageWeightBuilder.Rand(new TensorShape(1, 5, 32), 200, 0.2f);
            using Tensor glyph = HunyuanImageWeightBuilder.Rand(new TensorShape(1, 3, 12), 300, 0.5f);

            float[] plain = Run(dit, backend, latent, text, null);
            float[] withGlyph = Run(dit, backend, latent, text, glyph);
            float[] again = Run(dit, backend, latent, text, glyph);

            Assert.Equal(withGlyph, again);
            double diff = plain.Zip(withGlyph, (a, b) => Math.Abs(a - b)).Max();
            Assert.True(diff > 1e-6, $"the glyph stream changed nothing (max |Δ| {diff:E2})");
        }
        finally
        {
            HunyuanImageWeightBuilder.DisposeAll(weights);
        }
    }

    private static float[] Run(HunyuanImageTransformer dit, CpuBackend backend, Tensor latent, Tensor text, Tensor? glyph)
    {
        using Tensor v = dit.Forward(backend, latent, text, glyph, 700f, 1f, 4, 4);
        return v.AsSpan<float>().ToArray();
    }
}
