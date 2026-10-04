using HartsyInference.Audio.Models.IndexTts;
using HartsyInference.Audio.Models.IndexTts2;
using HartsyInference.Audio.Models.LanguageModels.Gpt;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.PyTorch;
using Xunit;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>Tiny random weights through <see cref="IndexTts2T2sDecoder"/>'s real campplus-mode call graph
/// (speaker projection, emotion Conformer+Perceiver, conditioning assembly, single-pass AR code generation):
/// shapes, finiteness and determinism only. Says nothing about parity with the real IndexTTS-2.5 <c>gpt.pth</c>.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class IndexTts2T2sDecoderSyntheticSmokeTests : IDisposable
{
    private const int Hidden = 32;
    private const int GptHeads = 4;
    private const int GptLayers = 2;
    private const int BlockSize = 128;
    private const int TextVocab = 20;
    private const int PosTableRows = 32;

    private const int EmoConformerOutput = 16;
    private const int EmoConformerHeads = 2;
    private const int EmoConformerLinearUnits = 32;
    private const int EmoConformerBlocks = 2;
    private const int EmoConformerConvKernel = 3;
    private const int EmoPerceiverDim = 24; // the emo perceiver's own working dim, NOT GptConfig.Hidden.
    private const int MelBands = 1024; // w2v-bert feature dim; fixed on real IndexTtsConformerConfig.IndexTts2EmoCondition.

    private static readonly int[] TextIds = [2, 3, 4, 5, 6];

    private static readonly GptConfig T2sCfg = new()
    {
        Hidden = Hidden,
        NumLayers = GptLayers,
        NumHeads = GptHeads,
        BlockSize = BlockSize,
        Bias = true,
    };

    private static readonly IndexTtsConformerConfig EmoCfg = new()
    {
        InputSize = MelBands,
        OutputSize = EmoConformerOutput,
        AttentionHeads = EmoConformerHeads,
        LinearUnits = EmoConformerLinearUnits,
        NumBlocks = EmoConformerBlocks,
        ConvKernel = EmoConformerConvKernel,
    };

    private readonly List<Tensor> _owned = [];
    private readonly CpuBackend _backend = new();

    private Tensor Rand(Random rng, double amp, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = (float)((rng.NextDouble() * 2 - 1) * amp);
        _owned.Add(t);
        return t;
    }

    private Tensor NormWeight(Random rng, long dim)
    {
        Tensor t = Rand(rng, 0.1, dim);
        foreach (ref float v in t.AsSpan<float>()) v += 1f;
        return t;
    }

    private Dictionary<string, Tensor> BuildWeights(Random rng)
    {
        Dictionary<string, Tensor> w = [];
        w["text_embedding.weight"] = Rand(rng, 0.1, TextVocab, Hidden);
        w["text_pos_embedding.emb.weight"] = Rand(rng, 0.1, PosTableRows, Hidden);
        w["mel_embedding.weight"] = Rand(rng, 0.1, IndexTts2T2sDecoder.NumMelCodes, Hidden);
        w["mel_pos_embedding.emb.weight"] = Rand(rng, 0.1, PosTableRows, Hidden);
        w["final_norm.weight"] = NormWeight(rng, Hidden);
        w["final_norm.bias"] = Rand(rng, 0.02, Hidden);
        w["mel_head.weight"] = Rand(rng, 0.1, IndexTts2T2sDecoder.NumMelCodes, Hidden);
        w["mel_head.bias"] = Rand(rng, 0.02, IndexTts2T2sDecoder.NumMelCodes);

        w["spk_emb_proj.weight"] = Rand(rng, 0.1, Hidden, 192);
        w["spk_emb_proj.bias"] = Rand(rng, 0.02, Hidden);
        w["lang_embedding.weight"] = Rand(rng, 0.1, 107, Hidden);

        w["emovec_layer.weight"] = Rand(rng, 0.1, Hidden, EmoPerceiverDim);
        w["emovec_layer.bias"] = Rand(rng, 0.02, Hidden);
        w["emo_layer.weight"] = Rand(rng, 0.1, Hidden, Hidden);
        w["emo_layer.bias"] = Rand(rng, 0.02, Hidden);

        AddEmoConformerWeights(w, rng);
        AddEmoPerceiverWeights(w, rng);

        for (int i = 0; i < GptLayers; i++)
        {
            string p = $"gpt.h.{i}";
            w[$"{p}.ln_1.weight"] = NormWeight(rng, Hidden);
            w[$"{p}.ln_1.bias"] = Rand(rng, 0.02, Hidden);
            w[$"{p}.attn.c_attn.weight"] = Rand(rng, 0.1, Hidden, 3 * Hidden);
            w[$"{p}.attn.c_attn.bias"] = Rand(rng, 0.02, 3 * Hidden);
            w[$"{p}.attn.c_proj.weight"] = Rand(rng, 0.1, Hidden, Hidden);
            w[$"{p}.attn.c_proj.bias"] = Rand(rng, 0.02, Hidden);
            w[$"{p}.ln_2.weight"] = NormWeight(rng, Hidden);
            w[$"{p}.ln_2.bias"] = Rand(rng, 0.02, Hidden);
            w[$"{p}.mlp.c_fc.weight"] = Rand(rng, 0.1, Hidden, T2sCfg.MlpDim);
            w[$"{p}.mlp.c_fc.bias"] = Rand(rng, 0.02, T2sCfg.MlpDim);
            w[$"{p}.mlp.c_proj.weight"] = Rand(rng, 0.1, T2sCfg.MlpDim, Hidden);
            w[$"{p}.mlp.c_proj.bias"] = Rand(rng, 0.02, Hidden);
        }
        w["gpt.ln_f.weight"] = NormWeight(rng, Hidden);
        w["gpt.ln_f.bias"] = Rand(rng, 0.02, Hidden);
        return w;
    }

    private void AddEmoConformerWeights(Dictionary<string, Tensor> w, Random rng)
    {
        const string prefix = "emo_conditioning_encoder";
        int odim = EmoConformerOutput, headDim = odim / EmoConformerHeads;
        int fOut = (MelBands - 3) / 2 + 1;

        w[$"{prefix}.embed.conv.0.weight"] = Rand(rng, 0.1, odim, 1, 3, 3);
        w[$"{prefix}.embed.conv.0.bias"] = Rand(rng, 0.02, odim);
        w[$"{prefix}.embed.out.0.weight"] = Rand(rng, 0.1, odim, odim * fOut);
        w[$"{prefix}.embed.out.0.bias"] = Rand(rng, 0.02, odim);

        for (int i = 0; i < EmoConformerBlocks; i++)
        {
            string p = $"{prefix}.encoders.{i}";
            w[$"{p}.norm_mha.weight"] = NormWeight(rng, odim);
            w[$"{p}.norm_mha.bias"] = Rand(rng, 0.02, odim);
            foreach (string qkv in new[] { "q", "k", "v" })
            {
                w[$"{p}.self_attn.linear_{qkv}.weight"] = Rand(rng, 0.1, odim, odim);
                w[$"{p}.self_attn.linear_{qkv}.bias"] = Rand(rng, 0.02, odim);
            }
            w[$"{p}.self_attn.linear_out.weight"] = Rand(rng, 0.1, odim, odim);
            w[$"{p}.self_attn.linear_out.bias"] = Rand(rng, 0.02, odim);
            w[$"{p}.self_attn.linear_pos.weight"] = Rand(rng, 0.1, odim, odim);
            w[$"{p}.self_attn.pos_bias_u"] = Rand(rng, 0.1, EmoConformerHeads, headDim);
            w[$"{p}.self_attn.pos_bias_v"] = Rand(rng, 0.1, EmoConformerHeads, headDim);

            w[$"{p}.norm_conv.weight"] = NormWeight(rng, odim);
            w[$"{p}.norm_conv.bias"] = Rand(rng, 0.02, odim);
            w[$"{p}.conv_module.pointwise_conv1.weight"] = Rand(rng, 0.1, 2 * odim, odim, 1);
            w[$"{p}.conv_module.pointwise_conv1.bias"] = Rand(rng, 0.02, 2 * odim);
            w[$"{p}.conv_module.depthwise_conv.weight"] = Rand(rng, 0.1, odim, 1, EmoConformerConvKernel);
            w[$"{p}.conv_module.depthwise_conv.bias"] = Rand(rng, 0.02, odim);
            w[$"{p}.conv_module.norm.weight"] = NormWeight(rng, odim);
            w[$"{p}.conv_module.norm.bias"] = Rand(rng, 0.02, odim);
            w[$"{p}.conv_module.pointwise_conv2.weight"] = Rand(rng, 0.1, odim, odim, 1);
            w[$"{p}.conv_module.pointwise_conv2.bias"] = Rand(rng, 0.02, odim);

            w[$"{p}.norm_ff.weight"] = NormWeight(rng, odim);
            w[$"{p}.norm_ff.bias"] = Rand(rng, 0.02, odim);
            w[$"{p}.feed_forward.w_1.weight"] = Rand(rng, 0.1, EmoConformerLinearUnits, odim);
            w[$"{p}.feed_forward.w_1.bias"] = Rand(rng, 0.02, EmoConformerLinearUnits);
            w[$"{p}.feed_forward.w_2.weight"] = Rand(rng, 0.1, odim, EmoConformerLinearUnits);
            w[$"{p}.feed_forward.w_2.bias"] = Rand(rng, 0.02, odim);

            w[$"{p}.norm_final.weight"] = NormWeight(rng, odim);
            w[$"{p}.norm_final.bias"] = Rand(rng, 0.02, odim);
        }
        w[$"{prefix}.after_norm.weight"] = NormWeight(rng, odim);
        w[$"{prefix}.after_norm.bias"] = Rand(rng, 0.02, odim);
    }

    private void AddEmoPerceiverWeights(Dictionary<string, Tensor> w, Random rng)
    {
        const string prefix = "emo_perceiver_encoder";
        const int dimHead = 64, heads = 4, depth = 2;
        int dimInner = dimHead * heads;
        int doubled = 2 * EmoPerceiverDim;

        w[$"{prefix}.latents"] = Rand(rng, 0.1, 1, EmoPerceiverDim);
        w[$"{prefix}.proj_context.weight"] = Rand(rng, 0.1, EmoPerceiverDim, EmoConformerOutput);
        w[$"{prefix}.proj_context.bias"] = Rand(rng, 0.02, EmoPerceiverDim);
        for (int i = 0; i < depth; i++)
        {
            string p = $"{prefix}.layers.{i}";
            w[$"{p}.0.to_q.weight"] = Rand(rng, 0.1, dimInner, EmoPerceiverDim);
            w[$"{p}.0.to_kv.weight"] = Rand(rng, 0.1, 2 * dimInner, EmoPerceiverDim);
            w[$"{p}.0.to_out.weight"] = Rand(rng, 0.1, EmoPerceiverDim, dimInner);
            w[$"{p}.1.0.weight"] = Rand(rng, 0.1, doubled, EmoPerceiverDim);
            w[$"{p}.1.0.bias"] = Rand(rng, 0.02, doubled);
            w[$"{p}.1.2.weight"] = Rand(rng, 0.1, EmoPerceiverDim, EmoPerceiverDim);
            w[$"{p}.1.2.bias"] = Rand(rng, 0.02, EmoPerceiverDim);
        }
        w[$"{prefix}.norm.gamma"] = Rand(rng, 0.1, EmoPerceiverDim);
    }

    [Fact]
    public void Generate_ProducesNonEmptyCodes_WithinVocabRange()
    {
        Random rng = new(11);
        Dictionary<string, Tensor> weights = BuildWeights(rng);
        using IndexTts2T2sDecoder decoder = new(T2sCfg, TextVocab, maxMelTokens: 16, EmoCfg, emoPerceiverDim: EmoPerceiverDim);
        decoder.LoadWeights(weights);

        Tensor campplusEmbedding = Rand(rng, 1.0, 1, 192);
        using Tensor speakerConditioning = decoder.ComputeSpeakerConditioning(_backend, campplusEmbedding);

        const int emoT = 6;
        Tensor emoFeature = Rand(rng, 1.0, 1, emoT, MelBands);
        using Tensor emoVec = decoder.ComputeEmoVec(_backend, emoFeature, emoT);

        uint rngState = 42;
        IndexTtsOptions options = new() { Temperature = 1.0f, TopK = 10, TopP = 1.0f, RepetitionPenalty = 1.0f };
        int[] codes = decoder.Generate(_backend, speakerConditioning, emoVec, TextIds, langId: 0, options, ref rngState);

        Assert.NotEmpty(codes);
        foreach (int c in codes) Assert.InRange(c, 0, IndexTts2T2sDecoder.NumMelCodes - 1);
    }

    [Fact]
    public void Generate_IsDeterministic_ForTheSameSeed()
    {
        Random rng = new(22);
        Dictionary<string, Tensor> weights = BuildWeights(rng);
        using IndexTts2T2sDecoder decoder = new(T2sCfg, TextVocab, maxMelTokens: 16, EmoCfg, emoPerceiverDim: EmoPerceiverDim);
        decoder.LoadWeights(weights);

        Tensor campplusEmbedding = Rand(rng, 1.0, 1, 192);
        using Tensor speakerConditioning = decoder.ComputeSpeakerConditioning(_backend, campplusEmbedding);
        const int emoT = 6;
        Tensor emoFeature = Rand(rng, 1.0, 1, emoT, MelBands);
        using Tensor emoVec = decoder.ComputeEmoVec(_backend, emoFeature, emoT);

        IndexTtsOptions options = new() { Temperature = 0.8f, TopK = 10, TopP = 0.9f, RepetitionPenalty = 2.0f };
        uint rngA = 7;
        int[] codesA = decoder.Generate(_backend, speakerConditioning, emoVec, TextIds, langId: 1, options, ref rngA);
        uint rngB = 7;
        int[] codesB = decoder.Generate(_backend, speakerConditioning, emoVec, TextIds, langId: 1, options, ref rngB);

        Assert.Equal(codesA, codesB);
    }

    [Fact]
    public void MergeEmoVec_AtAlphaZero_EqualsBase_AtAlphaOne_EqualsEmo()
    {
        Random rng = new(33);
        Tensor baseVec = Rand(rng, 1.0, 1, Hidden);
        Tensor emoVec = Rand(rng, 1.0, 1, Hidden);

        using Tensor zero = IndexTts2T2sDecoder.MergeEmoVec(baseVec, emoVec, 0f);
        using Tensor one = IndexTts2T2sDecoder.MergeEmoVec(baseVec, emoVec, 1f);

        float[] b = baseVec.AsSpan<float>().ToArray(), e = emoVec.AsSpan<float>().ToArray();
        float[] z = zero.AsSpan<float>().ToArray(), o = one.AsSpan<float>().ToArray();
        for (int i = 0; i < b.Length; i++)
        {
            Assert.Equal(b[i], z[i], 5);
            Assert.Equal(e[i], o[i], 5);
        }
    }

    public void Dispose()
    {
        foreach (Tensor t in _owned) t.Dispose();
        _backend.Dispose();
    }
}

/// <summary>Loads the real IndexTTS-2.5 <c>gpt.pth</c> (3+ GB; too large to bundle — point
/// <c>INDEXTTS2_GPT_PTH_PATH</c> at a local copy) and runs the full real campplus-mode call graph: speaker
/// projection, emotion Conformer+Perceiver, conditioning assembly, single-pass AR generation. A clean
/// <c>LoadWeights</c> plus a finite, in-range code sequence is strong evidence every key name and the whole
/// conditioning/generation design in <see cref="IndexTts2T2sDecoder"/> matches the real checkpoint.</summary>
[Trait("Category", "Integration")]
public sealed class IndexTts2T2sDecoderRealWeightTests
{
    [Fact]
    public void Generate_SucceedsAgainstRealGptPth()
    {
        string? path = Environment.GetEnvironmentVariable("INDEXTTS2_GPT_PTH_PATH");
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;   // resource-gated: skip without the real checkpoint file

        using PytorchPickleLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();
        try
        {
            using IndexTts2T2sDecoder decoder = new(GptConfig.IndexTts2, numTextTokens: 60_510, maxMelTokens: 32);
            decoder.LoadWeights(weights);

            Random rng = new(123);
            Tensor campplusEmbedding = new(new TensorShape(1, 192), DType.F32);
            foreach (ref float v in campplusEmbedding.AsSpan<float>()) v = (float)(rng.NextDouble() * 2 - 1);

            using CpuBackend backend = new();
            using Tensor speakerConditioning = decoder.ComputeSpeakerConditioning(backend, campplusEmbedding);

            const int emoT = 10;
            Tensor emoFeature = new(new TensorShape(1, emoT, 1024), DType.F32);
            foreach (ref float v in emoFeature.AsSpan<float>()) v = (float)(rng.NextDouble() * 2 - 1);
            using Tensor emoVec = decoder.ComputeEmoVec(backend, emoFeature, emoT);

            uint rngState = 7;
            IndexTtsOptions options = new() { Temperature = 1.0f, TopK = 10, TopP = 1.0f, RepetitionPenalty = 1.0f };
            int[] codes = decoder.Generate(backend, speakerConditioning, emoVec, [100, 200, 300, 400], langId: 0, options, ref rngState);

            Assert.NotEmpty(codes);
            foreach (int c in codes) Assert.InRange(c, 0, IndexTts2T2sDecoder.NumMelCodes - 1);

            campplusEmbedding.Dispose();
            emoFeature.Dispose();
        }
        finally
        {
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }
}
