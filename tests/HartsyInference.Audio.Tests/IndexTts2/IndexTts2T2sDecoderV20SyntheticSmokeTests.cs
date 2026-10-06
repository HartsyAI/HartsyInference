using HartsyInference.Audio.Models.IndexTts;
using HartsyInference.Audio.Models.IndexTts2;
using HartsyInference.Audio.Models.LanguageModels.Gpt;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>Tiny random weights through <see cref="IndexTts2T2sDecoder"/>'s IndexTTS-2.0 call graph
/// (<c>condition_type: conformer_perceiver</c>): checkpoint-driven mode selection, the Conformer+Perceiver speaker
/// latents, the <c>speed_emb</c> conditioning slots, single-pass AR generation, and the second GPT pass +
/// <c>gpt_layer</c>. Shapes, finiteness, determinism and structural invariants only — says nothing about parity with
/// the real IndexTTS-2.0 <c>gpt.pth</c> (the env-gated real-weight tests cover that).</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class IndexTts2T2sDecoderV20SyntheticSmokeTests : IDisposable
{
    private const int Hidden = 32;
    private const int GptLayers = 2;
    private const int TextVocab = 20;
    private const int PosTableRows = 48;
    private const int W2vDim = 1024;
    private const int SpkLatents = 32;
    private const int EmoPerceiverDim = 24;
    private const int GptLayerOut = 1024;

    private static readonly int[] TextIds = [2, 3, 4, 5, 6];

    private static readonly GptConfig GptCfg = new() { Hidden = Hidden, NumLayers = GptLayers, NumHeads = 4, BlockSize = 160, Bias = true };

    private static readonly IndexTtsConformerConfig EmoCfg = SmallConformer(heads: 2, units: 32, blocks: 2);
    private static readonly IndexTtsConformerConfig SpkCfg = SmallConformer(heads: 2, units: 32, blocks: 2);

    private static IndexTtsConformerConfig SmallConformer(int heads, int units, int blocks) => new()
    {
        InputSize = W2vDim, OutputSize = 16, AttentionHeads = heads, LinearUnits = units, NumBlocks = blocks, ConvKernel = 3,
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

    private Tensor Norm(Random rng, long dim)
    {
        Tensor t = Rand(rng, 0.1, dim);
        foreach (ref float v in t.AsSpan<float>()) v += 1f;
        return t;
    }

    private void AddConformer(Dictionary<string, Tensor> w, Random rng, string prefix, IndexTtsConformerConfig cfg)
    {
        int odim = cfg.OutputSize, headDim = odim / cfg.AttentionHeads, fOut = (cfg.InputSize - 3) / 2 + 1;
        w[$"{prefix}.embed.conv.0.weight"] = Rand(rng, 0.1, odim, 1, 3, 3);
        w[$"{prefix}.embed.conv.0.bias"] = Rand(rng, 0.02, odim);
        w[$"{prefix}.embed.out.0.weight"] = Rand(rng, 0.1, odim, odim * fOut);
        w[$"{prefix}.embed.out.0.bias"] = Rand(rng, 0.02, odim);
        for (int i = 0; i < cfg.NumBlocks; i++)
        {
            string p = $"{prefix}.encoders.{i}";
            w[$"{p}.norm_mha.weight"] = Norm(rng, odim);
            w[$"{p}.norm_mha.bias"] = Rand(rng, 0.02, odim);
            foreach (string qkv in new[] { "q", "k", "v" })
            {
                w[$"{p}.self_attn.linear_{qkv}.weight"] = Rand(rng, 0.1, odim, odim);
                w[$"{p}.self_attn.linear_{qkv}.bias"] = Rand(rng, 0.02, odim);
            }
            w[$"{p}.self_attn.linear_out.weight"] = Rand(rng, 0.1, odim, odim);
            w[$"{p}.self_attn.linear_out.bias"] = Rand(rng, 0.02, odim);
            w[$"{p}.self_attn.linear_pos.weight"] = Rand(rng, 0.1, odim, odim);
            w[$"{p}.self_attn.pos_bias_u"] = Rand(rng, 0.1, cfg.AttentionHeads, headDim);
            w[$"{p}.self_attn.pos_bias_v"] = Rand(rng, 0.1, cfg.AttentionHeads, headDim);
            w[$"{p}.norm_conv.weight"] = Norm(rng, odim);
            w[$"{p}.norm_conv.bias"] = Rand(rng, 0.02, odim);
            w[$"{p}.conv_module.pointwise_conv1.weight"] = Rand(rng, 0.1, 2 * odim, odim, 1);
            w[$"{p}.conv_module.pointwise_conv1.bias"] = Rand(rng, 0.02, 2 * odim);
            w[$"{p}.conv_module.depthwise_conv.weight"] = Rand(rng, 0.1, odim, 1, cfg.ConvKernel);
            w[$"{p}.conv_module.depthwise_conv.bias"] = Rand(rng, 0.02, odim);
            w[$"{p}.conv_module.norm.weight"] = Norm(rng, odim);
            w[$"{p}.conv_module.norm.bias"] = Rand(rng, 0.02, odim);
            w[$"{p}.conv_module.pointwise_conv2.weight"] = Rand(rng, 0.1, odim, odim, 1);
            w[$"{p}.conv_module.pointwise_conv2.bias"] = Rand(rng, 0.02, odim);
            w[$"{p}.norm_ff.weight"] = Norm(rng, odim);
            w[$"{p}.norm_ff.bias"] = Rand(rng, 0.02, odim);
            w[$"{p}.feed_forward.w_1.weight"] = Rand(rng, 0.1, cfg.LinearUnits, odim);
            w[$"{p}.feed_forward.w_1.bias"] = Rand(rng, 0.02, cfg.LinearUnits);
            w[$"{p}.feed_forward.w_2.weight"] = Rand(rng, 0.1, odim, cfg.LinearUnits);
            w[$"{p}.feed_forward.w_2.bias"] = Rand(rng, 0.02, odim);
            w[$"{p}.norm_final.weight"] = Norm(rng, odim);
            w[$"{p}.norm_final.bias"] = Rand(rng, 0.02, odim);
        }
        w[$"{prefix}.after_norm.weight"] = Norm(rng, odim);
        w[$"{prefix}.after_norm.bias"] = Rand(rng, 0.02, odim);
    }

    private void AddPerceiver(Dictionary<string, Tensor> w, Random rng, string prefix, int dim, int dimContext, int latents, int heads)
    {
        const int dimHead = 64, depth = 2;
        int dimInner = dimHead * heads;
        w[$"{prefix}.latents"] = Rand(rng, 0.1, latents, dim);
        w[$"{prefix}.proj_context.weight"] = Rand(rng, 0.1, dim, dimContext);
        w[$"{prefix}.proj_context.bias"] = Rand(rng, 0.02, dim);
        for (int i = 0; i < depth; i++)
        {
            string p = $"{prefix}.layers.{i}";
            w[$"{p}.0.to_q.weight"] = Rand(rng, 0.1, dimInner, dim);
            w[$"{p}.0.to_kv.weight"] = Rand(rng, 0.1, 2 * dimInner, dim);
            w[$"{p}.0.to_out.weight"] = Rand(rng, 0.1, dim, dimInner);
            w[$"{p}.1.0.weight"] = Rand(rng, 0.1, 2 * dim, dim);
            w[$"{p}.1.0.bias"] = Rand(rng, 0.02, 2 * dim);
            w[$"{p}.1.2.weight"] = Rand(rng, 0.1, dim, dim);
            w[$"{p}.1.2.bias"] = Rand(rng, 0.02, dim);
        }
        w[$"{prefix}.norm.gamma"] = Rand(rng, 0.1, dim);
    }

    /// <summary>A 2.0-shaped checkpoint: conditioning_encoder/perceiver_encoder/speed_emb, no spk_emb_proj/lang_embedding.</summary>
    private Dictionary<string, Tensor> BuildV20Weights(Random rng)
    {
        Dictionary<string, Tensor> w = [];
        w["text_embedding.weight"] = Rand(rng, 0.1, TextVocab, Hidden);
        w["text_pos_embedding.emb.weight"] = Rand(rng, 0.1, PosTableRows, Hidden);
        w["mel_embedding.weight"] = Rand(rng, 0.1, IndexTts2T2sDecoder.NumMelCodes, Hidden);
        w["mel_pos_embedding.emb.weight"] = Rand(rng, 0.1, PosTableRows, Hidden);
        w["final_norm.weight"] = Norm(rng, Hidden);
        w["final_norm.bias"] = Rand(rng, 0.02, Hidden);
        w["mel_head.weight"] = Rand(rng, 0.1, IndexTts2T2sDecoder.NumMelCodes, Hidden);
        w["mel_head.bias"] = Rand(rng, 0.02, IndexTts2T2sDecoder.NumMelCodes);
        w["speed_emb.weight"] = Rand(rng, 0.5, 2, Hidden);
        w["emovec_layer.weight"] = Rand(rng, 0.1, Hidden, EmoPerceiverDim);
        w["emovec_layer.bias"] = Rand(rng, 0.02, Hidden);
        w["emo_layer.weight"] = Rand(rng, 0.1, Hidden, Hidden);
        w["emo_layer.bias"] = Rand(rng, 0.02, Hidden);

        AddConformer(w, rng, "conditioning_encoder", SpkCfg);
        AddPerceiver(w, rng, "perceiver_encoder", dim: Hidden, dimContext: SpkCfg.OutputSize, latents: SpkLatents, heads: 8);
        AddConformer(w, rng, "emo_conditioning_encoder", EmoCfg);
        AddPerceiver(w, rng, "emo_perceiver_encoder", dim: EmoPerceiverDim, dimContext: EmoCfg.OutputSize, latents: 1, heads: 4);

        for (int i = 0; i < GptLayers; i++)
        {
            string p = $"gpt.h.{i}";
            w[$"{p}.ln_1.weight"] = Norm(rng, Hidden);
            w[$"{p}.ln_1.bias"] = Rand(rng, 0.02, Hidden);
            w[$"{p}.attn.c_attn.weight"] = Rand(rng, 0.1, Hidden, 3 * Hidden);
            w[$"{p}.attn.c_attn.bias"] = Rand(rng, 0.02, 3 * Hidden);
            w[$"{p}.attn.c_proj.weight"] = Rand(rng, 0.1, Hidden, Hidden);
            w[$"{p}.attn.c_proj.bias"] = Rand(rng, 0.02, Hidden);
            w[$"{p}.ln_2.weight"] = Norm(rng, Hidden);
            w[$"{p}.ln_2.bias"] = Rand(rng, 0.02, Hidden);
            w[$"{p}.mlp.c_fc.weight"] = Rand(rng, 0.1, Hidden, GptCfg.MlpDim);
            w[$"{p}.mlp.c_fc.bias"] = Rand(rng, 0.02, GptCfg.MlpDim);
            w[$"{p}.mlp.c_proj.weight"] = Rand(rng, 0.1, GptCfg.MlpDim, Hidden);
            w[$"{p}.mlp.c_proj.bias"] = Rand(rng, 0.02, Hidden);
        }
        w["gpt.ln_f.weight"] = Norm(rng, Hidden);
        w["gpt.ln_f.bias"] = Rand(rng, 0.02, Hidden);
        return w;
    }

    /// <summary><c>net.gpt_layer.{0,1,2}</c>: Linear(h,256), Linear(256,128), Linear(128,1024).</summary>
    private Dictionary<string, Tensor> BuildGptLayerWeights(Random rng)
    {
        Dictionary<string, Tensor> w = [];
        int[] dims = [Hidden, 256, 128, GptLayerOut];
        for (int i = 0; i < 3; i++)
        {
            w[$"net.gpt_layer.{i}.weight"] = Rand(rng, 0.1, dims[i + 1], dims[i]);
            w[$"net.gpt_layer.{i}.bias"] = Rand(rng, 0.02, dims[i + 1]);
        }
        return w;
    }

    private IndexTts2T2sDecoder LoadDecoder(Random rng, bool withGptLayer = true)
    {
        IndexTts2T2sDecoder decoder = new(GptCfg, TextVocab, maxMelTokens: 16, EmoCfg, EmoPerceiverDim, SpkCfg);
        decoder.LoadWeights(BuildV20Weights(rng));
        if (withGptLayer) decoder.LoadGptLayer(BuildGptLayerWeights(rng), "net.gpt_layer");
        return decoder;
    }

    private (Tensor Speaker, Tensor EmoVec) Conditioning(IndexTts2T2sDecoder decoder, Random rng)
    {
        const int t = 8;
        Tensor feature = Rand(rng, 1.0, 1, t, W2vDim);
        Tensor speaker = decoder.ComputeSpeakerConditioningConformerPerceiver(_backend, feature, t);
        Tensor emoFeature = Rand(rng, 1.0, 1, 6, W2vDim);
        Tensor emoVec = decoder.ComputeEmoVec(_backend, emoFeature, 6);
        return (speaker, emoVec);
    }

    [Fact]
    public void LoadWeights_DetectsConformerPerceiverMode_FromTheCheckpointKeys()
    {
        using IndexTts2T2sDecoder decoder = LoadDecoder(new Random(1));
        Assert.Equal(IndexTts2SpeakerConditioning.ConformerPerceiver, decoder.Mode);
    }

    [Fact]
    public void SpeakerConditioning_IsThirtyTwoFiniteLatents()
    {
        Random rng = new(2);
        using IndexTts2T2sDecoder decoder = LoadDecoder(rng);
        (Tensor speaker, Tensor emoVec) = Conditioning(decoder, rng);
        using (speaker) using (emoVec)
        {
            Assert.Equal(new TensorShape(1, SpkLatents, Hidden), speaker.Shape);
            foreach (float v in speaker.AsSpan<float>()) Assert.True(float.IsFinite(v));
        }
    }

    [Fact]
    public void CampplusEntryPoint_ThrowsOnAConformerPerceiverCheckpoint()
    {
        Random rng = new(3);
        using IndexTts2T2sDecoder decoder = LoadDecoder(rng);
        Tensor style = Rand(rng, 1.0, 1, 192);
        Assert.Throws<InvalidOperationException>(() => decoder.ComputeSpeakerConditioning(_backend, style));
    }

    [Fact]
    public void ConformerPerceiverConditioning_IsLatentsPlusEmoThenSpeedRowOneThenRowZero()
    {
        Random rng = new(4);
        Tensor latents = Rand(rng, 1.0, 1, 3, 5);
        Tensor emo = Rand(rng, 1.0, 1, 5);
        Tensor speed = Rand(rng, 1.0, 2, 5);

        using Tensor conds = IndexTts2T2sDecoder.BuildConformerPerceiverConditioning(latents, emo, speed, 5);

        Assert.Equal(new TensorShape(1, 5, 5), conds.Shape);
        float[] c = conds.AsSpan<float>().ToArray(), l = latents.AsSpan<float>().ToArray();
        float[] e = emo.AsSpan<float>().ToArray(), s = speed.AsSpan<float>().ToArray();
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 5; j++)
                Assert.Equal(l[i * 5 + j] + e[j], c[i * 5 + j], 5);
        for (int j = 0; j < 5; j++)
        {
            Assert.Equal(s[5 + j], c[3 * 5 + j], 5);   // speed_emb(ones) — the "half" slot comes first
            Assert.Equal(s[j], c[4 * 5 + j], 5);       // speed_emb(zeros)
        }
    }

    [Fact]
    public void Generate_ProducesInRangeCodes_AndIgnoresTheLanguageId()
    {
        Random rng = new(5);
        using IndexTts2T2sDecoder decoder = LoadDecoder(rng);
        (Tensor speaker, Tensor emoVec) = Conditioning(decoder, rng);
        using (speaker) using (emoVec)
        {
            IndexTtsOptions options = new() { Temperature = 1.0f, TopK = 10, TopP = 1.0f, RepetitionPenalty = 1.0f };
            uint a = 42, b = 42;
            int[] withLang = decoder.Generate(_backend, speaker, emoVec, TextIds, langId: 7, options, ref a);
            int[] noLang = decoder.Generate(_backend, speaker, emoVec, TextIds, langId: null, options, ref b);

            Assert.NotEmpty(withLang);
            foreach (int code in withLang) Assert.InRange(code, 0, IndexTts2T2sDecoder.NumMelCodes - 1);
            Assert.Equal(noLang, withLang);   // conformer_perceiver mode has no lang_embedding at all
        }
    }

    [Fact]
    public void SecondPassLatent_HasOneRowPerCode_AndIsFiniteAndDeterministic()
    {
        Random rng = new(6);
        using IndexTts2T2sDecoder decoder = LoadDecoder(rng);
        (Tensor speaker, Tensor emoVec) = Conditioning(decoder, rng);
        using (speaker) using (emoVec)
        {
            int[] codes = [10, 20, 30, 40, 50, 60];
            using Tensor a = decoder.ComputeSecondPassLatent(_backend, speaker, emoVec, TextIds, codes);
            using Tensor b = decoder.ComputeSecondPassLatent(_backend, speaker, emoVec, TextIds, codes);

            Assert.Equal(new TensorShape(1, codes.Length, GptLayerOut), a.Shape);
            foreach (float v in a.AsSpan<float>()) Assert.True(float.IsFinite(v));
            Assert.Equal(a.AsSpan<float>().ToArray(), b.AsSpan<float>().ToArray());
        }
    }

    [Fact]
    public void SecondPassLatent_IsCausal_APrefixOfTheCodesGivesThePrefixOfTheRows()
    {
        // Row k only sees [start, c1..ck] and the fixed conditioning/text prefix, so truncating the codes must not
        // change the earlier rows — the property that makes feeding [start, c1..c(T-1)] equal to the reference's
        // [start, c1..cT, stop] with the last two outputs dropped.
        Random rng = new(7);
        using IndexTts2T2sDecoder decoder = LoadDecoder(rng);
        (Tensor speaker, Tensor emoVec) = Conditioning(decoder, rng);
        using (speaker) using (emoVec)
        {
            int[] full = [11, 22, 33, 44, 55];
            using Tensor whole = decoder.ComputeSecondPassLatent(_backend, speaker, emoVec, TextIds, full);
            using Tensor prefix = decoder.ComputeSecondPassLatent(_backend, speaker, emoVec, TextIds, full[..3]);

            float[] w = whole.AsSpan<float>().ToArray(), p = prefix.AsSpan<float>().ToArray();
            for (int i = 0; i < p.Length; i++) Assert.Equal(w[i], p[i], 4);
        }
    }

    [Fact]
    public void SecondPassLatent_DependsOnTheCodes()
    {
        Random rng = new(8);
        using IndexTts2T2sDecoder decoder = LoadDecoder(rng);
        (Tensor speaker, Tensor emoVec) = Conditioning(decoder, rng);
        using (speaker) using (emoVec)
        {
            using Tensor a = decoder.ComputeSecondPassLatent(_backend, speaker, emoVec, TextIds, [10, 20, 30, 40]);
            using Tensor b = decoder.ComputeSecondPassLatent(_backend, speaker, emoVec, TextIds, [10, 20, 31, 40]);
            float[] x = a.AsSpan<float>().ToArray(), y = b.AsSpan<float>().ToArray();
            int firstRowLen = GptLayerOut;
            for (int i = 0; i < 2 * firstRowLen; i++) Assert.Equal(x[i], y[i], 4);          // rows 0-1 precede the change
            Assert.Contains(Enumerable.Range(3 * firstRowLen, firstRowLen), i => Math.Abs(x[i] - y[i]) > 1e-6f); // row 3 sees it
        }
    }

    [Fact]
    public void SecondPassLatent_WithoutGptLayer_Throws()
    {
        Random rng = new(9);
        using IndexTts2T2sDecoder decoder = LoadDecoder(rng, withGptLayer: false);
        (Tensor speaker, Tensor emoVec) = Conditioning(decoder, rng);
        using (speaker) using (emoVec)
            Assert.Throws<InvalidOperationException>(() => decoder.ComputeSecondPassLatent(_backend, speaker, emoVec, TextIds, [1, 2]));
    }

    public void Dispose()
    {
        foreach (Tensor t in _owned) t.Dispose();
        _backend.Dispose();
    }
}
