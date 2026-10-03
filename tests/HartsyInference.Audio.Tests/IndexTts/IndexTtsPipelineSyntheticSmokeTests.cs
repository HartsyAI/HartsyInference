using HartsyInference.Audio.Models.IndexTts;
using HartsyInference.Audio.Models.LanguageModels.Gpt;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests.IndexTts;

/// <summary>IndexTTS-1.5's real internal components (Conformer+Perceiver speaker conditioning, GPT T2S decoder,
/// BigVGAN vocoder with its embedded ECAPA-TDNN) chained together exactly as <c>IndexTtsPipeline.Synthesize</c>
/// does, but with tiny random weights and hand-crafted token ids instead of <c>IndexTtsPipeline.LoadAsync</c> and
/// <c>IndexTtsTokenizer</c> (there is no tiny SentencePiece model file to test against): shapes, finiteness and
/// determinism only. Says NOTHING about parity with the real checkpoints — see
/// <see cref="HartsyInference.Audio.Tests.IndexTts.IndexTtsConfigValuesTests"/> for the <c>V1_5</c> preset's
/// value-equality lock against the real <c>config.yaml</c>, and <c>docs/Research/INDEX_TTS_ARCHITECTURE.md</c> for
/// the deferred real-weight parity check.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class IndexTtsPipelineSyntheticSmokeTests : IDisposable
{
    // GPT T2S decoder: tiny but internally consistent (Hidden divisible by NumHeads).
    private const int Hidden = 32;
    private const int GptHeads = 4;
    private const int GptLayers = 2;
    private const int BlockSize = 64;
    private const int TextVocab = 16;
    private const int PosTableRows = 16;
    private const int MelCap = 6;

    // Conditioning Conformer. InputSize must stay 100: IndexTtsEcapaTdnn hardcodes its own mel-band count at 100
    // (InputMels), and both it and the Conformer consume the SAME reference-mel tensor in the real pipeline, so
    // this dimension cannot be shrunk independently of that fixed constant.
    private const int MelBands = 100;
    private const int ConformerOutputSize = 16;
    private const int ConformerHeads = 2;
    private const int ConformerLinearUnits = 32;
    private const int ConformerBlocks = 2;
    private const int ConformerConvKernel = 3;

    // Perceiver: NumLatents/DimHead/Heads (32/64/8) are compile-time constants on IndexTtsPerceiver itself, not
    // config-driven, so they are reproduced here rather than chosen.
    private const int PerceiverLatents = 32;
    private const int PerceiverDepth = 2;

    // BigVGAN. GptDim must equal Hidden (the T2S latent IS the vocoder's primary input); SpeakerEmbeddingDim must
    // be 512 because IndexTtsEcapaTdnn.LinNeurons is a hardcoded constant, not config-driven either.
    private const int BigVganInitialChannel = 16;
    private const int SpeakerEmbeddingDim = 512;
    private const int RefMelFrames = 16;

    private const string ConformerPrefix = "model.conditioning_encoder";
    private const string PerceiverPrefix = "model.perceiver_encoder";
    private const string BigVganPrefix = "generator";
    private const string SpeakerEncoderPrefix = "generator.speaker_encoder";

    private static readonly int[] TextIds = [2, 3, 4, 5];

    private static readonly IndexTtsConformerConfig ConformerCfg = new()
    {
        InputSize = MelBands,
        OutputSize = ConformerOutputSize,
        AttentionHeads = ConformerHeads,
        LinearUnits = ConformerLinearUnits,
        NumBlocks = ConformerBlocks,
        ConvKernel = ConformerConvKernel,
    };

    private static readonly GptConfig T2sCfg = new()
    {
        Hidden = Hidden,
        NumLayers = GptLayers,
        NumHeads = GptHeads,
        BlockSize = BlockSize,
        Bias = true, // IndexTTS's real gpt.pth is a standard biased HF GPT-2 export.
    };

    private static readonly IndexTtsBigVganConfig BigVganCfg = new()
    {
        GptDim = Hidden,
        SpeakerEmbeddingDim = SpeakerEmbeddingDim,
        UpsampleInitialChannel = BigVganInitialChannel,
        UpsampleRates = [2, 2],
        UpsampleKernelSizes = [4, 4],
        ResblockKernelSizes = [3],
        ResblockDilations = [[1]],
    };

    private readonly List<Tensor> _owned = [];

    private Tensor Own(Tensor t)
    {
        _owned.Add(t);
        return t;
    }

    private Tensor Rand(Random rng, double amp, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = (float)((rng.NextDouble() * 2 - 1) * amp);
        return Own(t);
    }

    private Tensor Const(float value, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        t.AsSpan<float>().Fill(value);
        return Own(t);
    }

    /// <summary>A LayerNorm/BatchNorm gain near 1 (not exactly, so a transposed or mis-shaped weight elsewhere
    /// would not be masked by every norm silently acting as identity).</summary>
    private Tensor NormWeight(Random rng, long dim)
    {
        Tensor t = Rand(rng, 0.1, dim);
        foreach (ref float v in t.AsSpan<float>()) v += 1f;
        return t;
    }

    private static Dictionary<string, Tensor> Merge(params Dictionary<string, Tensor>[] dicts)
    {
        Dictionary<string, Tensor> merged = [];
        foreach (Dictionary<string, Tensor> d in dicts) foreach ((string k, Tensor v) in d) merged[k] = v;
        return merged;
    }

    // ── Conformer ───────────────────────────────────────────────────────────────────────────────────────────

    private Dictionary<string, Tensor> BuildConformerWeights(IndexTtsConformerConfig cfg, Random rng, string prefix)
    {
        Dictionary<string, Tensor> w = [];
        int odim = cfg.OutputSize, headDim = odim / cfg.AttentionHeads;
        int fOut = (cfg.InputSize - 3) / 2 + 1; // Conv2dSubsampling2's own valid-padding formula.

        w[$"{prefix}.embed.conv.0.weight"] = Rand(rng, 0.1, odim, 1, 3, 3);
        w[$"{prefix}.embed.conv.0.bias"] = Rand(rng, 0.02, odim);
        w[$"{prefix}.embed.out.0.weight"] = Rand(rng, 0.1, odim, odim * fOut);
        w[$"{prefix}.embed.out.0.bias"] = Rand(rng, 0.02, odim);

        for (int i = 0; i < cfg.NumBlocks; i++)
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
            w[$"{p}.self_attn.pos_bias_u"] = Rand(rng, 0.1, cfg.AttentionHeads, headDim);
            w[$"{p}.self_attn.pos_bias_v"] = Rand(rng, 0.1, cfg.AttentionHeads, headDim);

            w[$"{p}.norm_conv.weight"] = NormWeight(rng, odim);
            w[$"{p}.norm_conv.bias"] = Rand(rng, 0.02, odim);
            w[$"{p}.conv_module.pointwise_conv1.weight"] = Rand(rng, 0.1, 2 * odim, odim, 1);
            w[$"{p}.conv_module.pointwise_conv1.bias"] = Rand(rng, 0.02, 2 * odim);
            w[$"{p}.conv_module.depthwise_conv.weight"] = Rand(rng, 0.1, odim, 1, cfg.ConvKernel);
            w[$"{p}.conv_module.depthwise_conv.bias"] = Rand(rng, 0.02, odim);
            w[$"{p}.conv_module.norm.weight"] = NormWeight(rng, odim);
            w[$"{p}.conv_module.norm.bias"] = Rand(rng, 0.02, odim);
            w[$"{p}.conv_module.pointwise_conv2.weight"] = Rand(rng, 0.1, odim, odim, 1);
            w[$"{p}.conv_module.pointwise_conv2.bias"] = Rand(rng, 0.02, odim);

            w[$"{p}.norm_ff.weight"] = NormWeight(rng, odim);
            w[$"{p}.norm_ff.bias"] = Rand(rng, 0.02, odim);
            w[$"{p}.feed_forward.w_1.weight"] = Rand(rng, 0.1, cfg.LinearUnits, odim);
            w[$"{p}.feed_forward.w_1.bias"] = Rand(rng, 0.02, cfg.LinearUnits);
            w[$"{p}.feed_forward.w_2.weight"] = Rand(rng, 0.1, odim, cfg.LinearUnits);
            w[$"{p}.feed_forward.w_2.bias"] = Rand(rng, 0.02, odim);

            w[$"{p}.norm_final.weight"] = NormWeight(rng, odim);
            w[$"{p}.norm_final.bias"] = Rand(rng, 0.02, odim);
        }

        w[$"{prefix}.after_norm.weight"] = NormWeight(rng, odim);
        w[$"{prefix}.after_norm.bias"] = Rand(rng, 0.02, odim);
        return w;
    }

    // ── Perceiver ───────────────────────────────────────────────────────────────────────────────────────────

    private Dictionary<string, Tensor> BuildPerceiverWeights(int dim, int dimContext, int depth, Random rng, string prefix)
    {
        const int dimHead = 64, heads = 8; // IndexTtsPerceiver's own fixed constants, not config-driven.
        int dimInner = dimHead * heads;
        int doubled = 2 * dim; // GEGLU feed-forward inner width; only its own internal consistency matters.

        Dictionary<string, Tensor> w = [];
        w[$"{prefix}.latents"] = Rand(rng, 0.1, PerceiverLatents, dim);
        w[$"{prefix}.proj_context.weight"] = Rand(rng, 0.1, dim, dimContext);
        w[$"{prefix}.proj_context.bias"] = Rand(rng, 0.02, dim);
        for (int i = 0; i < depth; i++)
        {
            w[$"{prefix}.layers.{i}.0.to_q.weight"] = Rand(rng, 0.1, dimInner, dim);
            w[$"{prefix}.layers.{i}.0.to_kv.weight"] = Rand(rng, 0.1, 2 * dimInner, dim);
            w[$"{prefix}.layers.{i}.0.to_out.weight"] = Rand(rng, 0.1, dim, dimInner);
            w[$"{prefix}.layers.{i}.1.0.weight"] = Rand(rng, 0.1, doubled, dim);
            w[$"{prefix}.layers.{i}.1.0.bias"] = Rand(rng, 0.02, doubled);
            w[$"{prefix}.layers.{i}.1.2.weight"] = Rand(rng, 0.1, dim, dim);
            w[$"{prefix}.layers.{i}.1.2.bias"] = Rand(rng, 0.02, dim);
        }
        w[$"{prefix}.norm.gamma"] = NormWeight(rng, dim);
        return w;
    }

    // ── GPT T2S decoder (raw model.* checkpoint keys, HF Conv1D [in,out] layout) ───────────────────────────────

    private Dictionary<string, Tensor> BuildGptWeights(GptConfig cfg, int textVocab, int textPosRows, int melPosRows, Random rng)
    {
        Dictionary<string, Tensor> w = [];
        w["model.text_embedding.weight"] = Rand(rng, 0.1, textVocab, cfg.Hidden);
        w["model.text_pos_embedding.emb.weight"] = Rand(rng, 0.1, textPosRows, cfg.Hidden);
        // mel_embedding/mel_head MUST span the full NumMelCodes: StartMelToken/StopMelToken/NumMelCodes are
        // compile-time constants on IndexTtsT2sDecoder, not config-driven, and any sampled code can land anywhere
        // in [0, NumMelCodes).
        w["model.mel_embedding.weight"] = Rand(rng, 0.1, IndexTtsT2sDecoder.NumMelCodes, cfg.Hidden);
        w["model.mel_pos_embedding.emb.weight"] = Rand(rng, 0.1, melPosRows, cfg.Hidden);
        w["model.final_norm.weight"] = NormWeight(rng, cfg.Hidden);
        w["model.final_norm.bias"] = Rand(rng, 0.02, cfg.Hidden);
        w["model.mel_head.weight"] = Rand(rng, 0.1, IndexTtsT2sDecoder.NumMelCodes, cfg.Hidden);
        w["model.mel_head.bias"] = Rand(rng, 0.02, IndexTtsT2sDecoder.NumMelCodes);

        for (int i = 0; i < cfg.NumLayers; i++)
        {
            string p = $"model.gpt.h.{i}";
            w[$"{p}.ln_1.weight"] = NormWeight(rng, cfg.Hidden);
            w[$"{p}.ln_1.bias"] = Rand(rng, 0.02, cfg.Hidden);
            w[$"{p}.attn.c_attn.weight"] = Rand(rng, 0.1, cfg.Hidden, 3 * cfg.Hidden); // [in, out]
            w[$"{p}.attn.c_attn.bias"] = Rand(rng, 0.02, 3 * cfg.Hidden);
            w[$"{p}.attn.c_proj.weight"] = Rand(rng, 0.1, cfg.Hidden, cfg.Hidden);
            w[$"{p}.attn.c_proj.bias"] = Rand(rng, 0.02, cfg.Hidden);
            w[$"{p}.ln_2.weight"] = NormWeight(rng, cfg.Hidden);
            w[$"{p}.ln_2.bias"] = Rand(rng, 0.02, cfg.Hidden);
            w[$"{p}.mlp.c_fc.weight"] = Rand(rng, 0.1, cfg.Hidden, cfg.MlpDim);
            w[$"{p}.mlp.c_fc.bias"] = Rand(rng, 0.02, cfg.MlpDim);
            w[$"{p}.mlp.c_proj.weight"] = Rand(rng, 0.1, cfg.MlpDim, cfg.Hidden);
            w[$"{p}.mlp.c_proj.bias"] = Rand(rng, 0.02, cfg.Hidden);
        }
        w["model.gpt.ln_f.weight"] = NormWeight(rng, cfg.Hidden);
        w["model.gpt.ln_f.bias"] = Rand(rng, 0.02, cfg.Hidden);
        return w;
    }

    // ── ECAPA-TDNN (embedded inside BigVGAN) — topology is entirely fixed constants on IndexTtsEcapaTdnn itself,
    //    not config-driven, so the shapes below mirror that class's hardcoded Channels/KernelSizes/Dilations/
    //    Res2NetScale/SeChannels/AttentionChannels/InputMels/LinNeurons exactly rather than choosing "tiny" ones. ──

    private void AddBatchNorm(Dictionary<string, Tensor> w, string prefix, Random rng, int channels)
    {
        w[$"{prefix}.norm.weight"] = NormWeight(rng, channels);
        w[$"{prefix}.norm.bias"] = Rand(rng, 0.02, channels);
        w[$"{prefix}.norm.running_mean"] = Rand(rng, 0.02, channels);
        w[$"{prefix}.norm.running_var"] = Const(1f, channels); // must stay positive (1/sqrt(var+eps)).
    }

    private void AddTdnn(Dictionary<string, Tensor> w, string prefix, Random rng, int inCh, int outCh, int kernel)
    {
        w[$"{prefix}.conv.conv.weight"] = Rand(rng, 0.1, outCh, inCh, kernel);
        w[$"{prefix}.conv.conv.bias"] = Rand(rng, 0.02, outCh);
        AddBatchNorm(w, $"{prefix}.norm", rng, outCh);
    }

    private void AddRes2Net(Dictionary<string, Tensor> w, string prefix, Random rng, int channels, int scale, int kernel)
    {
        int groupCh = channels / scale;
        for (int i = 0; i < scale - 1; i++) AddTdnn(w, $"{prefix}.blocks.{i}", rng, groupCh, groupCh, kernel);
    }

    private void AddSeBlock(Dictionary<string, Tensor> w, string prefix, Random rng, int inCh, int seCh)
    {
        w[$"{prefix}.conv1.conv.weight"] = Rand(rng, 0.1, seCh, inCh);
        w[$"{prefix}.conv1.conv.bias"] = Rand(rng, 0.02, seCh);
        w[$"{prefix}.conv2.conv.weight"] = Rand(rng, 0.1, inCh, seCh);
        w[$"{prefix}.conv2.conv.bias"] = Rand(rng, 0.02, inCh);
    }

    private void AddSeRes2Net(Dictionary<string, Tensor> w, string prefix, Random rng, int inCh, int outCh, int kernel, int seChannels)
    {
        AddTdnn(w, $"{prefix}.tdnn1", rng, inCh, outCh, 1);
        AddRes2Net(w, $"{prefix}.res2net_block", rng, outCh, 8, kernel);
        AddTdnn(w, $"{prefix}.tdnn2", rng, outCh, outCh, 1);
        AddSeBlock(w, $"{prefix}.se_block", rng, outCh, seChannels);
    }

    private void AddEcapaWeights(Dictionary<string, Tensor> w, string prefix, Random rng)
    {
        const int c0 = 512, c1 = 512, c2 = 512, c3 = 512, c4 = 1536;
        const int inputMels = 100, seChannels = 128, attnChannels = 128, linNeurons = 512;

        AddTdnn(w, $"{prefix}.blocks.0", rng, inputMels, c0, 5);
        AddSeRes2Net(w, $"{prefix}.blocks.1", rng, c0, c1, 3, seChannels);
        AddSeRes2Net(w, $"{prefix}.blocks.2", rng, c1, c2, 3, seChannels);
        AddSeRes2Net(w, $"{prefix}.blocks.3", rng, c2, c3, 3, seChannels);
        AddTdnn(w, $"{prefix}.mfa", rng, c0 * 3, c4, 1);

        AddTdnn(w, $"{prefix}.asp.tdnn", rng, c4 * 3, attnChannels, 1);
        w[$"{prefix}.asp.conv.conv.weight"] = Rand(rng, 0.1, c4, attnChannels);
        w[$"{prefix}.asp.conv.conv.bias"] = Rand(rng, 0.02, c4);
        AddBatchNorm(w, $"{prefix}.asp_bn", rng, c4 * 2);

        w[$"{prefix}.fc.conv.weight"] = Rand(rng, 0.1, linNeurons, c4 * 2);
        w[$"{prefix}.fc.conv.bias"] = Rand(rng, 0.02, linNeurons);
    }

    // ── BigVGAN ─────────────────────────────────────────────────────────────────────────────────────────────

    private void AddSnakeWeights(Dictionary<string, Tensor> w, string prefix, Random rng, int channels)
    {
        // alpha/beta are stored log-scale and exponentiated at load; a small amplitude keeps exp(·) near 1.
        w[$"{prefix}.act.alpha"] = Rand(rng, 0.05, channels);
        w[$"{prefix}.act.beta"] = Rand(rng, 0.05, channels);
        w[$"{prefix}.upsample.filter"] = Rand(rng, 0.3, 12); // AntiAliasedSnake's default kernel.
        w[$"{prefix}.downsample.lowpass.filter"] = Rand(rng, 0.3, 12);
    }

    private void AddBigVganResblock(Dictionary<string, Tensor> w, string prefix, Random rng, int channels, int kernel, int[] dilations)
    {
        for (int i = 0; i < dilations.Length; i++)
        {
            // Plain ".weight" keys (not weight_g/weight_v) so WeightNormFusion.Compose takes its already-fused
            // fallback branch and returns our own tensor by reference instead of allocating a fresh fused one.
            w[$"{prefix}.convs1.{i}.weight"] = Rand(rng, 0.1, channels, channels, kernel);
            w[$"{prefix}.convs1.{i}.bias"] = Rand(rng, 0.02, channels);
            w[$"{prefix}.convs2.{i}.weight"] = Rand(rng, 0.1, channels, channels, kernel);
            w[$"{prefix}.convs2.{i}.bias"] = Rand(rng, 0.02, channels);
        }
        for (int i = 0; i < 2 * dilations.Length; i++) AddSnakeWeights(w, $"{prefix}.activations.{i}", rng, channels);
    }

    private Dictionary<string, Tensor> BuildBigVganWeights(IndexTtsBigVganConfig cfg, Random rng, string prefix, string speakerEncoderPrefix)
    {
        Dictionary<string, Tensor> w = [];
        w[$"{prefix}.conv_pre.weight"] = Rand(rng, 0.1, cfg.UpsampleInitialChannel, cfg.GptDim, 7);
        w[$"{prefix}.conv_pre.bias"] = Rand(rng, 0.02, cfg.UpsampleInitialChannel);
        w[$"{prefix}.cond_layer.weight"] = Rand(rng, 0.1, cfg.UpsampleInitialChannel, cfg.SpeakerEmbeddingDim);
        w[$"{prefix}.cond_layer.bias"] = Rand(rng, 0.02, cfg.UpsampleInitialChannel);

        int numStages = cfg.UpsampleRates.Length;
        for (int i = 0; i < numStages; i++)
        {
            int inCh = cfg.UpsampleInitialChannel / (1 << i);
            int outCh = cfg.UpsampleInitialChannel / (1 << (i + 1));
            w[$"{prefix}.ups.{i}.0.weight"] = Rand(rng, 0.1, inCh, outCh, cfg.UpsampleKernelSizes[i]);
            w[$"{prefix}.ups.{i}.0.bias"] = Rand(rng, 0.02, outCh);
            w[$"{prefix}.conds.{i}.weight"] = Rand(rng, 0.1, outCh, cfg.SpeakerEmbeddingDim);
            w[$"{prefix}.conds.{i}.bias"] = Rand(rng, 0.02, outCh);
            for (int j = 0; j < cfg.ResblockKernelSizes.Length; j++)
            {
                int idx = i * cfg.ResblockKernelSizes.Length + j;
                AddBigVganResblock(w, $"{prefix}.resblocks.{idx}", rng, outCh, cfg.ResblockKernelSizes[j], cfg.ResblockDilations[j]);
            }
        }

        int finalCh = cfg.UpsampleInitialChannel / (1 << numStages);
        AddSnakeWeights(w, $"{prefix}.activation_post", rng, finalCh);
        w[$"{prefix}.conv_post.weight"] = Rand(rng, 0.1, 1, finalCh, 7);
        w[$"{prefix}.conv_post.bias"] = Rand(rng, 0.02, 1);

        AddEcapaWeights(w, speakerEncoderPrefix, rng);
        return w;
    }

    // ── Assembly ────────────────────────────────────────────────────────────────────────────────────────────

    private sealed record SpeakerAndT2s(IndexTtsSpeakerEncoder Speaker, IndexTtsT2sDecoder T2s) : IDisposable
    {
        public void Dispose()
        {
            Speaker.Dispose();
            T2s.Dispose();
        }
    }

    private SpeakerAndT2s BuildSpeakerAndT2s()
    {
        Dictionary<string, Tensor> w = Merge(
            BuildConformerWeights(ConformerCfg, new Random(11), ConformerPrefix),
            BuildPerceiverWeights(Hidden, ConformerOutputSize, PerceiverDepth, new Random(12), PerceiverPrefix));
        IndexTtsSpeakerEncoder speaker = new(ConformerCfg, Hidden);
        speaker.LoadWeights(w, ConformerPrefix, PerceiverPrefix);

        Dictionary<string, Tensor> gptW = BuildGptWeights(T2sCfg, TextVocab, PosTableRows, PosTableRows, new Random(13));
        IndexTtsT2sDecoder t2s = new(T2sCfg, maxMelTokens: 50);
        t2s.LoadWeights(gptW);

        return new SpeakerAndT2s(speaker, t2s);
    }

    private IndexTtsBigVganGenerator BuildVocoder()
    {
        Dictionary<string, Tensor> w = BuildBigVganWeights(BigVganCfg, new Random(14), BigVganPrefix, SpeakerEncoderPrefix);
        IndexTtsBigVganGenerator vocoder = new(BigVganCfg);
        vocoder.LoadWeights(w, BigVganPrefix, SpeakerEncoderPrefix);
        return vocoder;
    }

    [Fact]
    public void SpeakerEncoderThenT2sThenBigVgan_ProducesAFiniteBoundedWaveformOfTheExpectedLength()
    {
        CpuBackend backend = new();
        using SpeakerAndT2s parts = BuildSpeakerAndT2s();
        using IndexTtsBigVganGenerator vocoder = BuildVocoder();

        Tensor refMel = Own(Rand(new Random(21), 1.0, 1, RefMelFrames, MelBands));
        Tensor prefix = Own(parts.Speaker.Forward(backend, refMel, RefMelFrames));
        Assert.Equal(1, (int)prefix.Shape[0]);
        Assert.Equal(PerceiverLatents, (int)prefix.Shape[1]);
        Assert.Equal(Hidden, (int)prefix.Shape[2]);

        IndexTtsOptions opts = new() { Seed = 1, MaxMelTokens = MelCap };
        Tensor latent = Own(parts.T2s.Generate(backend, prefix, TextIds, opts, new Random(1)));
        int latentLen = (int)latent.Shape[1];
        Assert.InRange(latentLen, 1, MelCap + 1); // N generated codes + the StartMelToken's own latent frame.

        Tensor wave = Own(vocoder.Forward(backend, latent, latentLen, refMel, RefMelFrames));
        float[] pcm = wave.AsSpan<float>().ToArray();
        Assert.Equal(latentLen * 4, pcm.Length); // hop = Π UpsampleRates = 2*2.
        Assert.NotEmpty(pcm);
        foreach (float v in pcm) Assert.True(float.IsFinite(v) && MathF.Abs(v) <= 1f, "sample outside tanh's [-1,1] range");
    }

    [Fact]
    public void Generate_IsDeterministicForTheSameSeed_AndDiffersForADifferentSeed()
    {
        CpuBackend backend = new();
        using SpeakerAndT2s parts = BuildSpeakerAndT2s();

        Tensor refMel = Own(Rand(new Random(22), 1.0, 1, RefMelFrames, MelBands));
        Tensor prefix = Own(parts.Speaker.Forward(backend, refMel, RefMelFrames));
        IndexTtsOptions opts = new() { Seed = 1, MaxMelTokens = MelCap };

        Tensor a = Own(parts.T2s.Generate(backend, prefix, TextIds, opts, new Random(123)));
        Tensor b = Own(parts.T2s.Generate(backend, prefix, TextIds, opts, new Random(123)));
        Assert.Equal(a.AsSpan<float>().ToArray(), b.AsSpan<float>().ToArray());

        Tensor c = Own(parts.T2s.Generate(backend, prefix, TextIds, opts, new Random(999)));
        Assert.NotEqual(a.AsSpan<float>().ToArray(), c.AsSpan<float>().ToArray());
    }

    public void Dispose()
    {
        foreach (Tensor t in _owned) t.Dispose();
    }
}
