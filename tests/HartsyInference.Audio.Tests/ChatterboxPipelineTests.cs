using HartsyInference.Audio.Models.Chatterbox;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Checkpoint-free end-to-end test for <see cref="ChatterboxPipeline"/> on TINY configs: text
/// tokens + a random voice-encoder embedding flow through the net-new T3 LM → the reused CosyVoice S3Gen
/// flow → the HiFTNet vocoder, producing finite 24 kHz audio. Synthetic xorshift-filled weights on the
/// CPU backend — no checkpoint needed (exact-weights parity is the env-gated generation path).</summary>
public sealed unsafe class ChatterboxPipelineTests
{
    private static Tensor Rand(TensorShape shape, int seed, float scale = 0.05f)
    {
        Tensor t = new(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        uint s = (uint)(seed * 2654435761u + 1u);
        for (long i = 0; i < t.ElementCount; i++)
        {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            p[i] = ((s & 0xFFFFFF) / (float)0xFFFFFF - 0.5f) * 2f * scale;
        }
        return t;
    }

    private static Tensor Ones(int n)
    {
        Tensor t = new(new TensorShape(n), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < n; i++) p[i] = 1f;
        return t;
    }

    [Fact]
    public void Synthesize_SyntheticWeights_ProducesFinite24kHzAudio()
    {
        using CpuBackend backend = new();

        const int hidden = 32, mel = 8, flowDim = 24, speakerDim = 12, camDim = 16;
        ChatterboxConfig cfg = ChatterboxConfig.Default with
        {
            T3 = ChatterboxConfig.Default.T3 with
            {
                HiddenSize = hidden, NumHiddenLayers = 2, NumAttentionHeads = 4, NumKeyValueHeads = 4,
                IntermediateSize = 64, VocabSize = 24, MaxPositionEmbeddings = 256,
            },
            TextVocab = 10, SpeechVocab = 24, SpeakerEmbedDim = speakerDim,
            StartSpeechToken = 20, StopSpeechToken = 23, MaxTextTokens = 32, MaxSpeechTokens = 64,
        };

        CosyVoiceFlowConfig flowCfg = new()
        {
            MelBins = mel, InputSize = flowDim, EncoderOutputSize = flowDim, EncoderNumHeads = 8,
            EncoderNumPreBlocks = 1, EncoderNumPostBlocks = 1, UnetChannels = [16, 16],
            AttentionHeadDim = 4, NumHeads = 4, NumBlocks = 1, NumMidBlocks = 2,
            SpeakerEmbedDim = camDim, NumEulerSteps = 2, CfgRate = 0.7f,
        };
        CosyVoiceHiftConfig hiftCfg = new()
        {
            MelBins = mel, SampleRate = 24_000, UpsampleInitialChannel = 16,
            UpsampleRates = [4, 4], UpsampleKernelSizes = [8, 8],
            ResBlockKernelSizes = [3, 7], ResBlockDilationSizes = [[1, 3], [1, 3]],
            IstftNFft = 16, IstftHopSize = 4, HarmonicNum = 8,
        };

        CosyVoiceConfig cosyCfg = CosyVoiceConfig.V2_0_5B with { Flow = flowCfg, Hift = hiftCfg };
        using ChatterboxT3 t3 = new(cfg);
        using CosyVoiceFlow flow = new(cosyCfg);
        using HiFTNetVocoder vocoder = new(hiftCfg);
        using CamPlusSpeakerEncoder spkEnc = new(mel, camDim);
        using ChatterboxPipeline pipe = new(cfg, t3, flow, vocoder, spkEnc);

        t3.LoadWeights(T3Weights(cfg));
        flow.LoadWeights(FlowWeights(flowCfg));
        vocoder.LoadWeights(HiftWeights(hiftCfg));

        using Tensor refSpk = Rand(new TensorShape(cfg.SpeakerEmbedDim), 1234, 0.3f);
        using Tensor flowSpk = Rand(new TensorShape(1, camDim), 5678, 0.3f);
        int[] text = [2, 5, 1, 7];

        int steps = 0;
        float[] audio = pipe.Synthesize(backend, text, refSpk, cfg.Exaggeration, seed: 7,
            flowSpeakerEmbed: flowSpk, progress: _ => steps++);

        Assert.True(steps >= 3, "progress callback should fire for each stage.");
        Assert.True(audio.Length > 0, "expected non-empty audio.");
        foreach (float s in audio) Assert.True(float.IsFinite(s), "audio samples must be finite.");
    }

    private static Dictionary<string, Tensor> T3Weights(ChatterboxConfig c)
    {
        int h = c.T3.HiddenSize, qd = c.T3.NumAttentionHeads * c.T3.HeadDim, kvd = c.T3.NumKeyValueHeads * c.T3.HeadDim;
        Dictionary<string, Tensor> w = new()
        {
            ["tfmr.norm.weight"] = Ones(h),
            ["text_emb.weight"] = Rand(new TensorShape(c.TextVocab, h), 10),
            ["speech_emb.weight"] = Rand(new TensorShape(c.SpeechVocab, h), 11),
            ["text_pos_emb.emb.weight"] = Rand(new TensorShape(c.MaxTextTokens, h), 12),
            ["speech_pos_emb.emb.weight"] = Rand(new TensorShape(c.MaxSpeechTokens, h), 13),
            ["speech_head.weight"] = Rand(new TensorShape(c.SpeechVocab, h), 14),
            ["cond_enc.spkr_enc.weight"] = Rand(new TensorShape(h, c.SpeakerEmbedDim), 15),
            ["cond_enc.emotion_adv_fc.weight"] = Rand(new TensorShape(h, 1), 16),
        };
        for (int i = 0; i < c.T3.NumHiddenLayers; i++)
        {
            string p = $"tfmr.layers.{i}";
            int s = 100 + i * 50;
            w[$"{p}.input_layernorm.weight"] = Ones(h);
            w[$"{p}.post_attention_layernorm.weight"] = Ones(h);
            w[$"{p}.self_attn.q_proj.weight"] = Rand(new TensorShape(qd, h), s);
            w[$"{p}.self_attn.k_proj.weight"] = Rand(new TensorShape(kvd, h), s + 1);
            w[$"{p}.self_attn.v_proj.weight"] = Rand(new TensorShape(kvd, h), s + 2);
            w[$"{p}.self_attn.o_proj.weight"] = Rand(new TensorShape(h, qd), s + 3);
            w[$"{p}.mlp.gate_proj.weight"] = Rand(new TensorShape(c.T3.IntermediateSize, h), s + 4);
            w[$"{p}.mlp.up_proj.weight"] = Rand(new TensorShape(c.T3.IntermediateSize, h), s + 5);
            w[$"{p}.mlp.down_proj.weight"] = Rand(new TensorShape(h, c.T3.IntermediateSize), s + 6);
        }
        return w;
    }

    private static Dictionary<string, Tensor> FlowWeights(CosyVoiceFlowConfig c)
    {
        int inputSize = c.InputSize, outSize = c.EncoderOutputSize, mel = c.MelBins, ch = c.UnetChannels[0];
        Dictionary<string, Tensor> w = new()
        {
            ["input_embedding.weight"] = Rand(new TensorShape(6_561 + 3, inputSize), 1),
            ["encoder_proj.weight"] = Rand(new TensorShape(mel, outSize), 2),
            ["encoder_proj.bias"] = Rand(new TensorShape(mel), 3),
            ["spk_embed_affine_layer.weight"] = Rand(new TensorShape(mel, c.SpeakerEmbedDim), 4),
            ["spk_embed_affine_layer.bias"] = Rand(new TensorShape(mel), 5),
        };
        // UpsampleConformerEncoder.
        AddLinear(w, "encoder.embed.out.0", outSize, inputSize, 20);
        BuildConformer(w, "encoder.encoders", c.EncoderNumPreBlocks, outSize, 100);
        w["encoder.up_layer.weight"] = Rand(new TensorShape(outSize, outSize, 4), 700);
        w["encoder.up_layer.bias"] = Rand(new TensorShape(outSize), 701);
        BuildConformer(w, "encoder.up_encoders", c.EncoderNumPostBlocks, outSize, 2_000);
        AddNorm(w, "encoder.after_norm", outSize, 9_000);
        // CFM estimator (CausalConditionalDecoder).
        string e = "decoder.estimator";
        AddLinear(w, $"{e}.time_mlp.1", ch, ch, 3_000);
        AddLinear(w, $"{e}.time_mlp.3", ch, ch, 3_010);
        AddConv(w, $"{e}.in_conv", ch, mel * 4, 3, 3_020);
        for (int i = 0; i < c.NumMidBlocks; i++)
        {
            int s = 4_000 + i * 200;
            AddResnet(w, $"{e}.blocks.{i}.resnet", ch, s);
            AddAttn(w, $"{e}.blocks.{i}.attn", ch, c.NumHeads * c.AttentionHeadDim, s + 50);
        }
        AddNorm(w, $"{e}.out_norm", ch, 8_000);
        AddConv(w, $"{e}.out_conv", mel, ch, 3, 8_010);
        return w;
    }

    private static Dictionary<string, Tensor> HiftWeights(CosyVoiceHiftConfig c)
    {
        int numUp = c.UpsampleRates.Length;
        int initCh = c.UpsampleInitialChannel;
        Dictionary<string, Tensor> w = new();
        AddWeightNormConv(w, "conv_pre", initCh, c.MelBins, 7, 1);
        AddWeightNormConv(w, "conv_post", c.IstftNFft + 2, initCh >> numUp, 7, 100);
        w["m_source.l_linear.weight"] = Rand(new TensorShape(1, c.HarmonicNum + 1), 200);
        w["m_source.l_linear.bias"] = Rand(new TensorShape(1), 201);
        int[] srcK = numUp == 2 ? [7, 11] : [7, 7, 11];
        for (int i = 0; i < numUp; i++)
        {
            int levelCh = initCh >> (i + 1);
            int prevCh = initCh >> i;
            int kernel = c.UpsampleKernelSizes[i];
            AddWeightNormConvT(w, $"ups.{i}", prevCh, levelCh, kernel, 300 + i * 10);
            int srcStride = 1;
            for (int j = i + 1; j < numUp; j++) srcStride *= c.UpsampleRates[j];
            int srcK2 = srcStride == 1 ? 1 : srcStride * 2 + 1;
            w[$"source_downs.{i}.weight"] = Rand(new TensorShape(levelCh, c.IstftNFft + 2, srcK2), 400 + i * 10);
            w[$"source_downs.{i}.bias"] = Rand(new TensorShape(levelCh), 401 + i * 10);
            AddSnakeResBlock(w, $"source_resblocks.{i}", levelCh, srcK[i], [1, 3, 5], 500 + i * 30);
        }
        for (int i = 0; i < numUp; i++)
        {
            int levelCh = initCh >> (i + 1);
            for (int j = 0; j < c.ResBlockKernelSizes.Length; j++)
                AddSnakeResBlock(w, $"resblocks.{i * c.ResBlockKernelSizes.Length + j}", levelCh,
                    c.ResBlockKernelSizes[j], c.ResBlockDilationSizes[j], 1_000 + (i * 10 + j) * 40);
        }
        // F0Predictor: 5 condnet conv layers at even indices + classifier.
        for (int i = 0; i < 5; i++)
            AddWeightNormConv(w, $"f0_predictor.condnet.{i * 2}", 512, i == 0 ? c.MelBins : 512, 3, 5_000 + i * 10);
        w["f0_predictor.classifier.weight"] = Rand(new TensorShape(1, 512), 6_000);
        w["f0_predictor.classifier.bias"] = Rand(new TensorShape(1), 6_001);
        return w;
    }

    private static void AddNorm(Dictionary<string, Tensor> w, string prefix, int dim, int seed)
    {
        w[$"{prefix}.weight"] = Rand(new TensorShape(dim), seed, 0.1f);
        w[$"{prefix}.bias"] = Rand(new TensorShape(dim), seed + 1, 0.05f);
    }

    private static void AddLinear(Dictionary<string, Tensor> w, string prefix, int outDim, int inDim, int seed, bool bias = true)
    {
        w[$"{prefix}.weight"] = Rand(new TensorShape(outDim, inDim), seed);
        if (bias) w[$"{prefix}.bias"] = Rand(new TensorShape(outDim), seed + 1);
    }

    private static void AddConv(Dictionary<string, Tensor> w, string prefix, int outCh, int inCh, int kernel, int seed)
    {
        w[$"{prefix}.weight"] = Rand(new TensorShape(outCh, inCh, kernel), seed);
        w[$"{prefix}.bias"] = Rand(new TensorShape(outCh), seed + 1);
    }

    private static void AddWeightNormConv(Dictionary<string, Tensor> w, string prefix, int outCh, int inCh, int kernel, int seed)
    {
        w[$"{prefix}.weight_v"] = Rand(new TensorShape(outCh, inCh, kernel), seed);
        w[$"{prefix}.weight_g"] = Rand(new TensorShape(outCh, 1, 1), seed + 1, 0.5f);
        w[$"{prefix}.bias"] = Rand(new TensorShape(outCh), seed + 2);
    }

    private static void AddWeightNormConvT(Dictionary<string, Tensor> w, string prefix, int inCh, int outCh, int kernel, int seed)
    {
        // ConvTranspose1d weight is [Cin, Cout, K]; weight_norm dim-0 magnitude is [Cin, 1, 1].
        w[$"{prefix}.weight_v"] = Rand(new TensorShape(inCh, outCh, kernel), seed);
        w[$"{prefix}.weight_g"] = Rand(new TensorShape(inCh, 1, 1), seed + 1, 0.5f);
        w[$"{prefix}.bias"] = Rand(new TensorShape(outCh), seed + 2);
    }

    private static void AddSnakeResBlock(Dictionary<string, Tensor> w, string prefix, int ch, int kernel, int[] dilations, int seed)
    {
        for (int i = 0; i < dilations.Length; i++)
        {
            AddWeightNormConv(w, $"{prefix}.convs1.{i}", ch, ch, kernel, seed + i * 10);
            AddWeightNormConv(w, $"{prefix}.convs2.{i}", ch, ch, kernel, seed + i * 10 + 3);
            w[$"{prefix}.activations1.{i}.alpha"] = Ones(ch);
            w[$"{prefix}.activations2.{i}.alpha"] = Ones(ch);
        }
    }

    private static void AddResnet(Dictionary<string, Tensor> w, string prefix, int ch, int seed)
    {
        AddNorm(w, $"{prefix}.norm1", ch, seed);
        AddConv(w, $"{prefix}.conv1", ch, ch, 3, seed + 2);
        AddNorm(w, $"{prefix}.norm2", ch, seed + 4);
        AddConv(w, $"{prefix}.conv2", ch, ch, 3, seed + 6);
        AddLinear(w, $"{prefix}.time_emb", ch, ch, seed + 8);
    }

    private static void AddAttn(Dictionary<string, Tensor> w, string prefix, int ch, int attnDim, int seed)
    {
        AddNorm(w, $"{prefix}.norm1", ch, seed);
        AddLinear(w, $"{prefix}.to_q", attnDim, ch, seed + 2);
        AddLinear(w, $"{prefix}.to_k", attnDim, ch, seed + 4);
        AddLinear(w, $"{prefix}.to_v", attnDim, ch, seed + 6);
        AddLinear(w, $"{prefix}.to_out", ch, attnDim, seed + 8);
        AddNorm(w, $"{prefix}.norm2", ch, seed + 10);
        AddLinear(w, $"{prefix}.ff1", ch * 2, ch, seed + 12);
        AddLinear(w, $"{prefix}.ff2", ch, ch * 2, seed + 14);
    }

    private static void BuildConformer(Dictionary<string, Tensor> w, string stack, int blocks, int c, int seedBase)
    {
        const int ffDim = 2;
        for (int i = 0; i < blocks; i++)
        {
            string b = $"{stack}.{i}";
            int s = seedBase + i * 200;
            AddNorm(w, $"{b}.norm_ff_macaron", c, s);
            AddLinear(w, $"{b}.feed_forward_macaron.w_1", c * ffDim, c, s + 2);
            AddLinear(w, $"{b}.feed_forward_macaron.w_2", c, c * ffDim, s + 4);
            AddNorm(w, $"{b}.norm_mha", c, s + 6);
            AddLinear(w, $"{b}.self_attn.linear_q", c, c, s + 8);
            AddLinear(w, $"{b}.self_attn.linear_k", c, c, s + 10);
            AddLinear(w, $"{b}.self_attn.linear_v", c, c, s + 12);
            AddLinear(w, $"{b}.self_attn.linear_out", c, c, s + 14);
            AddNorm(w, $"{b}.norm_conv", c, s + 16);
            w[$"{b}.conv_module.pointwise_conv1.weight"] = Rand(new TensorShape(2 * c, c, 1), s + 18);
            w[$"{b}.conv_module.pointwise_conv1.bias"] = Rand(new TensorShape(2 * c), s + 19);
            w[$"{b}.conv_module.depthwise_conv.weight"] = Rand(new TensorShape(c, 1, 5), s + 20);
            w[$"{b}.conv_module.depthwise_conv.bias"] = Rand(new TensorShape(c), s + 21);
            AddNorm(w, $"{b}.conv_module.norm", c, s + 22);
            w[$"{b}.conv_module.pointwise_conv2.weight"] = Rand(new TensorShape(c, c, 1), s + 24);
            w[$"{b}.conv_module.pointwise_conv2.bias"] = Rand(new TensorShape(c), s + 25);
            AddNorm(w, $"{b}.norm_ff", c, s + 26);
            AddLinear(w, $"{b}.feed_forward.w_1", c * ffDim, c, s + 28);
            AddLinear(w, $"{b}.feed_forward.w_2", c, c * ffDim, s + 30);
            AddNorm(w, $"{b}.norm_final", c, s + 32);
        }
    }
}
