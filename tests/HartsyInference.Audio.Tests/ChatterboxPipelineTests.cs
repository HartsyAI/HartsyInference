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
        using CamPlusSpeakerEncoder spkEnc = new(camDim);
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
        // UpsampleConformerEncoder (CosyVoice 2 rel-pos Transformer encoder).
        AddLinear(w, "encoder.embed.out.0", outSize, inputSize, 20);     // embed Linear
        AddNorm(w, "encoder.embed.out.1", outSize, 25);                  // embed LayerNorm
        AddConv(w, "encoder.pre_lookahead_layer.conv1", outSize, outSize, 4, 30);
        AddConv(w, "encoder.pre_lookahead_layer.conv2", outSize, outSize, 3, 40);
        BuildRelPosBlocks(w, "encoder.encoders", c.EncoderNumPreBlocks, outSize, c.EncoderNumHeads, 100);
        AddConv(w, "encoder.up_layer.conv", outSize, outSize, 5, 700);   // Upsample1D conv (kernel = ratio*2+1)
        AddLinear(w, "encoder.up_embed.out.0", outSize, outSize, 710);   // up_embed Linear
        AddNorm(w, "encoder.up_embed.out.1", outSize, 715);              // up_embed LayerNorm
        BuildRelPosBlocks(w, "encoder.up_encoders", c.EncoderNumPostBlocks, outSize, c.EncoderNumHeads, 2_000);
        AddNorm(w, "encoder.after_norm", outSize, 9_000);
        // CFM estimator — CausalConditionalDecoder (Matcha U-Net: 1 down + N mid + 1 up).
        string e = "decoder.estimator";
        int inCh = mel * 4, timeDim = ch * 4, inner = c.NumHeads * c.AttentionHeadDim, nb = c.NumBlocks;
        AddLinear(w, $"{e}.time_mlp.linear_1", timeDim, inCh, 3_000);
        AddLinear(w, $"{e}.time_mlp.linear_2", timeDim, timeDim, 3_010);
        // down block.
        AddMatchaResnet(w, $"{e}.down_blocks.0.0", inCh, ch, timeDim, 3_100);
        for (int j = 0; j < nb; j++) AddMatchaTransformer(w, $"{e}.down_blocks.0.1.{j}", ch, inner, 3_200 + j * 40);
        AddConv(w, $"{e}.down_blocks.0.2", ch, ch, 3, 3_400);
        // mid blocks.
        for (int i = 0; i < c.NumMidBlocks; i++)
        {
            int s = 4_000 + i * 300;
            AddMatchaResnet(w, $"{e}.mid_blocks.{i}.0", ch, ch, timeDim, s);
            for (int j = 0; j < nb; j++) AddMatchaTransformer(w, $"{e}.mid_blocks.{i}.1.{j}", ch, inner, s + 100 + j * 40);
        }
        // up block (resnet input is 2·ch from skip concat).
        AddMatchaResnet(w, $"{e}.up_blocks.0.0", 2 * ch, ch, timeDim, 7_000);
        for (int j = 0; j < nb; j++) AddMatchaTransformer(w, $"{e}.up_blocks.0.1.{j}", ch, inner, 7_200 + j * 40);
        AddConv(w, $"{e}.up_blocks.0.2", ch, ch, 3, 7_400);
        // final.
        AddConv(w, $"{e}.final_block.block.0", ch, ch, 3, 8_000);
        AddNorm(w, $"{e}.final_block.block.2", ch, 8_010);
        AddConv(w, $"{e}.final_proj", mel, ch, 1, 8_020);
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

    private static void AddMatchaResnet(Dictionary<string, Tensor> w, string prefix, int inCh, int outCh, int timeDim, int seed)
    {
        AddConv(w, $"{prefix}.block1.block.0", outCh, inCh, 3, seed);          // CausalConv1d
        AddNorm(w, $"{prefix}.block1.block.2", outCh, seed + 2);               // LayerNorm
        AddConv(w, $"{prefix}.block2.block.0", outCh, outCh, 3, seed + 4);
        AddNorm(w, $"{prefix}.block2.block.2", outCh, seed + 6);
        AddLinear(w, $"{prefix}.mlp.1", outCh, timeDim, seed + 8);             // time-emb proj
        AddConv(w, $"{prefix}.res_conv", outCh, inCh, 1, seed + 10);          // 1×1 residual conv
    }

    private static void AddMatchaTransformer(Dictionary<string, Tensor> w, string prefix, int ch, int inner, int seed)
    {
        int ffInner = ch * 4;
        AddNorm(w, $"{prefix}.norm1", ch, seed);
        AddLinear(w, $"{prefix}.attn1.to_q", inner, ch, seed + 2, bias: false);
        AddLinear(w, $"{prefix}.attn1.to_k", inner, ch, seed + 4, bias: false);
        AddLinear(w, $"{prefix}.attn1.to_v", inner, ch, seed + 6, bias: false);
        AddLinear(w, $"{prefix}.attn1.to_out.0", ch, inner, seed + 8);         // has bias
        AddNorm(w, $"{prefix}.norm3", ch, seed + 10);
        AddLinear(w, $"{prefix}.ff.net.0.proj", ffInner, ch, seed + 12);       // GELU proj
        AddLinear(w, $"{prefix}.ff.net.2", ch, ffInner, seed + 14);
    }

    private static void BuildRelPosBlocks(Dictionary<string, Tensor> w, string stack, int blocks, int c, int numHeads, int seedBase)
    {
        const int ffDim = 2;
        int headDim = c / numHeads;
        for (int i = 0; i < blocks; i++)
        {
            string b = $"{stack}.{i}";
            int s = seedBase + i * 200;
            AddNorm(w, $"{b}.norm_mha", c, s);
            AddLinear(w, $"{b}.self_attn.linear_q", c, c, s + 2);
            AddLinear(w, $"{b}.self_attn.linear_k", c, c, s + 4);
            AddLinear(w, $"{b}.self_attn.linear_v", c, c, s + 6);
            AddLinear(w, $"{b}.self_attn.linear_out", c, c, s + 8);
            AddLinear(w, $"{b}.self_attn.linear_pos", c, c, s + 10, bias: false);
            w[$"{b}.self_attn.pos_bias_u"] = Rand(new TensorShape(numHeads, headDim), s + 12);
            w[$"{b}.self_attn.pos_bias_v"] = Rand(new TensorShape(numHeads, headDim), s + 13);
            AddNorm(w, $"{b}.norm_ff", c, s + 14);
            AddLinear(w, $"{b}.feed_forward.w_1", c * ffDim, c, s + 16);
            AddLinear(w, $"{b}.feed_forward.w_2", c, c * ffDim, s + 18);
        }
    }
}
