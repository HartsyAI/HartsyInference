using HartsyInference.Audio.Models.Chatterbox;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Checkpoint-free tests for Chatterbox: the T3 preset, a synthetic-weights forward through the
/// LSTM voice encoder (finite + L2-normalized), and T3 speech-token generation (valid in-range tokens) —
/// exercising the net-new T3 LM end-to-end. S3Gen is the reused CosyVoice stack.</summary>
public sealed unsafe class ChatterboxTests
{
    private static uint _rng = 0x1234abcdu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }

    [Fact]
    public void Default_T3IsLlama520M()
    {
        ChatterboxConfig c = ChatterboxConfig.Default;
        Assert.Equal(1_024, c.T3.HiddenSize);
        Assert.Equal(30, c.T3.NumHiddenLayers);
        Assert.Equal(16, c.T3.NumAttentionHeads);
        Assert.Equal(c.T3.NumAttentionHeads, c.T3.NumKeyValueHeads);  // full MHA
        Assert.Equal(64, c.T3.HeadDim);
        Assert.Equal(500_000f, c.T3.RopeTheta);
        Assert.False(c.T3.AttentionBias);
        Assert.Equal(6_561, c.S3CodebookSize);        // 3^8 FSQ
        Assert.Equal(6_561, c.StartSpeechToken);
        Assert.Equal(6_562, c.StopSpeechToken);
        Assert.Equal(25, c.SpeechTokenRate);
    }

    [Fact]
    public void VoiceEncoder_SyntheticForward_IsFiniteAndL2Normalized()
    {
        using CpuBackend backend = new();
        using ChatterboxVoiceEncoder ve = new();
        ve.LoadWeights(VoiceEncoderWeights());
        int t = 12;
        using Tensor mel = F(t, 40).Reshape(new TensorShape(1, t, 40));
        using Tensor emb = ve.Forward(backend, mel, t);
        Assert.Equal(256, (int)emb.Shape[1]);
        float* p = (float*)emb.DataPointer;
        double norm = 0;
        for (int i = 0; i < 256; i++) { Assert.True(float.IsFinite(p[i])); norm += (double)p[i] * p[i]; }
        Assert.True(Math.Abs(Math.Sqrt(norm) - 1.0) < 1e-3);   // L2-normalized
    }

    [Fact]
    public void T3_SyntheticForward_GeneratesValidSpeechTokens()
    {
        ChatterboxConfig c = ChatterboxConfig.Default with
        {
            T3 = ChatterboxConfig.Default.T3 with
            {
                HiddenSize = 32, NumHiddenLayers = 2, NumAttentionHeads = 4, NumKeyValueHeads = 4,
                IntermediateSize = 64, VocabSize = 16, MaxPositionEmbeddings = 256,
            },
            TextVocab = 10, SpeechVocab = 16, SpeakerEmbedDim = 12,
            StartSpeechToken = 14, StopSpeechToken = 15, MaxTextTokens = 32, MaxSpeechTokens = 64,
        };
        using CpuBackend backend = new();
        using ChatterboxT3 t3 = new(c);
        t3.LoadWeights(T3Weights(c));

        using Tensor speaker = F1(c.SpeakerEmbedDim);
        int[] text = [2, 5, 1, 7];
        List<int> tokens = t3.GenerateSpeechTokens(backend, text, speaker, c.Exaggeration, maxNew: 6, seed: 7);
        Assert.True(tokens.Count <= 6);
        foreach (int tok in tokens)
        {
            Assert.InRange(tok, 0, c.SpeechVocab - 1);
            Assert.NotEqual(c.StopSpeechToken, tok);
        }
    }

    private static Dictionary<string, Tensor> VoiceEncoderWeights()
    {
        Dictionary<string, Tensor> w = new() { ["proj.weight"] = F(256, 256), ["proj.bias"] = F1(256) };
        for (int i = 0; i < 3; i++)
        {
            int inDim = i == 0 ? 40 : 256;
            w[$"lstm.weight_ih_l{i}"] = F(4 * 256, inDim);
            w[$"lstm.weight_hh_l{i}"] = F(4 * 256, 256);
            w[$"lstm.bias_ih_l{i}"] = F1(4 * 256);
            w[$"lstm.bias_hh_l{i}"] = F1(4 * 256);
        }
        return w;
    }

    private static Dictionary<string, Tensor> T3Weights(ChatterboxConfig c)
    {
        int h = c.T3.HiddenSize, qd = c.T3.NumAttentionHeads * c.T3.HeadDim, kvd = c.T3.NumKeyValueHeads * c.T3.HeadDim;
        Dictionary<string, Tensor> w = new()
        {
            ["tfmr.norm.weight"] = Ones(h),
            ["text_emb.weight"] = F(c.TextVocab, h),
            ["speech_emb.weight"] = F(c.SpeechVocab, h),
            ["text_pos_emb.emb.weight"] = F(c.MaxTextTokens, h),
            ["speech_pos_emb.emb.weight"] = F(c.MaxSpeechTokens, h),
            ["speech_head.weight"] = F(c.SpeechVocab, h),
            ["cond_enc.spkr_enc.weight"] = F(h, c.SpeakerEmbedDim),
            ["cond_enc.emotion_adv_fc.weight"] = F(h, 1),
        };
        for (int i = 0; i < c.T3.NumHiddenLayers; i++)
        {
            string p = $"tfmr.layers.{i}";
            w[$"{p}.input_layernorm.weight"] = Ones(h);
            w[$"{p}.post_attention_layernorm.weight"] = Ones(h);
            w[$"{p}.self_attn.q_proj.weight"] = F(qd, h);
            w[$"{p}.self_attn.k_proj.weight"] = F(kvd, h);
            w[$"{p}.self_attn.v_proj.weight"] = F(kvd, h);
            w[$"{p}.self_attn.o_proj.weight"] = F(h, qd);
            w[$"{p}.mlp.gate_proj.weight"] = F(c.T3.IntermediateSize, h);
            w[$"{p}.mlp.up_proj.weight"] = F(c.T3.IntermediateSize, h);
            w[$"{p}.mlp.down_proj.weight"] = F(h, c.T3.IntermediateSize);
        }
        return w;
    }
}
