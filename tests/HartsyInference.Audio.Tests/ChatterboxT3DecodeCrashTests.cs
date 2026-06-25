using System.Collections.Generic;
using HartsyInference.Audio.Models.Chatterbox;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Lightweight reproduction of the T3 CPU single-token-decode AccessViolation: real T3 dims
/// (hidden 1024, intermediate 4096, 16 heads) but only a few layers with random weights (~250 MB, no
/// multi-GB checkpoint). Runs the prefill + autoregressive decode through <see cref="ChatterboxT3"/>. If the
/// crash is dimension-driven it reproduces here; if it needs 30 layers / real weights it won't.</summary>
public sealed unsafe class ChatterboxT3DecodeCrashTests
{
    private static Tensor Rand(TensorShape shape, int seed, float scale = 0.02f)
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

    [Theory]
    [InlineData(16)]
    [InlineData(30)]
    public void T3_PrefillThenDecode_RealDims(int layers)
    {
        using CpuBackend backend = new();
        ChatterboxConfig baseCfg = ChatterboxConfig.Default;
        ChatterboxConfig cfg = baseCfg with
        {
            T3 = baseCfg.T3 with { NumHiddenLayers = layers },
            MaxNewTokens = 4,
        };
        using ChatterboxT3 t3 = new(cfg);

        Dictionary<string, Tensor> w = new();
        int h = cfg.T3.HiddenSize, inter = cfg.T3.IntermediateSize;
        w["text_emb.weight"] = Rand(new TensorShape(cfg.TextVocab, h), 1);
        w["speech_emb.weight"] = Rand(new TensorShape(cfg.SpeechVocab, h), 2);
        w["text_pos_emb.emb.weight"] = Rand(new TensorShape(2048, h), 3);
        w["speech_pos_emb.emb.weight"] = Rand(new TensorShape(4096, h), 4);
        w["speech_head.weight"] = Rand(new TensorShape(cfg.SpeechVocab, h), 5);
        w["cond_enc.spkr_enc.weight"] = Rand(new TensorShape(h, cfg.SpeakerEmbedDim), 6);
        w["cond_enc.emotion_adv_fc.weight"] = Rand(new TensorShape(h, 1), 7);
        w["tfmr.norm.weight"] = Rand(new TensorShape(h), 8, 0.1f);
        for (int i = 0; i < layers; i++)
        {
            string p = $"tfmr.layers.{i}";
            int s = 100 + i * 50;
            w[$"{p}.input_layernorm.weight"] = Rand(new TensorShape(h), s, 0.1f);
            w[$"{p}.post_attention_layernorm.weight"] = Rand(new TensorShape(h), s + 1, 0.1f);
            w[$"{p}.self_attn.q_proj.weight"] = Rand(new TensorShape(h, h), s + 2);
            w[$"{p}.self_attn.k_proj.weight"] = Rand(new TensorShape(h, h), s + 3);
            w[$"{p}.self_attn.v_proj.weight"] = Rand(new TensorShape(h, h), s + 4);
            w[$"{p}.self_attn.o_proj.weight"] = Rand(new TensorShape(h, h), s + 5);
            w[$"{p}.mlp.gate_proj.weight"] = Rand(new TensorShape(inter, h), s + 6);
            w[$"{p}.mlp.up_proj.weight"] = Rand(new TensorShape(inter, h), s + 7);
            w[$"{p}.mlp.down_proj.weight"] = Rand(new TensorShape(h, inter), s + 8);
        }
        t3.LoadWeights(w);

        using Tensor spk = Rand(new TensorShape(cfg.SpeakerEmbedDim), 5000, 0.3f);
        int[] text = [cfg.StartTextToken, 10, 42, 7, cfg.StopTextToken];

        // The interesting bit: this drives the prefill (t>1) then single-token decode (t=1) — the crash path.
        List<int> tokens = t3.GenerateSpeechTokens(backend, text, spk, cfg.Exaggeration, cfg.MaxNewTokens, seed: 0);

        Assert.True(tokens.Count <= cfg.MaxNewTokens);
        foreach (Tensor t in w.Values) t.Dispose();
    }
}
