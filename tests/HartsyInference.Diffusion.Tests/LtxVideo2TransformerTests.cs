using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tiny-config end-to-end smoke test for the LTX-2.3 dual-stream <see cref="LtxVideo2Transformer"/> on CPU:
/// verifies the full forward (patchify → 8 AdaLN tables → blocks → per-stream output) produces correctly-shaped,
/// finite video and audio velocities. No GPU/checkpoint; numerics vs the real checkpoint are validation-pending.</summary>
public unsafe class LtxVideo2TransformerTests
{
    [Fact]
    public void Transformer_TinyConfig_ProducesDualOutputShapes()
    {
        CpuBackend backend = new();
        LtxVideo2Config cfg = new()
        {
            InChannels = 4, OutChannels = 4, NumHeads = 2, HeadDim = 4, CrossAttentionDim = 8,
            AudioInChannels = 4, AudioOutChannels = 4, AudioNumHeads = 2, AudioHeadDim = 2, AudioCrossAttentionDim = 4,
            NumLayers = 2, FfnMultiplier = 4,
        };
        LtxVideo2Transformer transformer = new(cfg);
        transformer.LoadWeights(BuildWeights(cfg));

        int f = 2, h = 2, w = 2, sv = f * h * w;     // 8 video tokens
        int audioFrames = 3;
        Tensor video = RandRows(sv, cfg.InChannels, 11);
        Tensor audio = RandRows(audioFrames, cfg.AudioInChannels, 12);
        Tensor encV = RandRows(3, cfg.CrossAttentionDim, 13);       // connector video features
        Tensor encA = RandRows(3, cfg.AudioCrossAttentionDim, 14);  // connector audio features

        (Tensor outV, Tensor outA) = transformer.Forward(backend, video, audio, encV, encA,
            timestep: 500f, (f, h, w), audioFrames, fps: 24.0, null, null);

        Assert.Equal(sv, (int)outV.Shape[0]);
        Assert.Equal(cfg.OutChannels, (int)outV.Shape[1]);
        Assert.Equal(audioFrames, (int)outA.Shape[0]);
        Assert.Equal(cfg.AudioOutChannels, (int)outA.Shape[1]);
        AssertFinite(outV);
        AssertFinite(outA);
    }

    private static Dictionary<string, Tensor> BuildWeights(LtxVideo2Config c)
    {
        int v = c.InnerDim, a = c.AudioInnerDim;
        int ffV = c.FfnMultiplier * v, ffA = c.FfnMultiplier * a;
        Dictionary<string, Tensor> w = new()
        {
            ["proj_in.weight"] = R([v, c.InChannels]), ["proj_in.bias"] = R([v]),
            ["audio_proj_in.weight"] = R([a, c.AudioInChannels]), ["audio_proj_in.bias"] = R([a]),
            ["proj_out.weight"] = R([c.OutChannels, v]), ["proj_out.bias"] = R([c.OutChannels]),
            ["audio_proj_out.weight"] = R([c.AudioOutChannels, a]), ["audio_proj_out.bias"] = R([c.AudioOutChannels]),
            ["scale_shift_table"] = R([2, v]), ["audio_scale_shift_table"] = R([2, a]),
        };
        AddAdaLn(w, "time_embed", v, 9);
        AddAdaLn(w, "audio_time_embed", a, 9);
        AddAdaLn(w, "prompt_adaln", v, 2);
        AddAdaLn(w, "audio_prompt_adaln", a, 2);
        AddAdaLn(w, "av_cross_attn_video_scale_shift", v, 4);
        AddAdaLn(w, "av_cross_attn_audio_scale_shift", a, 4);
        AddAdaLn(w, "av_cross_attn_video_a2v_gate", v, 1);
        AddAdaLn(w, "av_cross_attn_audio_v2a_gate", a, 1);

        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"transformer_blocks.{i}";
            AddAttn(w, $"{p}.attn1", v, v, c.NumHeads, c.HeadDim, v);
            AddAttn(w, $"{p}.attn2", v, c.CrossAttentionDim, c.NumHeads, c.HeadDim, v);
            AddAttn(w, $"{p}.audio_attn1", a, a, c.AudioNumHeads, c.AudioHeadDim, a);
            AddAttn(w, $"{p}.audio_attn2", a, c.AudioCrossAttentionDim, c.AudioNumHeads, c.AudioHeadDim, a);
            AddAttn(w, $"{p}.audio_to_video_attn", v, a, c.AudioNumHeads, c.AudioHeadDim, v);
            AddAttn(w, $"{p}.video_to_audio_attn", a, v, c.AudioNumHeads, c.AudioHeadDim, a);

            w[$"{p}.scale_shift_table"] = R([9, v]);
            w[$"{p}.audio_scale_shift_table"] = R([9, a]);
            w[$"{p}.prompt_scale_shift_table"] = R([2, v]);
            w[$"{p}.audio_prompt_scale_shift_table"] = R([2, a]);
            w[$"{p}.scale_shift_table_a2v_ca_video"] = R([5, v]);
            w[$"{p}.scale_shift_table_a2v_ca_audio"] = R([5, a]);

            w[$"{p}.ff.net.0.proj.weight"] = R([ffV, v]); w[$"{p}.ff.net.0.proj.bias"] = R([ffV]);
            w[$"{p}.ff.net.2.weight"] = R([v, ffV]); w[$"{p}.ff.net.2.bias"] = R([v]);
            w[$"{p}.audio_ff.net.0.proj.weight"] = R([ffA, a]); w[$"{p}.audio_ff.net.0.proj.bias"] = R([ffA]);
            w[$"{p}.audio_ff.net.2.weight"] = R([a, ffA]); w[$"{p}.audio_ff.net.2.bias"] = R([a]);
        }
        return w;
    }

    private static void AddAdaLn(Dictionary<string, Tensor> w, string p, int dim, int numParams)
    {
        w[$"{p}.emb.timestep_embedder.linear_1.weight"] = R([dim, 256]); w[$"{p}.emb.timestep_embedder.linear_1.bias"] = R([dim]);
        w[$"{p}.emb.timestep_embedder.linear_2.weight"] = R([dim, dim]); w[$"{p}.emb.timestep_embedder.linear_2.bias"] = R([dim]);
        w[$"{p}.linear.weight"] = R([numParams * dim, dim]); w[$"{p}.linear.bias"] = R([numParams * dim]);
    }

    private static void AddAttn(Dictionary<string, Tensor> w, string p, int qIn, int kvIn, int heads, int hd, int outDim)
    {
        int inner = heads * hd;
        w[$"{p}.to_q.weight"] = R([inner, qIn]); w[$"{p}.to_q.bias"] = R([inner]);
        w[$"{p}.to_k.weight"] = R([inner, kvIn]); w[$"{p}.to_k.bias"] = R([inner]);
        w[$"{p}.to_v.weight"] = R([inner, kvIn]); w[$"{p}.to_v.bias"] = R([inner]);
        w[$"{p}.to_out.0.weight"] = R([outDim, inner]); w[$"{p}.to_out.0.bias"] = R([outDim]);
        w[$"{p}.q_norm.weight"] = R([inner]); w[$"{p}.k_norm.weight"] = R([inner]);
        w[$"{p}.to_gate_logits.weight"] = R([heads, qIn]); w[$"{p}.to_gate_logits.bias"] = R([heads]);
    }

    private static int _seed = 1;
    private static Tensor R(int[] dims)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(_seed++);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        return t;
    }

    private static Tensor RandRows(int rows, int cols, int seed)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static void AssertFinite(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }
}
