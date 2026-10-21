using System.Reflection;
using HartsyInference.Audio.Models.LanguageModels.Qwen3;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using static HartsyInference.Tests.Common.Attention.AttentionGolden;
using Owner = HartsyInference.Tests.Common.Attention.GoldenOwner;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Audio.Tests.Attention;

internal static unsafe partial class AttentionWaveEGoldenCases
{
    private static void RegisterDevice3(Dictionary<string, Func<CpuBackend, string>> d)
    {
        d["musicgen_step_b1"] = cpu => MusicGenSteps(cpu, 1);
        d["musicgen_step_b2_cfg"] = cpu => MusicGenSteps(cpu, 2);
        d["minimaxmusic3_depth_b1"] = cpu => MiniMaxDepth(cpu, batch: 1, [3, 1, 1]);
        d["minimaxmusic3_depth_b2_fold"] = cpu => MiniMaxDepth(cpu, batch: 2, [2, 1]);
        d["yue2_acoustic"] = Yue2Velocity;
    }

    // ── MusicGen decoder block: device-position self Flash + non-causal cross Flash through the CPU defaults ────────

    private static string MusicGenSteps(CpuBackend cpu, int b)
    {
        const int h = 32, heads = 2, hd = h / heads, crossLen = 5, steps = 4;
        Owner o = new();
        MusicGenConfig cfg = new() { Hidden = h, NumLayers = 1, NumHeads = heads };
        Weights w = new(100u + (uint)b);
        foreach (string n in new[] { "self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm" })
        {
            w.AddNorm($"m.{n}.weight", h); w.AddBias($"m.{n}.bias", h);
        }
        foreach (string a in new[] { "self_attn", "encoder_attn" })
            foreach (string n in new[] { "q_proj", "k_proj", "v_proj", "out_proj" }) w.Add($"m.{a}.{n}.weight", h, h);
        w.Add("m.fc1.weight", 4 * h, h); w.Add("m.fc2.weight", h, 4 * h);
        using MusicGenBlock block = new(cfg);
        block.LoadWeights(w, "m");
        using MusicGenKvCache cache = new(b, 1, heads, 16, hd);
        block.PrimeCross(cpu, Seq(o, 1001, 1f, b, crossLen, h), cache.Layers[0]);
        List<(string, Tensor)> outs = [];
        for (int pos = 0; pos < steps; pos++)
            outs.Add(($"step{pos}", o.Own(block.ForwardStep(cpu, Seq(o, 1010 + (uint)pos, 1f, b, 1, h), cache.Layers[0], pos, crossLen))));
        return Finish(o, [.. outs]);
    }

    // ── MiniMax-Music3 depth layer (fixed 4096 wide, one layer, built by reflection: Layer is a private nested class) ─

    private static string MiniMaxDepth(CpuBackend cpu, int batch, int[] steps)
    {
        const int h = 4096, inter = 6144, heads = 16, hd = h / heads;
        const float s = 0.02f;
        Owner o = new();
        Weights w = new(110u + (uint)batch);
        w.AddNorm("l.input_layernorm.weight", h); w.AddNorm("l.post_attention_layernorm.weight", h);
        foreach (string n in new[] { "to_q", "to_k", "to_v", "to_out" }) w.AddTensor($"l.attn.{n}.weight", s, h, h);
        w.AddTensor("l.gate_proj.weight", s, inter, h); w.AddTensor("l.up_proj.weight", s, inter, h); w.AddTensor("l.down_proj.weight", s, h, inter);
        using MiniMaxMusic3DepthDecoder owner = new();
        Type layerType = typeof(MiniMaxMusic3DepthDecoder).GetNestedType("Layer", BindingFlags.NonPublic)!;
        object layer = Activator.CreateInstance(layerType, nonPublic: true)!;
        layerType.GetMethod("Load")!.Invoke(layer, [owner, w, "l"]);
        MethodInfo forward = layerType.GetMethod("Forward")!;
        using MiniMaxMusic3DepthCache cache = new(1, batch * heads, hd, 8);
        List<(string, Tensor)> outs = [];
        for (int i = 0; i < steps.Length; i++)
        {
            Tensor hidden = Seq(o, 1100 + (uint)i, 1f, batch, steps[i], h);
            outs.Add(($"call{i}", o.Own((Tensor)forward.Invoke(layer, [cpu, hidden, batch, steps[i], cache, 0])!)));
            cache.Advance(steps[i]);
        }
        return Finish(o, [.. outs]);
    }

    // ── Yue2 acoustic transformer (the F32 path; F16 activations are unreachable on the CPU backend) ────────────────

    private static string Yue2Velocity(CpuBackend cpu)
    {
        const int hidden = 32, layers = 2, heads = 4, kvHeads = 2, hd = 8, inter = 48, latent = 8, tsWidth = 16, frames = 4, arLength = 5;
        Owner o = new();
        Qwen3Config nar = new()
        {
            HiddenSize = hidden, NumHiddenLayers = layers, NumAttentionHeads = heads, NumKeyValueHeads = kvHeads, HeadDim = hd,
            IntermediateSize = inter, MaxPositionEmbeddings = 64, RopeTheta = 10_000f,
        };
        Yue2Config cfg = new() { Ar = nar, Nar = nar, LatentDim = latent, TimestepEmbedWidth = tsWidth };
        Weights w = new(120);
        w.Linear("vae2llm", hidden, latent, true); w.Linear("llm2vae", latent, hidden, true);
        w.Linear("time_embedder.mlp.0", hidden, tsWidth, true); w.Linear("time_embedder.mlp.2", hidden, hidden, true);
        w.AddTensor("latent_pos_embed.pe", 0.3f, 32, hidden);
        w.AddNorm("model.norm.weight", hidden);
        for (int i = 0; i < layers; i++)
        {
            string p = $"model.layers.{i}";
            w.Add($"{p}.self_attn.q_proj.weight", heads * hd, hidden); w.Add($"{p}.self_attn.k_proj.weight", kvHeads * hd, hidden);
            w.Add($"{p}.self_attn.v_proj.weight", kvHeads * hd, hidden); w.Add($"{p}.self_attn.o_proj.weight", hidden, heads * hd);
            w.AddNorm($"{p}.self_attn.q_norm.weight", hd); w.AddNorm($"{p}.self_attn.k_norm.weight", hd);
            w.Add($"{p}.mlp.gate_proj.weight", inter, hidden); w.Add($"{p}.mlp.up_proj.weight", inter, hidden); w.Add($"{p}.mlp.down_proj.weight", hidden, inter);
            w.AddNorm($"{p}.input_layernorm.weight", hidden); w.AddNorm($"{p}.post_attention_layernorm.weight", hidden);
        }
        using Yue2AcousticTransformer tr = new(cfg);
        tr.LoadWeights(w);
        (Tensor Key, Tensor Value)[] prefix = new (Tensor, Tensor)[layers];
        for (int i = 0; i < layers; i++)
            prefix[i] = (Seq(o, 1200 + (uint)i, 1f, 1, kvHeads, arLength, hd), Seq(o, 1210 + (uint)i, 1f, 1, kvHeads, arLength, hd));
        Tensor state = Seq(o, 1220, 1f, frames * latent);
        float[] velocity = new float[frames * latent];
        tr.Velocity(cpu, new ReadOnlySpan<float>(state.DataPointer, frames * latent), 0.3f, prefix, arLength, velocity);
        string digest = Digest("velocity", velocity);
        Finish(o);
        return digest;
    }
}
