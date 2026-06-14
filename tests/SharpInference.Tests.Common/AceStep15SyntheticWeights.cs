using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;

namespace SharpInference.Tests.Common;

/// <summary>Synthetic weights for the ACE-Step v1.5 turbo stack (DiT + condition encoders) at tiny dims for CPU
/// structural tests, emitting the exact §5 checkpoint key spellings (<c>decoder.*</c> / <c>encoder.*</c>).</summary>
public static unsafe class AceStep15SyntheticWeights
{
    private static int _seed = 21500;

    /// <summary>Tiny v1.5 config used across the tests: dim 16 (2 heads × 8, GQA 2:1), 2 DiT layers, SwiGLU 32,
    /// sliding window 3 (small enough that tiny sequences exercise the mask), 4-ch latents (in 12), patch 2.</summary>
    public static AceStep15Config TinyConfig => new()
    {
        HiddenSize = 16, NumLayers = 2, NumHeads = 2, NumKvHeads = 1, HeadDim = 8,
        IntermediateSize = 32, SlidingWindow = 3, InChannels = 12, PatchSize = 2, LatentChannels = 4,
        TextHiddenDim = 6, TimbreHiddenDim = 4, LyricEncoderLayers = 2, TimbreEncoderLayers = 2,
        FreqDim = 8, SampleRate = 800, SamplesPerLatent = 2,
    };

    /// <summary>One dict mirroring the single v1.5 main safetensors: DiT under <c>decoder.</c>, condition encoders
    /// under <c>encoder.</c>.</summary>
    public static Dictionary<string, Tensor> BuildModel(AceStep15Config c)
    {
        int dim = c.HiddenSize, inter = c.IntermediateSize;
        Dictionary<string, Tensor> w = new()
        {
            ["decoder.proj_in.1.weight"] = R([dim, c.InChannels, c.PatchSize]),
            ["decoder.proj_in.1.bias"] = R([dim]),
            ["decoder.condition_embedder.weight"] = R([dim, dim]),
            ["decoder.condition_embedder.bias"] = R([dim]),
            ["decoder.norm_out.weight"] = R([dim]),
            ["decoder.scale_shift_table"] = R([1, 2, dim]),
            ["decoder.proj_out.1.weight"] = R([dim, c.LatentChannels, c.PatchSize]),   // ConvTranspose1d [C_in, C_out, K]
            ["decoder.proj_out.1.bias"] = R([c.LatentChannels]),
        };
        foreach (string embed in new[] { "decoder.time_embed", "decoder.time_embed_r" })
        {
            w[$"{embed}.linear_1.weight"] = R([dim, c.FreqDim]); w[$"{embed}.linear_1.bias"] = R([dim]);
            w[$"{embed}.linear_2.weight"] = R([dim, dim]); w[$"{embed}.linear_2.bias"] = R([dim]);
            w[$"{embed}.time_proj.weight"] = R([6 * dim, dim]); w[$"{embed}.time_proj.bias"] = R([6 * dim]);
        }
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"decoder.layers.{i}";
            w[$"{p}.scale_shift_table"] = R([1, 6, dim]);
            w[$"{p}.self_attn_norm.weight"] = R([dim]);
            w[$"{p}.cross_attn_norm.weight"] = R([dim]);
            w[$"{p}.mlp_norm.weight"] = R([dim]);
            AddAttention(w, $"{p}.self_attn", c);
            AddAttention(w, $"{p}.cross_attn", c);
            w[$"{p}.mlp.gate_proj.weight"] = R([inter, dim]);
            w[$"{p}.mlp.up_proj.weight"] = R([inter, dim]);
            w[$"{p}.mlp.down_proj.weight"] = R([dim, inter]);
        }

        w["encoder.text_projector.weight"] = R([dim, c.TextHiddenDim]);
        AddConditionEncoder(w, "encoder.lyric_encoder", c.TextHiddenDim, c.LyricEncoderLayers, c, special: false);
        AddConditionEncoder(w, "encoder.timbre_encoder", c.TimbreHiddenDim, c.TimbreEncoderLayers, c, special: true);
        return w;
    }

    private static void AddConditionEncoder(Dictionary<string, Tensor> w, string prefix, int inputDim, int layers,
        AceStep15Config c, bool special)
    {
        int dim = c.HiddenSize;
        w[$"{prefix}.embed_tokens.weight"] = R([dim, inputDim]);
        w[$"{prefix}.embed_tokens.bias"] = R([dim]);
        w[$"{prefix}.norm.weight"] = R([dim]);
        if (special) w[$"{prefix}.special_token"] = R([1, 1, dim]);
        for (int i = 0; i < layers; i++)
        {
            string p = $"{prefix}.layers.{i}";
            w[$"{p}.input_layernorm.weight"] = R([dim]);
            w[$"{p}.post_attention_layernorm.weight"] = R([dim]);
            AddAttention(w, $"{p}.self_attn", c);
            w[$"{p}.mlp.gate_proj.weight"] = R([c.IntermediateSize, dim]);
            w[$"{p}.mlp.up_proj.weight"] = R([c.IntermediateSize, dim]);
            w[$"{p}.mlp.down_proj.weight"] = R([dim, c.IntermediateSize]);
        }
    }

    private static void AddAttention(Dictionary<string, Tensor> w, string prefix, AceStep15Config c)
    {
        int dim = c.HiddenSize, q = c.NumHeads * c.HeadDim, kv = c.NumKvHeads * c.HeadDim;
        w[$"{prefix}.q_proj.weight"] = R([q, dim]);
        w[$"{prefix}.k_proj.weight"] = R([kv, dim]);
        w[$"{prefix}.v_proj.weight"] = R([kv, dim]);
        w[$"{prefix}.o_proj.weight"] = R([dim, q]);
        w[$"{prefix}.q_norm.weight"] = R([c.HeadDim]);
        w[$"{prefix}.k_norm.weight"] = R([c.HeadDim]);
    }

    private static Tensor R(int[] dims)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(_seed++);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        return t;
    }
}
