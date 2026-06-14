using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;

namespace SharpInference.Tests.Common;

/// <summary>Shared synthetic-weight generator for the Wan-Video DiT (<see cref="WanVideoTransformer"/>). One source of truth for the Diffusion + Video test projects. (The Wan VAE reuses <see cref="LanceSyntheticWeights.BuildVae"/> — it's the same Wan2.2 decoder.)</summary>
public static unsafe class WanSyntheticWeights
{
    private static int _seed = 5000;

    public static Dictionary<string, Tensor> BuildTransformer(WanVideoConfig c)
    {
        int dim = c.InnerDim, ff = c.FfnDim;
        int outVec = c.OutChannels * c.PatchSize.T * c.PatchSize.H * c.PatchSize.W;
        Dictionary<string, Tensor> w = new()
        {
            ["patch_embedding.weight"] = R([dim, c.InChannels, c.PatchSize.T, c.PatchSize.H, c.PatchSize.W]),
            ["patch_embedding.bias"] = R([dim]),
            ["proj_out.weight"] = R([outVec, dim]), ["proj_out.bias"] = R([outVec]),
            ["scale_shift_table"] = R([1, 2, dim]),
            ["condition_embedder.time_embedder.linear_1.weight"] = R([dim, c.FreqDim]), ["condition_embedder.time_embedder.linear_1.bias"] = R([dim]),
            ["condition_embedder.time_embedder.linear_2.weight"] = R([dim, dim]), ["condition_embedder.time_embedder.linear_2.bias"] = R([dim]),
            ["condition_embedder.time_proj.weight"] = R([6 * dim, dim]), ["condition_embedder.time_proj.bias"] = R([6 * dim]),
            ["condition_embedder.text_embedder.linear_1.weight"] = R([dim, c.TextDim]), ["condition_embedder.text_embedder.linear_1.bias"] = R([dim]),
            ["condition_embedder.text_embedder.linear_2.weight"] = R([dim, dim]), ["condition_embedder.text_embedder.linear_2.bias"] = R([dim]),
        };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"blocks.{i}";
            w[$"{p}.scale_shift_table"] = R([1, 6, dim]);
            w[$"{p}.norm2.weight"] = R([dim]); w[$"{p}.norm2.bias"] = R([dim]);
            foreach (string a in new[] { "attn1", "attn2" })
            {
                w[$"{p}.{a}.to_q.weight"] = R([dim, dim]); w[$"{p}.{a}.to_q.bias"] = R([dim]);
                w[$"{p}.{a}.to_k.weight"] = R([dim, dim]); w[$"{p}.{a}.to_k.bias"] = R([dim]);
                w[$"{p}.{a}.to_v.weight"] = R([dim, dim]); w[$"{p}.{a}.to_v.bias"] = R([dim]);
                w[$"{p}.{a}.to_out.0.weight"] = R([dim, dim]); w[$"{p}.{a}.to_out.0.bias"] = R([dim]);
                w[$"{p}.{a}.norm_q.weight"] = R([dim]); w[$"{p}.{a}.norm_k.weight"] = R([dim]);
            }
            w[$"{p}.ffn.net.0.proj.weight"] = R([ff, dim]); w[$"{p}.ffn.net.0.proj.bias"] = R([ff]);
            w[$"{p}.ffn.net.2.weight"] = R([dim, ff]); w[$"{p}.ffn.net.2.bias"] = R([dim]);
        }
        return w;
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
