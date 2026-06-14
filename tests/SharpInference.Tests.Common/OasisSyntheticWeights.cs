using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;

namespace SharpInference.Tests.Common;

/// <summary>Synthetic weights for the Oasis-500m DiT-S/2 and ViT-VAE structural tests.</summary>
public static unsafe class OasisSyntheticWeights
{
    private static int _seed = 12000;

    public static Dictionary<string, Tensor> BuildDit(OasisDitConfig c)
    {
        int dim = c.HiddenSize, p = c.PatchSize, inC = c.InChannels;
        int mlp = (int)(dim * c.MlpRatio);
        Dictionary<string, Tensor> w = new()
        {
            ["x_embedder.proj.weight"] = R([dim, inC, p, p]),
            ["x_embedder.proj.bias"] = R([dim]),
            ["t_embedder.mlp.0.weight"] = R([dim, c.FreqDim]), ["t_embedder.mlp.0.bias"] = R([dim]),
            ["t_embedder.mlp.2.weight"] = R([dim, dim]), ["t_embedder.mlp.2.bias"] = R([dim]),
            ["external_cond.weight"] = R([dim, c.ExternalCondDim]), ["external_cond.bias"] = R([dim]),
            ["final_layer.adaLN_modulation.1.weight"] = R([2 * dim, dim]), ["final_layer.adaLN_modulation.1.bias"] = R([2 * dim]),
            ["final_layer.linear.weight"] = R([p * p * inC, dim]), ["final_layer.linear.bias"] = R([p * p * inC]),
        };
        for (int i = 0; i < c.Depth; i++)
        {
            string b = $"blocks.{i}";
            foreach (string half in new[] { "s", "t" })
            {
                w[$"{b}.{half}_adaLN_modulation.1.weight"] = R([6 * dim, dim]); w[$"{b}.{half}_adaLN_modulation.1.bias"] = R([6 * dim]);
                w[$"{b}.{half}_attn.to_qkv.weight"] = R([3 * dim, dim]);   // no bias upstream
                w[$"{b}.{half}_attn.to_out.weight"] = R([dim, dim]); w[$"{b}.{half}_attn.to_out.bias"] = R([dim]);
                w[$"{b}.{half}_mlp.fc1.weight"] = R([mlp, dim]); w[$"{b}.{half}_mlp.fc1.bias"] = R([mlp]);
                w[$"{b}.{half}_mlp.fc2.weight"] = R([dim, mlp]); w[$"{b}.{half}_mlp.fc2.bias"] = R([dim]);
            }
        }
        return w;
    }

    public static Dictionary<string, Tensor> BuildVitVae(int latentDim, int patchSize, int dim, int encDepth, int decDepth, float mlpRatio = 4.0f)
    {
        int mlp = (int)(dim * mlpRatio);
        Dictionary<string, Tensor> w = new()
        {
            ["patch_embed.proj.weight"] = R([dim, 3, patchSize, patchSize]),
            ["patch_embed.proj.bias"] = R([dim]),
            ["enc_norm.weight"] = R([dim]), ["enc_norm.bias"] = R([dim]),
            ["quant_conv.weight"] = R([2 * latentDim, dim]), ["quant_conv.bias"] = R([2 * latentDim]),
            ["post_quant_conv.weight"] = R([dim, latentDim]), ["post_quant_conv.bias"] = R([dim]),
            ["dec_norm.weight"] = R([dim]), ["dec_norm.bias"] = R([dim]),
            ["predictor.weight"] = R([3 * patchSize * patchSize, dim]), ["predictor.bias"] = R([3 * patchSize * patchSize]),
        };
        for (int i = 0; i < encDepth; i++) AddVitBlock(w, $"encoder.{i}", dim, mlp);
        for (int i = 0; i < decDepth; i++) AddVitBlock(w, $"decoder.{i}", dim, mlp);
        return w;
    }

    private static void AddVitBlock(Dictionary<string, Tensor> w, string p, int dim, int mlp)
    {
        w[$"{p}.norm1.weight"] = R([dim]); w[$"{p}.norm1.bias"] = R([dim]);
        w[$"{p}.attn.qkv.weight"] = R([3 * dim, dim]); w[$"{p}.attn.qkv.bias"] = R([3 * dim]);
        w[$"{p}.attn.proj.weight"] = R([dim, dim]); w[$"{p}.attn.proj.bias"] = R([dim]);
        w[$"{p}.norm2.weight"] = R([dim]); w[$"{p}.norm2.bias"] = R([dim]);
        w[$"{p}.mlp.fc1.weight"] = R([mlp, dim]); w[$"{p}.mlp.fc1.bias"] = R([mlp]);
        w[$"{p}.mlp.fc2.weight"] = R([dim, mlp]); w[$"{p}.mlp.fc2.bias"] = R([dim]);
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
