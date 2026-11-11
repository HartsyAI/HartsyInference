using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.UniRig;

namespace HartsyInference.ThreeD.Tests.UniRig;

/// <summary>Random weights with the checkpoint's key names and shapes, for tiny-config synthetic tests.</summary>
internal static class UniRigTestWeights
{
    public static Dictionary<string, Tensor> Encoder(UniRigShapeEncoderConfig c, string prefix, int seed)
    {
        Random random = new(seed);
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal);
        int d = c.Width;
        Linear(w, random, $"{prefix}.input_proj", d, c.InputWidth, bias: true);
        Norm(w, random, $"{prefix}.cross_attn.ln_1", d);
        Norm(w, random, $"{prefix}.cross_attn.ln_2", d);
        Norm(w, random, $"{prefix}.cross_attn.ln_3", d);
        Linear(w, random, $"{prefix}.cross_attn.attn.c_q", d, d, bias: false);
        Linear(w, random, $"{prefix}.cross_attn.attn.c_kv", 2 * d, d, bias: false);
        Linear(w, random, $"{prefix}.cross_attn.attn.c_proj", d, d, bias: true);
        Linear(w, random, $"{prefix}.cross_attn.mlp.c_fc", 4 * d, d, bias: true);
        Linear(w, random, $"{prefix}.cross_attn.mlp.c_proj", d, 4 * d, bias: true);
        for (int i = 0; i < c.Layers; i++)
        {
            string b = $"{prefix}.self_attn.resblocks.{i}";
            Norm(w, random, $"{b}.ln_1", d);
            Norm(w, random, $"{b}.ln_2", d);
            Linear(w, random, $"{b}.attn.c_qkv", 3 * d, d, bias: false);
            Linear(w, random, $"{b}.attn.c_proj", d, d, bias: true);
            Linear(w, random, $"{b}.mlp.c_fc", 4 * d, d, bias: true);
            Linear(w, random, $"{b}.mlp.c_proj", d, 4 * d, bias: true);
        }

        Norm(w, random, $"{prefix}.ln_post", d);
        return w;
    }

    public static void Linear(Dictionary<string, Tensor> w, Random random, string path, int outDim, int inDim, bool bias)
    {
        w[$"{path}.weight"] = Random(random, 0.3 / Math.Sqrt(inDim) * Math.Sqrt(3), outDim, inDim);
        if (bias)
        {
            w[$"{path}.bias"] = Random(random, 0.05, outDim);
        }
    }

    public static void Norm(Dictionary<string, Tensor> w, Random random, string path, int dim)
    {
        Tensor weight = Random(random, 0.1, dim);
        foreach (ref float v in weight.AsSpan<float>())
        {
            v += 1f;
        }

        w[$"{path}.weight"] = weight;
        w[$"{path}.bias"] = Random(random, 0.05, dim);
    }

    public static Tensor Random(Random random, double amplitude, params long[] shape)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in tensor.AsSpan<float>())
        {
            v = (float)((random.NextDouble() * 2 - 1) * amplitude);
        }

        return tensor;
    }

    public static void Dispose(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor tensor in weights.Values)
        {
            tensor.Dispose();
        }
    }
}
