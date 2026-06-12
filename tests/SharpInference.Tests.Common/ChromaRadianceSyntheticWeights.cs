using SharpInference.Core.Tensors;

namespace SharpInference.Tests.Common;

/// <summary>Synthetic weights for Chroma Radiance pixel-IO structural tests (conv patchifier + NeRF head).
/// Tiny dims only — the Chroma backbone itself is covered by the existing Chroma tests.</summary>
public static unsafe class ChromaRadianceSyntheticWeights
{
    private static int s_seed = 14000;

    /// <summary>NeRF head weights for the given dims. <paramref name="convFinalLayer"/> selects variant A
    /// (<c>nerf_final_layer_conv</c>, 3×3 conv) vs variant B (<c>nerf_final_layer</c>, per-pixel linear).</summary>
    public static Dictionary<string, Tensor> BuildNerfHead(
        int tokenDim, int nerfHidden, int maxFreqs, int depth, int mlpRatio, bool convFinalLayer)
    {
        int inner = nerfHidden * mlpRatio;
        int paramDim = 3 * nerfHidden * inner;
        Dictionary<string, Tensor> w = new()
        {
            ["nerf_image_embedder.embedder.0.weight"] = R([nerfHidden, 3 + maxFreqs * maxFreqs]),
            ["nerf_image_embedder.embedder.0.bias"] = R([nerfHidden]),
        };
        for (int i = 0; i < depth; i++)
        {
            w[$"nerf_blocks.{i}.param_generator.weight"] = R([paramDim, tokenDim]);
            w[$"nerf_blocks.{i}.param_generator.bias"] = R([paramDim]);
            w[$"nerf_blocks.{i}.norm.scale"] = R([nerfHidden]);
        }
        if (convFinalLayer)
        {
            w["nerf_final_layer_conv.norm.scale"] = R([nerfHidden]);
            w["nerf_final_layer_conv.conv.weight"] = R([3, nerfHidden, 3, 3]);
            w["nerf_final_layer_conv.conv.bias"] = R([3]);
        }
        else
        {
            w["nerf_final_layer.norm.scale"] = R([nerfHidden]);
            w["nerf_final_layer.linear.weight"] = R([3, nerfHidden]);
            w["nerf_final_layer.linear.bias"] = R([3]);
        }
        return w;
    }

    /// <summary>Conv patchifier weights: <c>img_in_patch = Conv2d(3 → hidden, kP, sP)</c>.</summary>
    public static Dictionary<string, Tensor> BuildPatchifier(int hidden, int patchSize)
    {
        return new Dictionary<string, Tensor>
        {
            ["img_in_patch.weight"] = R([hidden, 3, patchSize, patchSize]),
            ["img_in_patch.bias"] = R([hidden]),
        };
    }

    private static Tensor R(int[] dims)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(s_seed++);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        return t;
    }
}
