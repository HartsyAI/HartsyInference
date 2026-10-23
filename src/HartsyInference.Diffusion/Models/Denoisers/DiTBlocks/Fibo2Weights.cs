using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Load-time splits shared by the Fibo 2.0 blocks. The fused <c>to_kv</c> is K rows then V rows, and the fused
/// <c>w13</c> is w1 (gate, SiLU'd) rows then w3 (linear). The split tensors are appended to the caller's owned list.</summary>
internal static class Fibo2Weights
{
    /// <summary>F32 form of <paramref name="key"/>: the stored tensor when it is already F32, otherwise a widened copy that joins
    /// <paramref name="owned"/> so the owning block frees it.</summary>
    internal static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> weights, string key, List<Tensor> owned)
    {
        Tensor stored = weights[key];
        Tensor f32 = TensorCasts.EnsureF32(stored);
        if (!ReferenceEquals(f32, stored))
            owned.Add(f32);
        return f32;
    }

    /// <summary>Loads a cross-attention (no RoPE) from <c>{prefix}to_q/to_kv/to_out/norm_q/norm_k</c>.</summary>
    internal static AttentionLayer LoadCrossAttention(IReadOnlyDictionary<string, Tensor> weights, string prefix,
        AttentionLayerConfig config, int dim, float eps, List<Tensor> owned)
    {
        Tensor[] keyValue = CheckpointConvertUtils.SplitRows(weights[prefix + "to_kv.weight"], [dim, dim],
            [prefix + "to_kv.k.weight", prefix + "to_kv.v.weight"]);
        owned.AddRange(keyValue);
        return new AttentionLayer(config, new AttentionLayerWeights
        {
            Query = weights[prefix + "to_q.weight"],
            Key = keyValue[0],
            Value = keyValue[1],
            Output = weights[prefix + "to_out.weight"],
            QueryNorm = LoadF32(weights, prefix + "norm_q.weight", owned),
            KeyNorm = LoadF32(weights, prefix + "norm_k.weight", owned),
            QueryNormEps = eps,
            KeyNormEps = eps,
        });
    }

    /// <summary>Loads a SwiGLU FFN from <c>{prefix}w13/w2</c>, splitting w13 into its gate and linear halves.</summary>
    internal static void LoadSwiGlu(SwiGluFfn ffn, IReadOnlyDictionary<string, Tensor> weights, string prefix, int ffnDim,
        List<Tensor> owned)
    {
        Tensor[] gateAndLinear = CheckpointConvertUtils.SplitRows(weights[prefix + "w13.weight"], [ffnDim, ffnDim],
            [prefix + "w1.weight", prefix + "w3.weight"]);
        owned.AddRange(gateAndLinear);
        ffn.LoadSwiGluWeights(gateAndLinear[0], null, gateAndLinear[1], null, weights[prefix + "w2.weight"], null);
    }
}
