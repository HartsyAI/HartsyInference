using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>The F32 weights of one vision-tower block, named after the checkpoint's <c>vision.blocks.{i}.*</c> keys; matrices are <c>[out, in]</c>.</summary>
/// <param name="Norm1">Pre-attention RMSNorm weight, <c>[dim]</c>.</param>
/// <param name="Wqkv">Fused query, key and value projection, <c>[3 * dim, dim]</c>, rows ordered q, k, v and by head within each.</param>
/// <param name="WqkvBias">Its bias, <c>[3 * dim]</c>.</param>
/// <param name="Wo">Attention output projection, <c>[dim, dim]</c>.</param>
/// <param name="WoBias">Its bias, <c>[dim]</c>.</param>
/// <param name="Norm2">Pre-MLP RMSNorm weight, <c>[dim]</c>.</param>
/// <param name="W1">Gate and up projection without bias, <c>[2 * inter, dim]</c>, gate rows first.</param>
/// <param name="W2">Down projection without bias, <c>[dim, inter]</c>.</param>
public sealed record DeepSeekV41VisionBlockWeights(
    Tensor Norm1, Tensor Wqkv, Tensor WqkvBias, Tensor Wo, Tensor WoBias, Tensor Norm2, Tensor W1, Tensor W2)
{
    /// <summary>Checks every tensor against <paramref name="config"/>; <paramref name="index"/> names the block in the refusal.</summary>
    internal void Validate(DeepSeekV41VisionConfig config, int index)
    {
        long dim = config.HiddenSize, inter = config.IntermediateSize;
        string prefix = $"vision.blocks.{index}.";
        DeepSeekV41VisionTensors.Require(Norm1, prefix + "norm1.weight", dim);
        DeepSeekV41VisionTensors.Require(Wqkv, prefix + "attn.wqkv.weight", 3 * dim, dim);
        DeepSeekV41VisionTensors.Require(WqkvBias, prefix + "attn.wqkv.bias", 3 * dim);
        DeepSeekV41VisionTensors.Require(Wo, prefix + "attn.wo.weight", dim, dim);
        DeepSeekV41VisionTensors.Require(WoBias, prefix + "attn.wo.bias", dim);
        DeepSeekV41VisionTensors.Require(Norm2, prefix + "norm2.weight", dim);
        DeepSeekV41VisionTensors.Require(W1, prefix + "mlp.w1.weight", 2 * inter, dim);
        DeepSeekV41VisionTensors.Require(W2, prefix + "mlp.w2.weight", dim, inter);
    }

    internal void Dispose()
    {
        foreach (Tensor tensor in new[] { Norm1, Wqkv, WqkvBias, Wo, WoBias, Norm2, W1, W2 }) tensor?.Dispose();
    }
}
