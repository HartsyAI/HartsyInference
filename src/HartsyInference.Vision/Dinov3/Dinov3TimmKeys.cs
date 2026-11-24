using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.Vision.Dinov3;

/// <summary>
/// A timm DINOv3 backbone (<c>patch_embed.proj</c>, <c>blocks.i.attn.qkv</c>, <c>gamma_1</c>, <c>mlp.fc1</c> or <c>mlp.fc1_g</c>/<c>fc1_x</c>) renamed to the HF
/// layout <see cref="Dinov3VisionEncoder.LoadWeights"/> reads, the way <see cref="Dinov2HubKeys"/> does for hub DINOv2. Nothing is copied: the fused QKV
/// projection becomes zero-copy <see cref="Tensor.SliceRows"/> views that keep the fused tensor alive, and every other entry is the caller's tensor under its new
/// name. Used by models that embed a timm DINOv3 under a prefix (Depth Pro 2's <c>encoder.</c>).
/// </summary>
public static class Dinov3TimmKeys
{
    private static readonly string[] QkvNames = ["q_proj", "k_proj", "v_proj"];

    /// <summary>The encoder's weights by HF name.</summary>
    /// <param name="weights">The timm tensors by name.</param>
    /// <param name="prefix">Everything before <c>patch_embed.proj.weight</c> and the other keys, such as <c>encoder.</c>; empty for none.</param>
    /// <param name="preset">The network the weights belong to. <see cref="Dinov2Preset.GatedMlp"/> selects <c>fc1_g</c>/<c>fc1_x</c> (SwiGLU) over <c>fc1</c>,
    /// <see cref="Dinov2Preset.AttentionQkvBias"/> whether <c>attn.qkv.bias</c> is read, <see cref="Dinov2Preset.HasFinalNorm"/> whether <c>norm.*</c> is.</param>
    /// <exception cref="InvalidOperationException">A weight is missing or has the wrong shape.</exception>
    public static Dictionary<string, Tensor> ToHuggingFace(IReadOnlyDictionary<string, Tensor> weights, string prefix, Dinov2Preset preset)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(prefix);
        ArgumentNullException.ThrowIfNull(preset);
        int d = preset.HiddenSize, ff = preset.IntermediateSize, patch = preset.PatchSize;
        Tensor Get(string key, params long[] shape)
        {
            if (!weights.TryGetValue(prefix + key, out Tensor? tensor))
                throw new InvalidOperationException($"timm DINOv3 weight '{prefix + key}' is missing.");
            if (!tensor.Shape.Equals(new TensorShape(shape)))
                throw new InvalidOperationException($"timm DINOv3 weight '{prefix + key}' has shape {tensor.Shape}, expected {new TensorShape(shape)}.");
            return tensor;
        }

        Dictionary<string, Tensor> result = new(StringComparer.Ordinal)
        {
            ["embeddings.patch_embeddings.weight"] = Get("patch_embed.proj.weight", d, 3, patch, patch),
            ["embeddings.patch_embeddings.bias"] = Get("patch_embed.proj.bias", d),
            ["embeddings.cls_token"] = Get("cls_token", 1, 1, d),
            ["embeddings.register_tokens"] = Get("reg_token", 1, preset.NumRegisterTokens, d),
        };
        if (preset.HasFinalNorm)
        {
            result["norm.weight"] = Get("norm.weight", d);
            result["norm.bias"] = Get("norm.bias", d);
        }

        for (int i = 0; i < preset.NumLayers; i++)
        {
            string timm = $"blocks.{i}.", hf = $"layer.{i}.";
            result[hf + "norm1.weight"] = Get(timm + "norm1.weight", d);
            result[hf + "norm1.bias"] = Get(timm + "norm1.bias", d);
            result[hf + "norm2.weight"] = Get(timm + "norm2.weight", d);
            result[hf + "norm2.bias"] = Get(timm + "norm2.bias", d);
            result[hf + "attention.o_proj.weight"] = Get(timm + "attn.proj.weight", d, d);
            result[hf + "attention.o_proj.bias"] = Get(timm + "attn.proj.bias", d);
            result[hf + "layer_scale1.lambda1"] = Get(timm + "gamma_1", d);
            result[hf + "layer_scale2.lambda1"] = Get(timm + "gamma_2", d);
            result[hf + "mlp.down_proj.weight"] = Get(timm + "mlp.fc2.weight", d, ff);
            result[hf + "mlp.down_proj.bias"] = Get(timm + "mlp.fc2.bias", d);
            if (preset.GatedMlp)
            {
                // timm SwiGLU: act(fc1_g(x)) * fc1_x(x), the gate and the up projection of the HF layout.
                result[hf + "mlp.gate_proj.weight"] = Get(timm + "mlp.fc1_g.weight", ff, d);
                result[hf + "mlp.gate_proj.bias"] = Get(timm + "mlp.fc1_g.bias", ff);
                result[hf + "mlp.up_proj.weight"] = Get(timm + "mlp.fc1_x.weight", ff, d);
                result[hf + "mlp.up_proj.bias"] = Get(timm + "mlp.fc1_x.bias", ff);
            }
            else
            {
                result[hf + "mlp.up_proj.weight"] = Get(timm + "mlp.fc1.weight", ff, d);
                result[hf + "mlp.up_proj.bias"] = Get(timm + "mlp.fc1.bias", ff);
            }

            Tensor qkvWeight = Get(timm + "attn.qkv.weight", 3 * d, d);
            for (int part = 0; part < 3; part++)
                result[$"{hf}attention.{QkvNames[part]}.weight"] = qkvWeight.SliceRows(part * d, d);
            if (preset.AttentionQkvBias)
            {
                // The key bias is masked in the reference and the HF layout has none, so only the query and value slices are kept.
                Tensor qkvBias = Get(timm + "attn.qkv.bias", 3 * d);
                result[hf + "attention.q_proj.bias"] = qkvBias.SliceRows(0, d);
                result[hf + "attention.v_proj.bias"] = qkvBias.SliceRows(2L * d, d);
            }
        }

        return result;
    }
}
