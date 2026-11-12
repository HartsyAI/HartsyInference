using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Dinov2;

/// <summary>
/// The torch.hub DINOv2 names (<c>patch_embed.proj</c>, <c>blocks.i.attn.qkv</c>, <c>ls1.gamma</c>, <c>norm</c>) renamed to the HF-style names <see cref="Dinov2VisionEncoder"/>
/// reads. Nothing is copied: the fused QKV projection is split into zero-copy <see cref="Tensor.SliceRows"/> views that keep the fused tensor alive, every other
/// entry is the caller's tensor under its new name. Shared by every model that carries a hub-style DINOv2 (Depth-Anything-V2, SAM 3D).
/// </summary>
public static class Dinov2HubKeys
{
    private static readonly string[] QkvNames = ["query", "key", "value"];

    /// <summary>The encoder's weights by HF name.</summary>
    /// <param name="weights">The hub tensors by name.</param>
    /// <param name="prefix">Everything before <c>patch_embed.proj.weight</c> and the other keys, such as <c>pretrained.</c>; empty for none.</param>
    /// <param name="preset">The network the weights belong to (hidden size, layer count, register count, patch size, position count).</param>
    /// <param name="requireF32">Whether every tensor must already be F32 (a module that reads raw spans); the encoder widens other dtypes itself otherwise.</param>
    /// <exception cref="InvalidOperationException">A weight is missing, has the wrong shape, or is not F32 while <paramref name="requireF32"/> is set.</exception>
    public static Dictionary<string, Tensor> Remap(IReadOnlyDictionary<string, Tensor> weights, string prefix, Dinov2Preset preset, bool requireF32 = false)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(prefix);
        ArgumentNullException.ThrowIfNull(preset);
        int d = preset.HiddenSize, ff = preset.IntermediateSize, patch = preset.PatchSize;
        Tensor Get(string key, params long[] shape)
        {
            if (!weights.TryGetValue(prefix + key, out Tensor? tensor))
                throw new InvalidOperationException($"DINOv2 weight '{prefix + key}' is missing.");
            if (requireF32 && tensor.DType != DType.F32)
                throw new InvalidOperationException($"DINOv2 weight '{prefix + key}' is {tensor.DType}; the module takes F32 weights, so widen 16-bit weights first.");
            if (!tensor.Shape.Equals(new TensorShape(shape)))
                throw new InvalidOperationException($"DINOv2 weight '{prefix + key}' has shape {tensor.Shape}, expected {new TensorShape(shape)}.");
            return tensor;
        }

        Dictionary<string, Tensor> result = new(StringComparer.Ordinal)
        {
            ["embeddings.patch_embeddings.projection.weight"] = Get("patch_embed.proj.weight", d, 3, patch, patch),
            ["embeddings.patch_embeddings.projection.bias"] = Get("patch_embed.proj.bias", d),
            ["embeddings.cls_token"] = Get("cls_token", 1, 1, d),
            ["embeddings.position_embeddings"] = Get("pos_embed", 1, 1 + preset.NumPatches, d),
            ["layernorm.weight"] = Get("norm.weight", d),
            ["layernorm.bias"] = Get("norm.bias", d),
        };
        if (preset.NumRegisterTokens > 0)
            result["embeddings.register_tokens"] = Get("register_tokens", 1, preset.NumRegisterTokens, d);

        for (int i = 0; i < preset.NumLayers; i++)
        {
            string hub = $"blocks.{i}.", hf = $"encoder.layer.{i}.";
            result[hf + "norm1.weight"] = Get(hub + "norm1.weight", d);
            result[hf + "norm1.bias"] = Get(hub + "norm1.bias", d);
            result[hf + "norm2.weight"] = Get(hub + "norm2.weight", d);
            result[hf + "norm2.bias"] = Get(hub + "norm2.bias", d);
            result[hf + "attention.output.dense.weight"] = Get(hub + "attn.proj.weight", d, d);
            result[hf + "attention.output.dense.bias"] = Get(hub + "attn.proj.bias", d);
            result[hf + "layer_scale1.lambda1"] = Get(hub + "ls1.gamma", d);
            result[hf + "layer_scale2.lambda1"] = Get(hub + "ls2.gamma", d);
            result[hf + "mlp.fc1.weight"] = Get(hub + "mlp.fc1.weight", ff, d);
            result[hf + "mlp.fc1.bias"] = Get(hub + "mlp.fc1.bias", ff);
            result[hf + "mlp.fc2.weight"] = Get(hub + "mlp.fc2.weight", d, ff);
            result[hf + "mlp.fc2.bias"] = Get(hub + "mlp.fc2.bias", d);
            Tensor qkvWeight = Get(hub + "attn.qkv.weight", 3 * d, d);
            Tensor qkvBias = Get(hub + "attn.qkv.bias", 3 * d);
            for (int part = 0; part < 3; part++)
            {
                result[$"{hf}attention.attention.{QkvNames[part]}.weight"] = qkvWeight.SliceRows(part * d, d);
                result[$"{hf}attention.attention.{QkvNames[part]}.bias"] = qkvBias.SliceRows(part * d, d);
            }
        }

        return result;
    }
}
