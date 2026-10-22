using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Nvfp4;

/// <summary>Brings every NVFP4 group a checkpoint carries to the Comfy layout before anything reads it. A group with a
/// ComfyUI layer config (its own <c>.comfy_quant</c> blob, or an entry in the file-level <c>_quantization_metadata</c>) is
/// Comfy layout: ComfyUI will not treat a weight as nvfp4 without one. A group with neither is ModelOpt layout, which is
/// what NVIDIA's ModelOpt exports and this repo's repacker wrote before it emitted Comfy layout; decoding it as Comfy
/// layout scores about 0.00 against the source weights, the right decode about 0.995.</summary>
internal static class Nvfp4LayoutNormalizer
{
    /// <summary>Replaces, in <paramref name="weights"/>, each config-less group's packed weight and block scale with
    /// Comfy-layout copies and returns those copies, which the caller owns. Groups whose scale tensor is not ModelOpt's
    /// <c>[N, K/16]</c> are left alone.</summary>
    public static List<Tensor> ConvertModelOptGroups(Dictionary<string, Tensor> weights, IReadOnlyDictionary<string, string>? metadata)
    {
        HashSet<string>? fileLayers = FileLevelLayers(metadata);
        List<Tensor> converted = [];
        int groups = 0;
        foreach (string key in weights.Keys.ToList())
        {
            if (!key.EndsWith(".weight", StringComparison.Ordinal)) continue;
            Tensor packed = weights[key];
            if (packed.DType != DType.U8 || packed.Shape.Rank != 2) continue;
            string stem = key[..^".weight".Length];
            if (!weights.TryGetValue(stem + ".weight_scale", out Tensor? scale) || scale.DType != DType.F8E4M3
                || scale.Shape.Rank != 2 || !weights.TryGetValue(stem + ".weight_scale_2", out Tensor? global)
                || global.DType != DType.F32)
            {
                continue;
            }
            if (weights.ContainsKey(stem + ".comfy_quant") || fileLayers?.Contains(stem) == true) continue;
            if (scale.Shape[0] != packed.Shape[0] || scale.Shape[1] * Nvfp4Codec.GroupSize != packed.Shape[1] * 2) continue;

            (Tensor comfyPacked, Tensor comfyScale) = Nvfp4Codec.FromModelOptLayout(packed, scale);
            weights[key] = comfyPacked;
            weights[stem + ".weight_scale"] = comfyScale;
            converted.Add(comfyPacked);
            converted.Add(comfyScale);
            groups++;
        }
        if (groups > 0)
            Logs.Info($"Nvfp4 layout: {groups} groups carry no ComfyUI layer config and were read as ModelOpt layout "
                + "(row-major scales, low nibble first).");
        return converted;
    }

    /// <summary>The layer names a file-level <c>_quantization_metadata</c> lists, or null when the file has none.</summary>
    private static HashSet<string>? FileLevelLayers(IReadOnlyDictionary<string, string>? metadata)
    {
        if (metadata is null || !metadata.TryGetValue("_quantization_metadata", out string? json) || string.IsNullOrEmpty(json))
            return null;
        try
        {
            using System.Text.Json.JsonDocument doc = System.Text.Json.JsonDocument.Parse(json);
            if (!doc.RootElement.TryGetProperty("layers", out System.Text.Json.JsonElement layers)
                || layers.ValueKind != System.Text.Json.JsonValueKind.Object)
                return null;
            return [.. layers.EnumerateObject().Select(p => p.Name)];
        }
        catch (System.Text.Json.JsonException)
        {
            return null;
        }
    }
}
