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
        // Read only once a group without its own blob needs it, so a file with no NVFP4 never parses the marker.
        Lazy<HashSet<string>?> fileLayers = new(() => FileLevelLayers(metadata));
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
            if (weights.ContainsKey(stem + ".comfy_quant") || fileLayers.Value?.Contains(stem) == true) continue;
            // Shape cannot tell the layouts apart once rows are a multiple of 128 and K/16 of 4, so the missing config is
            // the only evidence: a Comfy group without one would be scrambled here.
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

    /// <summary>The layer names a file-level <c>_quantization_metadata</c> lists, or null when the file has none. A marker
    /// that is present but unreadable throws: read as absent, it would turn a Comfy-layout file into a ModelOpt one.</summary>
    internal static HashSet<string>? FileLevelLayers(IReadOnlyDictionary<string, string>? metadata)
    {
        if (metadata is null || !metadata.TryGetValue(MetadataKey, out string? json)) return null;
        return [.. LayerMap(json).Select(p => p.Key)];
    }

    /// <summary>Every listed layer with its config, or null when the file has no marker.</summary>
    internal static IReadOnlyList<KeyValuePair<string, System.Text.Json.Nodes.JsonNode?>>? LayerConfigs(
        IReadOnlyDictionary<string, string>? metadata)
        => metadata is null || !metadata.TryGetValue(MetadataKey, out string? json) ? null : LayerMap(json);

    /// <summary>One marker for a shard set: each shard lists only its own layers, so the maps are united.</summary>
    internal static string MergeMarkers(string first, string next)
    {
        System.Text.Json.Nodes.JsonObject merged = Parse(first);
        System.Text.Json.Nodes.JsonObject layers = (System.Text.Json.Nodes.JsonObject)merged["layers"]!;
        foreach ((string name, System.Text.Json.Nodes.JsonNode? config) in LayerMap(next))
            layers[name] = config?.DeepClone();
        return merged.ToJsonString();
    }

    internal const string MetadataKey = "_quantization_metadata";

    private static List<KeyValuePair<string, System.Text.Json.Nodes.JsonNode?>> LayerMap(string json)
        => [.. ((System.Text.Json.Nodes.JsonObject)Parse(json)["layers"]!)];

    private static System.Text.Json.Nodes.JsonObject Parse(string json)
    {
        System.Text.Json.Nodes.JsonNode? root;
        try
        {
            root = System.Text.Json.Nodes.JsonNode.Parse(json);
        }
        catch (System.Text.Json.JsonException error)
        {
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException(
                $"The checkpoint's {MetadataKey} is not valid JSON ({error.Message}), so which NVFP4 groups are Comfy layout is unknown.");
        }
        if (root is not System.Text.Json.Nodes.JsonObject obj || obj["layers"] is not System.Text.Json.Nodes.JsonObject)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException(
                $"The checkpoint's {MetadataKey} has no \"layers\" object, so which NVFP4 groups are Comfy layout is unknown.");
        return obj;
    }
}
