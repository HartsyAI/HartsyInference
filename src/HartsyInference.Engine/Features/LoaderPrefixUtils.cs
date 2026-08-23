using HartsyInference.Core.Tensors;

namespace HartsyInference.Engine.Features;

/// <summary>Key normalization for standalone text-encoder safetensors files: unwraps whichever container prefix the publisher wrapped the checkpoint in, and optionally drops non-weight buffers.</summary>
internal static class LoaderPrefixUtils
{
    /// <summary>HF's <c>position_ids</c> buffer — a derivable arange that ships in CLIP checkpoints and is read by no encoder here (T5 uses relative position bias instead).</summary>
    internal static readonly IReadOnlyList<string> PositionIdsBuffer = ["position_ids"];

    /// <summary>Comfy's wrapper prefix for a standalone T5-XXL text-encoder file.</summary>
    internal const string ComfyT5XxlPrefix = "text_encoders.t5xxl.transformer.";

    /// <summary>Unwraps a standalone T5-XXL safetensors file from Comfy's container prefix so the encoder finds its keys.</summary>
    internal static Dictionary<string, Tensor> StripT5XxlPrefix(IReadOnlyDictionary<string, Tensor> raw) =>
        StripPrefixes(raw, [ComfyT5XxlPrefix], PositionIdsBuffer);

    /// <summary>Strips the first matching entry of <paramref name="prefixes"/> off each key and drops keys whose stripped form ends with any <paramref name="dropSuffixes"/> entry; keys matching no prefix are kept under the same drop rule.</summary>
    internal static Dictionary<string, Tensor> StripPrefixes(
        IReadOnlyDictionary<string, Tensor> raw,
        IReadOnlyList<string> prefixes,
        IReadOnlyList<string>? dropSuffixes = null)
    {
        ArgumentNullException.ThrowIfNull(raw);
        ArgumentNullException.ThrowIfNull(prefixes);
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(raw.Count);
        foreach (KeyValuePair<string, Tensor> kv in raw)
        {
            string key = kv.Key;
            for (int i = 0; i < prefixes.Count; i++)
            {
                if (key.StartsWith(prefixes[i], StringComparison.Ordinal))
                {
                    key = key[prefixes[i].Length..];
                    break;
                }
            }
            if (dropSuffixes is not null && EndsWithAny(key, dropSuffixes))
            {
                continue;
            }
            result[key] = kv.Value;
        }
        return result;
    }

    private static bool EndsWithAny(string key, IReadOnlyList<string> suffixes)
    {
        for (int i = 0; i < suffixes.Count; i++)
        {
            if (key.EndsWith(suffixes[i], StringComparison.Ordinal))
            {
                return true;
            }
        }
        return false;
    }
}
