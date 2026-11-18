using System.Globalization;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Haru;

/// <summary>Haru's <c>codec.*</c> tensors renamed to the layout the VibeVoice acoustic tokenizer classes read. The weights are the
/// same network under different names, so the shared classes load them unchanged.</summary>
internal static class HaruCodecWeights
{
    /// <summary>Key prefix of the remapped tensors, passed to <c>VibeVoiceAcousticTokenizerModel</c>.</summary>
    internal const string Prefix = "codec";

    /// <summary>Maps one checkpoint key (<c>codec.encoder.stages.3.1.fc1.weight</c>) to the engine's key; a key outside the codec's
    /// layout throws, so a renamed or extra tensor never loads silently.</summary>
    internal static string RemapKey(string key)
    {
        string[] p = key.Split('.');
        if (p.Length < 4 || p[0] != "codec" || p[1] is not ("encoder" or "decoder"))
        {
            throw new InvalidDataException($"'{key}' is not a Haru codec tensor.");
        }
        string side = $"{Prefix}.{p[1]}";
        bool encoder = p[1] == "encoder";
        string mapped = (p[2], p.Length) switch
        {
            ("stem", 4) => $"{side}.{(encoder ? "downsample" : "upsample")}_layers.0.0.{ConvLeaf(key, p[3])}",
            ("down", 5) when encoder => $"{side}.downsample_layers.{Index(key, p[3]) + 1}.0.{ConvLeaf(key, p[4])}",
            ("up", 5) when !encoder => $"{side}.upsample_layers.{Index(key, p[3]) + 1}.0.convtr.convtr.{Leaf(key, p[4])}",
            ("head", 4) => $"{side}.head.{ConvLeaf(key, p[3])}",
            ("stages", >= 6) => $"{side}.stages.{Index(key, p[3])}.{Index(key, p[4])}.{BlockLeaf(key, p[5..])}",
            _ => throw new InvalidDataException($"Unknown Haru codec tensor '{key}'."),
        };
        return mapped;
    }

    /// <summary>The same tensors under the engine's keys; the tensors stay owned by <paramref name="weights"/>.</summary>
    internal static Dictionary<string, Tensor> Remap(IReadOnlyDictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> mapped = new(StringComparer.Ordinal);
        foreach ((string key, Tensor tensor) in weights)
        {
            if (!key.StartsWith("codec.", StringComparison.Ordinal))
            {
                continue;
            }
            string target = RemapKey(key);
            if (!mapped.TryAdd(target, tensor))
            {
                throw new InvalidDataException($"Two Haru codec tensors map to '{target}'.");
            }
        }
        return mapped;
    }

    private static string ConvLeaf(string key, string leaf) => $"conv.conv.{Leaf(key, leaf)}";

    private static string Leaf(string key, string leaf) =>
        leaf is "weight" or "bias" ? leaf : throw new InvalidDataException($"Unknown Haru codec tensor '{key}'.");

    // Plain digits only: '01', ' 1' and '+1' would name the same layer as '1' under a different key.
    private static int Index(string key, string part) =>
        int.TryParse(part, NumberStyles.None, CultureInfo.InvariantCulture, out int index) && part.Length > 0 && (part.Length == 1 || part[0] != '0')
            ? index
            : throw new InvalidDataException($"Bad layer index in '{key}'.");

    private static string BlockLeaf(string key, string[] rest) => string.Join('.', rest) switch
    {
        "conv.weight" => "mixer.conv.conv.conv.weight",
        "conv.bias" => "mixer.conv.conv.conv.bias",
        "fc1.weight" => "ffn.linear1.weight",
        "fc1.bias" => "ffn.linear1.bias",
        "fc2.weight" => "ffn.linear2.weight",
        "fc2.bias" => "ffn.linear2.bias",
        "norm.weight" or "ffn_norm.weight" or "gamma" or "ffn_gamma" => string.Join('.', rest),
        _ => throw new InvalidDataException($"Unknown Haru codec tensor '{key}'."),
    };
}
