using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelHandler.CheckpointConverters;

/// <summary>Splits a TripoSR checkpoint into the three component weight dicts the pipeline loads: the DINO
/// image tokenizer (<c>Dino</c>), the image→triplane <c>Transformer</c>, and the triplane NeRF
/// <c>Decoder</c>. Follows the project's converter pattern (route keys by prefix → strip prefix).
/// <para><b>Numerics validation-pending</b> — prefix set and per-key tables are <b>validation-gated</b>,
/// finalized against the reference during the diff pass.</para></summary>
public static class TripoSrCheckpointConverter
{
    public sealed class ConvertedWeights
    {
        public required Dictionary<string, Tensor> Dino { get; init; }
        public required Dictionary<string, Tensor> Transformer { get; init; }
        public required Dictionary<string, Tensor> Decoder { get; init; }
    }

    private static readonly string[] DinoPrefixes = ["image_tokenizer.", "backbone.", "dino.", "image_encoder."];
    private static readonly string[] DecoderPrefixes = ["decoder.", "renderer.", "nerf.", "mlp_decoder."];
    private static readonly string[] TransformerPrefixes = ["backbone_transformer.", "tokenizer.", "transformer.", "triplane."];

    public static ConvertedWeights Convert(IReadOnlyDictionary<string, Tensor> all)
    {
        ArgumentNullException.ThrowIfNull(all);
        Dictionary<string, Tensor> dino = new(), tr = new(), dec = new();
        foreach ((string key, Tensor t) in all)
        {
            if (TryRoute(key, t, DinoPrefixes, dino)) continue;
            if (TryRoute(key, t, DecoderPrefixes, dec)) continue;
            // Default → transformer (strip a transformer prefix if present).
            tr[StripFirst(key, TransformerPrefixes)] = t;
        }
        return new ConvertedWeights { Dino = dino, Transformer = tr, Decoder = dec };
    }

    private static bool TryRoute(string key, Tensor t, string[] prefixes, Dictionary<string, Tensor> dst)
    {
        foreach (string p in prefixes)
            if (key.StartsWith(p, StringComparison.Ordinal)) { dst[key[p.Length..]] = t; return true; }
        return false;
    }

    private static string StripFirst(string key, string[] prefixes)
    {
        foreach (string p in prefixes)
            if (key.StartsWith(p, StringComparison.Ordinal)) return key[p.Length..];
        return key;
    }
}
