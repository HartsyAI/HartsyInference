namespace HartsyInference.ModelHandler.Gguf.KeyMappers;

/// <summary>Zeta-Chroma GGUF mapper. Despite the name, Zeta-Chroma is the Z-Image S3-DiT (NextDiT single-file
/// naming: <c>layers.*</c>, <c>noise_refiner.*</c>, <c>context_refiner.*</c>) retrained for pixel space, with a
/// DeCo-style <c>dec_net.*</c> decoder head replacing <c>final_layer</c>. Must be probed BEFORE
/// <see cref="ZImageKeyMapper"/> in heuristic detection — every Zeta checkpoint also matches the Z-Image signature.
/// Keys pass through unchanged; <see cref="HartsyInference.ModelHandler.CheckpointConverters.ZetaChromaCheckpointConverter"/>
/// reuses the Z-Image partitioner.</summary>
public sealed class ZetaChromaKeyMapper : IGgufKeyMapper
{
    public string Architecture => "zeta-chroma";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        bool hasRefiner = false, hasDecoder = false;
        foreach (string name in tensorNames)
        {
            if (name.Contains("noise_refiner.", StringComparison.Ordinal)) hasRefiner = true;
            if (name.Contains("dec_net.", StringComparison.Ordinal)) hasDecoder = true;
            if (hasRefiner && hasDecoder) return true;
        }
        return false;
    }

    public string? MapKey(string ggufKey) => ggufKey;
}
