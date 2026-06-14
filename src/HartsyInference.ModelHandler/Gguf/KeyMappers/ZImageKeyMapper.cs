namespace HartsyInference.ModelHandler.Gguf.KeyMappers;

/// <summary>Z-Image (Lumina2/NextDiT) GGUF mapper. Tongyi-Lab dumps ship with the single-file naming with <c>model.diffusion_model.</c> wrapper or the <c>transformer.</c> wrapper that <see cref="HartsyInference.ModelHandler.CheckpointConverters.ZImageCheckpointConverter.Convert"/> already strips.</summary>
public sealed class ZImageKeyMapper : IGgufKeyMapper
{
    public string Architecture => "zimage";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        bool hasNoiseRefiner = false, hasContextRefiner = false;
        foreach (string name in tensorNames)
        {
            if (name.Contains("noise_refiner.", StringComparison.Ordinal)) hasNoiseRefiner = true;
            if (name.Contains("context_refiner.", StringComparison.Ordinal)) hasContextRefiner = true;
            if (hasNoiseRefiner && hasContextRefiner) return true;
        }
        return false;
    }

    public string? MapKey(string ggufKey) => ggufKey;
}
