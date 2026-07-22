namespace HartsyInference.ModelAssets.Gguf.KeyMappers;

/// <summary>AuraFlow GGUF mapper. <c>city96/AuraFlow-v0.3-gguf</c> ships with BFL-style naming with <c>double_layers.*</c> and <c>single_layers.*</c> prefixes — <see cref="HartsyInference.ModelAssets.CheckpointConverters.AuraFlowCheckpointConverter.Convert"/> handles the BFL→diffusers remap.</summary>
public sealed class AuraFlowKeyMapper : IGgufKeyMapper
{
    public string Architecture => "auraflow";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        foreach (string name in tensorNames)
        {
            if (name.Contains("double_layers.", StringComparison.Ordinal)) return true;
            if (name.Contains("modF.", StringComparison.Ordinal)) return true;
            if (name.Contains("cond_seq_linear.", StringComparison.Ordinal)) return true;
        }
        return false;
    }

    public string? MapKey(string ggufKey) => ggufKey;
}
