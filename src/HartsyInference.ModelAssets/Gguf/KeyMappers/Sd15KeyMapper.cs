namespace HartsyInference.ModelAssets.Gguf.KeyMappers;

/// <summary>SD 1.5 GGUF mapper. SD15 GGUFs use the LDM naming (<c>model.diffusion_model.input_blocks.X.Y.weight</c> etc.) that <see cref="HartsyInference.ModelAssets.CheckpointConverters.Sd15CheckpointConverter.Convert"/> already handles.</summary>
public sealed class Sd15KeyMapper : IGgufKeyMapper
{
    public string Architecture => "sd15";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        bool hasInputBlocks = false, hasNoLabelEmb = true;
        foreach (string name in tensorNames)
        {
            if (name.Contains("input_blocks.", StringComparison.Ordinal)) hasInputBlocks = true;
            if (name.Contains("label_emb.", StringComparison.Ordinal)) hasNoLabelEmb = false;
        }
        return hasInputBlocks && hasNoLabelEmb;
    }

    public string? MapKey(string ggufKey) => ggufKey;
}
