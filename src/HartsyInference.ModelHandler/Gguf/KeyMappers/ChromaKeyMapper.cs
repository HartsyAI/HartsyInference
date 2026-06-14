namespace HartsyInference.ModelHandler.Gguf.KeyMappers;

/// <summary>Chroma GGUF mapper. Chroma ships in BFL single-file naming with the additional <c>distilled_guidance_layer.*</c> tree. <see cref="HartsyInference.ModelHandler.CheckpointConverters.ChromaCheckpointConverter.Convert"/> already handles those keys.</summary>
public sealed class ChromaKeyMapper : IGgufKeyMapper
{
    public string Architecture => "chroma";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        bool hasDistilledGuidance = false, hasFluxBlocks = false;
        foreach (string name in tensorNames)
        {
            if (name.Contains("distilled_guidance_layer.", StringComparison.Ordinal)) hasDistilledGuidance = true;
            if (name.Contains("double_blocks.", StringComparison.Ordinal) || name.Contains("single_blocks.", StringComparison.Ordinal)) hasFluxBlocks = true;
            if (hasDistilledGuidance && hasFluxBlocks) return true;
        }
        return false;
    }

    public string? MapKey(string ggufKey) => ggufKey;
}
