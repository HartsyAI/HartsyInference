namespace HartsyInference.ModelAssets.Gguf.KeyMappers;

/// <summary>Chroma Radiance GGUF mapper. Radiance is the Chroma family (BFL naming with <c>distilled_guidance_layer.*</c>) plus the pixel-space IO ends: <c>img_in_patch.*</c> conv patchify and the <c>nerf_blocks.*</c> NeRF head. Must be probed BEFORE <see cref="ChromaKeyMapper"/> in heuristic detection — every Radiance checkpoint also matches the classic-Chroma key signature. <see cref="HartsyInference.ModelAssets.CheckpointConverters.ChromaCheckpointConverter.Convert"/> passes the Radiance-only keys through verbatim.</summary>
public sealed class ChromaRadianceKeyMapper : IGgufKeyMapper
{
    public string Architecture => "chroma-radiance";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        bool hasDistilledGuidance = false, hasNerfHead = false;
        foreach (string name in tensorNames)
        {
            if (name.Contains("distilled_guidance_layer.", StringComparison.Ordinal)) hasDistilledGuidance = true;
            if (name.Contains("nerf_blocks.", StringComparison.Ordinal)) hasNerfHead = true;
            if (hasDistilledGuidance && hasNerfHead) return true;
        }
        return false;
    }

    public string? MapKey(string ggufKey) => ggufKey;
}
