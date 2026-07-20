namespace HartsyInference.ModelAssets.Gguf.KeyMappers;

/// <summary>SDXL GGUF mapper. SDXL GGUFs (city96 / unsloth) ship with the LDM/CompVis naming that <see cref="HartsyInference.ModelAssets.CheckpointConverters.SdxlCheckpointConverter.Convert"/> already handles — keys like <c>model.diffusion_model.input_blocks.X.Y.weight</c>, <c>conditioner.embedders.0.transformer.text_model.*</c>, <c>first_stage_model.*</c>. Pass-through for now; add rewrites here if a GGUF builder uses an alternate prefix.</summary>
public sealed class SdxlKeyMapper : IGgufKeyMapper
{
    public string Architecture => "sdxl";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        bool hasInputBlocks = false, hasLabelEmb = false;
        foreach (string name in tensorNames)
        {
            if (name.Contains("input_blocks.", StringComparison.Ordinal)) hasInputBlocks = true;
            if (name.Contains("label_emb.", StringComparison.Ordinal)) hasLabelEmb = true;
            if (hasInputBlocks && hasLabelEmb) return true;
        }
        return false;
    }

    public string? MapKey(string ggufKey) => ggufKey;
}
