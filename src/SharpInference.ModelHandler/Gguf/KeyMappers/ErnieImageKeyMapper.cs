namespace SharpInference.ModelHandler.Gguf.KeyMappers;

/// <summary>ERNIE-Image (Baidu) GGUF mapper. ERNIE-Image GGUFs (e.g. `unsloth/ERNIE-Image-GGUF`) ship with the diffusers `transformer.*` naming. The architecture's distinguishing features are the **shared AdaLN modulation** (single top-level linear, broadcast across all 36 layers) and the patch_embed Conv2d with kernel=1.</summary>
public sealed class ErnieImageKeyMapper : IGgufKeyMapper
{
    public string Architecture => "ernie_image";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        bool hasSharedAdaLN = false;
        bool hasErnieBlocks = false;
        foreach (string name in tensorNames)
        {
            if (name.Contains("shared_adaLN_modulation.", StringComparison.Ordinal)) hasSharedAdaLN = true;
            if (name.Contains("transformer_blocks.", StringComparison.Ordinal) && name.Contains(".mlp.gate_proj.", StringComparison.Ordinal))
                hasErnieBlocks = true;
            if (hasSharedAdaLN && hasErnieBlocks) return true;
        }
        return false;
    }

    public string? MapKey(string ggufKey) => ggufKey;
}
