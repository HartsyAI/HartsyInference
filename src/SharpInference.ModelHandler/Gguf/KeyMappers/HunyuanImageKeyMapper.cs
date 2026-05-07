namespace SharpInference.ModelHandler.Gguf.KeyMappers;

/// <summary>Hunyuan Image 2.1 GGUF mapper. Hunyuan-Image dumps follow the diffusers single-file convention with `transformer.*` blocks + 32-channel latent path.</summary>
public sealed class HunyuanImageKeyMapper : IGgufKeyMapper
{
    public string Architecture => "hunyuan_image";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        bool hasHunyuanBlocks = false;
        foreach (string name in tensorNames)
        {
            if (name.Contains("transformer_blocks.", StringComparison.Ordinal) &&
                (name.Contains(".dual_attention", StringComparison.Ordinal) || name.Contains(".moe.", StringComparison.Ordinal)))
            {
                hasHunyuanBlocks = true;
                break;
            }
        }
        return hasHunyuanBlocks;
    }

    public string? MapKey(string ggufKey) => ggufKey;
}
