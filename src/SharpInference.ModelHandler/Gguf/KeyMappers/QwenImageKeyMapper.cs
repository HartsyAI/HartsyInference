namespace SharpInference.ModelHandler.Gguf.KeyMappers;

/// <summary>Qwen-Image GGUF mapper. The flagship `Qwen/Qwen-Image` (20B) is canonical diffusers single-stream MMDiT — keys like <c>transformer_blocks.{i}.attn.*</c> + <c>transformer_blocks.{i}.ff.*</c>.</summary>
public sealed class QwenImageKeyMapper : IGgufKeyMapper
{
    public string Architecture => "qwen_image";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        bool hasQwenJointBlocks = false;
        foreach (string name in tensorNames)
        {
            if (name.Contains("transformer_blocks.", StringComparison.Ordinal) &&
                name.Contains(".attn.add_q_proj.", StringComparison.Ordinal))
            {
                hasQwenJointBlocks = true;
                break;
            }
        }
        return hasQwenJointBlocks;
    }

    public string? MapKey(string ggufKey) => ggufKey;
}
