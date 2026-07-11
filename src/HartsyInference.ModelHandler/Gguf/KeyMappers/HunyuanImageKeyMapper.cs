namespace HartsyInference.ModelHandler.Gguf.KeyMappers;

/// <summary>Hunyuan Image 2.1 GGUF mapper. Hunyuan-Image dumps follow the diffusers single-file convention with `transformer.*` blocks + 32-channel latent path.</summary>
public sealed class HunyuanImageKeyMapper : IGgufKeyMapper
{
    public string Architecture => "hunyuan_image";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        // Two shipped layouts: diffusers-style dumps (transformer_blocks + dual_attention/moe) and
        // original-Tencent GGUF repacks (double_blocks.*.img_attn_qkv + byt5_in — byt5_in distinguishes
        // HunyuanImage 2.1 from HunyuanVideo, which shares the double_blocks naming).
        bool hasTencentAttn = false, hasByt5 = false;
        foreach (string name in tensorNames)
        {
            if (name.Contains("transformer_blocks.", StringComparison.Ordinal) &&
                (name.Contains(".dual_attention", StringComparison.Ordinal) || name.Contains(".moe.", StringComparison.Ordinal)))
            {
                return true;
            }
            if (name.Contains("double_blocks.", StringComparison.Ordinal) && name.Contains("img_attn_qkv.", StringComparison.Ordinal))
                hasTencentAttn = true;
            if (name.StartsWith("byt5_in.", StringComparison.Ordinal))
                hasByt5 = true;
        }
        return hasTencentAttn && hasByt5;
    }

    public string? MapKey(string ggufKey) => ggufKey;
}
