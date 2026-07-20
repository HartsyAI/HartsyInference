namespace HartsyInference.ModelAssets.Gguf.KeyMappers;

/// <summary>F-Lite GGUF mapper. F-Lite ships in diffusers folder format natively (each component in its own subfolder). If/when a community GGUF dump appears it will likely follow llama.cpp's <c>blk.{i}.</c> convention; we'll add the rewrite then. For now this mapper is passthrough + a key heuristic that recognizes F-Lite by its 16 register tokens parameter.</summary>
public sealed class FLiteKeyMapper : IGgufKeyMapper
{
    public string Architecture => "flite";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        bool hasRegisterTokens = false, hasFLiteBlocks = false;
        foreach (string name in tensorNames)
        {
            if (name.Equals("register_tokens", StringComparison.Ordinal)) hasRegisterTokens = true;
            if (name.Contains("blocks.", StringComparison.Ordinal) && name.Contains(".self_attn.", StringComparison.Ordinal)) hasFLiteBlocks = true;
        }
        return hasRegisterTokens && hasFLiteBlocks;
    }

    public string? MapKey(string ggufKey) => ggufKey;
}
