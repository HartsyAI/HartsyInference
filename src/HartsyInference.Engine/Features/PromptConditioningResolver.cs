namespace HartsyInference.Engine.Features;

/// <summary>Strips the structural <c>&lt;tag&gt;</c> layer off a prompt before it reaches the tokenizer, so region/stage
/// tag characters don't leak into the base conditioning. Encoder-level syntax the engine itself honors — weighting
/// <c>(x:1.3)</c>, alternation <c>[a|b]</c>, <c>&lt;break&gt;</c>, embed markers — is deliberately left intact.</summary>
public static class PromptConditioningResolver
{
    /// <summary>The base-stage text: the prompt's global text with structural tags removed. A tagless prompt is returned byte-identical.</summary>
    public static string BaseText(string? raw)
    {
        if (string.IsNullOrEmpty(raw))
        {
            return raw ?? "";
        }
        if (!raw.Contains('<', StringComparison.Ordinal))
        {
            return raw;
        }
        return new PromptRegionParser(raw).GlobalPrompt;
    }

    /// <summary>The video-stage text: the <c>&lt;video&gt;</c> sub-prompt when present (image-to-video's alternate motion
    /// prompt), else the tag-stripped global text.</summary>
    public static string VideoText(string? raw)
    {
        if (string.IsNullOrEmpty(raw))
        {
            return raw ?? "";
        }
        if (!raw.Contains('<', StringComparison.Ordinal))
        {
            return raw;
        }
        PromptRegionParser region = new PromptRegionParser(raw);
        return !string.IsNullOrWhiteSpace(region.VideoPrompt) ? region.VideoPrompt : region.GlobalPrompt;
    }
}
