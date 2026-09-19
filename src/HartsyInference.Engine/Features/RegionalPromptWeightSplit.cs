using HartsyInference.Diffusion.Prompting;

namespace HartsyInference.Engine.Features;

/// <summary>Who owns the emphasis in a prompt that also carries region tags.
/// <para>A recipe that declares a weighting mode encodes its base prompt from the WHOLE string, region tags
/// included — that is what it has always done, and the region conditioning is appended to that stream rather than
/// replacing it. So a region's own <c>(word:N)</c> shows up in the base weights too, where it would scale rows
/// belonging to the tag text. Splitting the question is what lets a weighted region work while a base that is
/// genuinely weighted alongside one is refused rather than quietly losing its emphasis.</para></summary>
public static class RegionalPromptWeightSplit
{
    /// <summary>Whether the text OUTSIDE every region tag carries an emphasis. The region parser hands the
    /// global/base/background text back separately, so this asks about the part the base encode would weight
    /// rather than about the prompt as a whole.</summary>
    public static bool BaseTextCarriesWeight(string? prompt)
    {
        if (string.IsNullOrEmpty(prompt))
        {
            return false;
        }
        PromptRegionParser parsed = new PromptRegionParser(prompt);
        foreach (string text in (string[])[parsed.GlobalPrompt, parsed.BasePrompt, parsed.BackgroundPrompt])
        {
            if (PromptWeighting.HasWeights(PromptWeighting.Parse(PromptTagFlattening.Flatten(text))))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>The base prompt text to tokenize, with the weight grammar taken off in BOTH its spellings when
    /// regions are present.</summary>
    /// <remarks>Flattening alone is not enough. It rewrites SwarmUI's <c>&lt;weight[N]:&gt;</c> tags, but a literal
    /// <c>(word:N)</c> typed at a CLI is Comfy grammar that reaches the recipe untouched, and
    /// <see cref="WeightedTokenBuilder"/> splits on it either way — so a region-only emphasis would still produce
    /// base-prompt weights. <c>Join</c> collapses both.</remarks>
    public static string BaseText(string? prompt, bool hasRegionParts) => hasRegionParts
        ? PromptWeighting.Join(PromptWeighting.Parse(PromptTagFlattening.Flatten(prompt, weightsAsParens: false)))
        : PromptTagFlattening.Flatten(prompt);
}
