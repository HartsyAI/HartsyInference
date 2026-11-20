using System.Text.Json.Serialization;

namespace HartsyInference.ModelAssets.Metadata;

/// <summary>What shapes the prompt text a model takes, for a consumer that rewrites a user's prompt into the form the
/// model wants. Generation settings (steps, guidance, size) are not here: the recipe's defaults own those.
/// <para>Every field except <see cref="Kind"/> is optional, and an unset field means the fact was not established,
/// never "no". <see cref="Supports"/> lists only what is affirmed, a capability the source calls weak or limited
/// included, so a consumer must use nothing outside the list; null means no source assessed the model. The facts come
/// from the reviewed per-model prompting guides, and <see cref="Guide"/> names the one that holds the caveats.</para>
/// <para>Written into a checkpoint header as flat <c>modelspec.prompt_*</c> string keys (see <see cref="ToModelSpecKeys"/>),
/// which SwarmUI keeps with the rest of the model's metadata.</para></summary>
public sealed record PromptProfile
{
    private const string KeyPrefix = "modelspec.prompt_";

    /// <summary>Header key for <see cref="Format"/>.</summary>
    public const string FormatKey = KeyPrefix + "format";

    /// <summary>Header key for <see cref="Negative"/>.</summary>
    public const string NegativeKey = KeyPrefix + "negative";

    /// <summary>Header key for <see cref="Supports"/>, a comma list.</summary>
    public const string SupportsKey = KeyPrefix + "supports";

    /// <summary>Header key for <see cref="Boxes"/>.</summary>
    public const string BoxesKey = KeyPrefix + "boxes";

    /// <summary>Header key for <see cref="Encoders"/>, a comma list.</summary>
    public const string EncoderKey = KeyPrefix + "encoder";

    /// <summary>Whether this is an image or a video model: one of <see cref="PromptVocabulary.Kinds"/>.</summary>
    public required string Kind { get; init; }

    /// <summary>The text encoders the prompt reaches, in the order the guide lists them; each one of <see cref="PromptVocabulary.Encoders"/>.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public IReadOnlyList<string>? Encoders { get; init; }

    /// <summary>The prompt format the model wants, one of <see cref="PromptVocabulary.Formats"/>. <c>text</c> covers prose
    /// that may carry an optional trailing structure; <c>json</c> is for a model trained only on structured captions.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Format { get; init; }

    /// <summary>How the model takes a negative prompt, one of <see cref="PromptVocabulary.Negatives"/>. <c>none</c> means
    /// the model has no negative branch (distilled or embedded guidance); <c>separate</c> means a CFG negative branch
    /// exists, whatever guidance a request then sets.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Negative { get; init; }

    /// <summary>What the model honours in a prompt, from <see cref="PromptVocabulary.Supports"/>. Distinct from the recipe's
    /// <c>PromptWeighting</c> and regional features, which say what the engine's pipeline applies.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public IReadOnlyList<string>? Supports { get; init; }

    /// <summary>The box coordinate form as <c>{axes}-{scale}</c>, e.g. <c>yxyx-1000</c> for <c>[y_min, x_min, y_max, x_max]</c>
    /// on a 0-1000 grid. Set only when <see cref="Supports"/> has <c>boxes</c> and the guide states the form.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Boxes { get; init; }

    /// <summary>The longest prompt the model reads, in tokens, when its own card or README states one.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? MaxTokens { get; init; }

    /// <summary>The prompting guide for this model, as its path in the SwarmUI-LLMAssistant repo
    /// (<c>docs/prompting/&lt;name&gt;.md</c>); null when no guide covers the model.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Guide { get; init; }

    /// <summary>The flat <c>modelspec.prompt_*</c> header entries for the fields that are set. <see cref="Kind"/>,
    /// <see cref="MaxTokens"/> and <see cref="Guide"/> have no key: the header's tags already carry the category.</summary>
    public IReadOnlyDictionary<string, string> ToModelSpecKeys()
    {
        Dictionary<string, string> keys = new(StringComparer.Ordinal);
        if (Format is not null)
        {
            keys[FormatKey] = Format;
        }
        if (Negative is not null)
        {
            keys[NegativeKey] = Negative;
        }
        if (Supports is not null)
        {
            keys[SupportsKey] = string.Join(",", Supports);
        }
        if (Boxes is not null)
        {
            keys[BoxesKey] = Boxes;
        }
        if (Encoders is not null)
        {
            keys[EncoderKey] = string.Join(",", Encoders);
        }
        return keys;
    }

    /// <summary>Every way this profile leaves the vocabulary or contradicts itself; empty when it is valid.</summary>
    public IReadOnlyList<string> Problems()
    {
        List<string> problems = [];
        Expect(Kind, PromptVocabulary.Kinds, "kind", problems);
        ExpectAll(Encoders, PromptVocabulary.Encoders, "encoders", problems);
        Expect(Format, PromptVocabulary.Formats, "format", problems);
        Expect(Negative, PromptVocabulary.Negatives, "negative", problems);
        ExpectAll(Supports, PromptVocabulary.Supports, "supports", problems);
        if (Boxes is not null)
        {
            int dash = Boxes.IndexOf('-', StringComparison.Ordinal);
            bool formed = dash > 0 && PromptVocabulary.BoxAxes.Contains(Boxes[..dash])
                && int.TryParse(Boxes[(dash + 1)..], out int scale) && scale > 0;
            if (!formed)
            {
                problems.Add($"boxes '{Boxes}' is not '<yxyx|xyxy>-<positive scale>'.");
            }
            if (Supports is null || !Supports.Contains(PromptVocabulary.Boxes))
            {
                problems.Add("boxes is set but supports does not list 'boxes'.");
            }
        }
        if (MaxTokens is <= 0)
        {
            problems.Add("maxTokens must be positive.");
        }
        if (Guide is not null && string.IsNullOrWhiteSpace(Guide))
        {
            problems.Add("guide is blank.");
        }
        return problems;
    }

    private static void Expect(string? value, IReadOnlyList<string> allowed, string field, List<string> problems)
    {
        if (value is not null && !allowed.Contains(value))
        {
            problems.Add($"{field} '{value}' is not one of {string.Join("|", allowed)}.");
        }
    }

    private static void ExpectAll(IReadOnlyList<string>? values, IReadOnlyList<string> allowed, string field, List<string> problems)
    {
        if (values is null)
        {
            return;
        }
        HashSet<string> seen = new(StringComparer.Ordinal);
        foreach (string value in values)
        {
            Expect(value, allowed, field, problems);
            if (!seen.Add(value))
            {
                problems.Add($"{field} lists '{value}' twice.");
            }
        }
    }
}
