using System.Globalization;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Per-call settings for an <see cref="IConversationEncoder"/>.</summary>
public sealed record EncodeOptions
{
    /// <summary>Highest accepted numeric reasoning effort.</summary>
    public const int MaxReasoningEffort = 100;

    /// <summary>Appends the assistant header (or task token) after a final user-side message.</summary>
    public bool AddGenerationPrompt { get; init; } = true;

    /// <summary>Thinking mode; false renders chat mode.</summary>
    public bool Thinking { get; init; }

    /// <summary>Reasoning effort in [1, 100] (thinking mode only); null means the default of 75.</summary>
    public int? ReasoningEffort { get; init; }

    /// <summary>Tool schemas rendered into the first message; overrides tools set on the first message itself.</summary>
    public IReadOnlyList<ToolSpec>? Tools { get; init; }

    /// <summary>Drops earlier assistant reasoning in thinking mode; null means true. Ignored (forced off) when any message defines tools.</summary>
    public bool? DropThinking { get; init; }

    /// <summary>Patch grids of the images referenced by <see cref="ImageBlock"/>s, indexed by <see cref="ImageBlock.ImageIndex"/>.</summary>
    public IReadOnlyList<ImageGrid>? Images { get; init; }

    /// <summary>Parses a reasoning effort: a named effort ("low"/"high"/"max", which are 50/75/100) or an integer in [1, <see cref="MaxReasoningEffort"/>]
    /// written in plain digits. The CLI and the chat API both read effort through it, so they accept and refuse the same values.</summary>
    /// <exception cref="ArgumentException">Any other name, or an integer outside the range, including one too large for an <see cref="int"/>.</exception>
    public static int ParseReasoningEffort(string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        switch (value)
        {
            case "low": return 50;
            case "high": return 75;
            case "max": return 100;
        }
        if (value.Length == 0 || value.AsSpan().ContainsAnyExceptInRange('0', '9'))
            throw new ArgumentException($"Unknown reasoning effort '{value}'; use low, high, max or an integer in [1, {MaxReasoningEffort}].");
        // Digits only: an integer, perhaps one too large for an int, which is out of range all the same.
        if (int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int effort) && effort is >= 1 and <= MaxReasoningEffort)
            return effort;
        throw new ArgumentException($"Reasoning effort {value} is out of range; use an integer in [1, {MaxReasoningEffort}], or low, high or max.");
    }
}
