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

    /// <summary>Maps the named efforts "low"/"high"/"max" to 50/75/100.</summary>
    public static int ParseReasoningEffort(string name) => name switch
    {
        "low" => 50,
        "high" => 75,
        "max" => 100,
        _ => throw new ArgumentException($"Unknown reasoning effort '{name}'; use low, high, max or an integer in [1,100]."),
    };
}
