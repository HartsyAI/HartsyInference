namespace HartsyInference.LLM.ChatTemplates;

/// <summary>A tool call emitted by an assistant turn. <paramref name="ArgumentsJson"/> is the JSON argument text; a qualified <c>namespace::name</c> Name or <see cref="Namespace"/> selects the tool namespace.</summary>
public sealed record ChatToolCall(string Id, string Name, string ArgumentsJson)
{
    /// <summary>Explicit tool namespace; must agree with any prefix already present in <see cref="Name"/>.</summary>
    public string? Namespace { get; init; }
}
