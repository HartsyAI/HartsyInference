namespace HartsyInference.Tools.Parsing;

/// <summary>Everything the parser needs to read one <see cref="ToolCallFormat"/>: the markers that open a span, the optional closing marker consumed after a completed call, whether an identifier followed by <c>{</c> at the start of a line opens a call (Mistral's <c>name{…}</c>), and which JSON keys carry the arguments, in preference order.</summary>
public sealed record ToolCallFormatRules
{
    /// <summary>The format these rules describe.</summary>
    public required ToolCallFormat Format { get; init; }

    /// <summary>Opening markers, checked in order; a one-character marker matches immediately, longer ones are held until they match or diverge.</summary>
    public required IReadOnlyList<ToolCallMarker> Markers { get; init; }

    /// <summary>Closing marker swallowed (with the whitespace before it) after a completed call; null when the format has none or the call simply ends at the balanced value.</summary>
    public string? CloseMarker { get; init; }

    /// <summary>True when an identifier immediately followed by <c>{</c> at the start of a line opens a call whose name is that identifier.</summary>
    public bool NamedFormAtLineStart { get; init; }

    /// <summary>JSON keys that may carry the arguments object (<c>arguments</c>, <c>parameters</c>), first match wins.</summary>
    public required IReadOnlyList<string> ArgumentKeys { get; init; }
}
