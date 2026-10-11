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

    /// <summary>Further control-token literals the format's span needs as text (for example Gemma's string delimiter), beyond its markers.</summary>
    public IReadOnlyList<string> ExtraLiterals { get; init; } = [];

    /// <summary>The control-token literals a filter for this format needs to see as text: its non-payload opening markers, its close marker and <see cref="ExtraLiterals"/>. The payload-only forms are plain text already.</summary>
    public IReadOnlyList<string> MarkerLiterals
    {
        get
        {
            List<string> literals = [];
            foreach (ToolCallMarker marker in Markers)
            {
                if (!marker.TextIsPayload) literals.Add(marker.Text);
                if (marker.Close is not null) literals.Add(marker.Close);
            }
            if (CloseMarker is not null) literals.Add(CloseMarker);
            literals.AddRange(ExtraLiterals);
            return literals;
        }
    }
}
