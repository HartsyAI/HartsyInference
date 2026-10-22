namespace HartsyInference.Engine.Planning;

/// <summary>One resolved parameter: the value that runs, rendered invariantly, and where it came from.</summary>
/// <param name="Value">The value, or null when the family pipeline chooses it internally.</param>
/// <param name="Source">Where <paramref name="Value"/> came from.</param>
/// <param name="Note">Why it differs from what was requested, when it does.</param>
public readonly record struct ResolvedParameter(string? Value, ParameterSource Source, string? Note = null)
{
    /// <inheritdoc/>
    public override string ToString() =>
        $"{Value ?? "family"} ({Source}{(Note is null ? "" : $": {Note}")})";
}
