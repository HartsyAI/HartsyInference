namespace HartsyInference.Engine.Requests;

/// <summary>The result of an embedding request: one L2-normalized dense vector per input string, in the same order.</summary>
public sealed record EmbeddingResult
{
    /// <summary>One vector per <see cref="EmbeddingRequest.Input"/> entry, same order.</summary>
    public required IReadOnlyList<float[]> Vectors { get; init; }

    /// <summary>Vector width — every entry in <see cref="Vectors"/> has exactly this length.</summary>
    public required int Dimensions { get; init; }

    /// <summary>Total tokens actually fed through the model across every input (including the appended EOS per
    /// input) — a real count from the same tokenize pass that produced <see cref="Vectors"/>, not an estimate.</summary>
    public required int TotalTokens { get; init; }
}
