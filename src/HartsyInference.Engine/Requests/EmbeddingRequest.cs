namespace HartsyInference.Engine.Requests;

/// <summary>Native text-embedding request: one or more input strings to encode into dense vectors. Always a list internally, even for a single input — callers that accept OpenAI's string-or-array polymorphism (the compat route) normalize to a one-element list before reaching this type.</summary>
public sealed record EmbeddingRequest
{
    /// <summary>The strings to embed, in order — the result's vectors are returned in the same order.</summary>
    public required IReadOnlyList<string> Input { get; init; }
}
