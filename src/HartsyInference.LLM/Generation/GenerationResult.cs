namespace HartsyInference.LLM.Generation;

/// <summary>Result of a text-generation run.</summary>
public sealed record GenerationResult
{
    /// <summary>The generated token ids (not including the prompt).</summary>
    public required IReadOnlyList<int> TokenIds { get; init; }

    /// <summary>The decoded generated text.</summary>
    public required string Text { get; init; }

    /// <summary>The prompt length in tokens (for throughput accounting).</summary>
    public required int PromptTokens { get; init; }

    /// <summary>True when generation stopped on an end-of-turn/end-of-text/stop token rather than the <c>MaxTokens</c> limit.</summary>
    public required bool StoppedOnStopToken { get; init; }

    /// <summary>Prompt tokens served from a reused KV prefix instead of freshly prefilled (0 when no prefix-cache
    /// entry was supplied, or none of its tokens matched this call's prompt).</summary>
    public int ReusedPromptTokens { get; init; }

    /// <summary>Wall time from the first prefill to the sampled first token, in milliseconds (0 when the run was not timed).</summary>
    public double PrefillMilliseconds { get; init; }

    /// <summary>Wall time of the decode loop after the first token, in milliseconds (0 when the run was not timed).</summary>
    public double DecodeMilliseconds { get; init; }
}
