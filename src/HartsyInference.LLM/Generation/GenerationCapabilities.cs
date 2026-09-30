namespace HartsyInference.LLM.Generation;

/// <summary>What a drive loop may rely on from an <see cref="IGenerationModel"/>; the defaults describe a plain single-shot decoder.</summary>
public sealed record GenerationCapabilities
{
    /// <summary>Prefill with an embeds override must still receive the token ids (per-layer embedding inputs, Engram).</summary>
    public bool NeedsTokenIdsWithEmbeds { get; init; }

    /// <summary>Verify-forward plus <see cref="ISequenceState.Truncate"/> rollback is supported, so prompt-lookup speculation may run.</summary>
    public bool SupportsSpeculation { get; init; }

    /// <summary>Most draft tokens one verify pass may carry.</summary>
    public int MaxSpeculativeDepth { get; init; }

    /// <summary>Longest token span one <see cref="IGenerationModel.Prefill"/> call accepts; longer prompts must be chunked.</summary>
    public int MaxPrefillChunk { get; init; } = int.MaxValue;

    /// <summary>Every chunk except the last must be a multiple of this many tokens.</summary>
    public int ChunkAlignment { get; init; } = 1;

    /// <summary>The chunk planner must move boundaries so no image span is split across two chunks.</summary>
    public bool NoImageSpanStraddlesChunk { get; init; }

    /// <summary><see cref="IGenerationModel.DecodeBatch"/> is available for ragged multi-sequence rounds.</summary>
    public bool SupportsBatchDecode { get; init; }
}
