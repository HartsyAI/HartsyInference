namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Token ids of a rendered conversation plus the per-position image metadata the model needs. Masks are positional with <paramref name="Ids"/>; both are true exactly inside image spans (Engram n-grams skip them, the vision router bias applies to them).</summary>
public sealed record EncodedConversation(
    int[] Ids,
    IReadOnlyList<ImageSpan> ImageSpans,
    bool[] DeadMask,
    bool[] VisionRouteMask,
    int PromptLength,
    ParserInitialState InitialParserState);
