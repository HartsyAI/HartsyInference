namespace HartsyInference.Engine.Requests;

/// <summary>Token accounting of a finished text request; the draft counts are zero when speculative decoding was off.</summary>
public readonly record struct TextUsage(
    int PromptTokens, int CompletionTokens, int CachedPromptTokens = 0, int DraftAccepted = 0, int DraftProposed = 0);
