namespace HartsyInference.LLM.Kimi;

/// <summary>One sampled reply from <see cref="KimiK31VisionGenerator"/>: its text, the prompt's length after the image splice, its token count, and
/// whether it ran out of its token budget rather than sampling a stop id.</summary>
public readonly record struct KimiK31VisionReply(string Text, int PromptTokens, int Tokens, bool HitTokenLimit);
