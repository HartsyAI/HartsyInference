using HartsyInference.Diffusion.Prompting;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>One FLUX.3 prompt as the text encoder takes it: the chat-templated ids right-padded to a multiple of 80
/// with their emphasis weights, and how many of them are real (the rest are masked padding at RoPE position 1).</summary>
public sealed record Flux3PromptTokens(WeightedTokenSequence Tokens, int ValidLength)
{
    /// <summary>Whether the cond rows need per-token scaling after encoding.</summary>
    public bool IsWeighted => !Tokens.IsUniformlyUnweighted;
}
