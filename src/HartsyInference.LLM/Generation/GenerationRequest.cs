using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Sampling;

namespace HartsyInference.LLM.Generation;

/// <summary>A text-generation request. Provide either <see cref="Messages"/> (multi-turn chat) or
/// <see cref="Prompt"/> (single user turn, wrapped with the chat template + <see cref="SystemPrompt"/>);
/// <see cref="Messages"/> wins when both are set. <see cref="RawTokenIds"/> bypasses templating entirely.</summary>
public sealed record GenerationRequest
{
    /// <summary>Single user prompt (templated as one user turn). Ignored when <see cref="Messages"/> is set.</summary>
    public string? Prompt { get; init; }

    /// <summary>System prompt for the single-prompt path; null uses the template default, empty omits it.</summary>
    public string? SystemPrompt { get; init; }

    /// <summary>Multi-turn chat messages (templated). Takes precedence over <see cref="Prompt"/>.</summary>
    public IReadOnlyList<ChatMessage>? Messages { get; init; }

    /// <summary>Sets the chat template's <c>enable_thinking</c> variable (Qwen3-family reasoning toggle); null
    /// leaves it undefined so the template falls back to its own default. Ignored by templates without a
    /// thinking slot (e.g. ChatML).</summary>
    public bool? EnableThinking { get; init; }

    /// <summary>Pre-tokenized prompt ids; when set, templating and tokenization are skipped entirely.</summary>
    public IReadOnlyList<int>? RawTokenIds { get; init; }

    /// <summary>Maximum number of new tokens to generate.</summary>
    public int MaxTokens { get; init; } = 256;

    /// <summary>Sampling configuration (defaults to greedy-equivalent: temp 1, no top-k/p).</summary>
    public SamplingOptions Sampling { get; init; } = SamplingOptions.Default;

    /// <summary>Extra stop token ids beyond the model's end-of-turn / end-of-text tokens.</summary>
    public IReadOnlyList<int>? StopTokenIds { get; init; }

    /// <summary>Overrides whether CUDA-graph decode is attempted. Null defers to the <c>HARTSY_GRAPH_DECODE</c>
    /// environment variable. Graph decode still requires <see cref="Sampling"/> to be greedy and the model/backend
    /// to report eligibility via <c>SupportsGraphDecode</c> — this only controls the opt-in gate itself.</summary>
    public bool? GraphDecode { get; init; }

    /// <summary>Overrides whether prompt-lookup speculative decoding is attempted. Null defers to the
    /// <c>HARTSY_SPEC_DECODE</c> environment variable. Requires <see cref="Sampling"/> to be greedy and not
    /// JSON-mode, and is skipped whenever <see cref="GraphDecode"/> is actually eligible (graph decode wins).
    /// No draft model: drafts come from n-gram matches against the prompt/generated-so-far, so this only
    /// speeds up repetitive content (e.g. regenerating similar JSON, quoting earlier text) — on prose it costs
    /// nothing extra, since an unmatched draft degenerates to one plain decode step.</summary>
    public bool? SpeculativeDecode { get; init; }
}
