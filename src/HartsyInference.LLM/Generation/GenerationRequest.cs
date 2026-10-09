using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Sampling;

namespace HartsyInference.LLM.Generation;

/// <summary>Provide either <see cref="Messages"/> (multi-turn chat, wins if both are set) or <see cref="Prompt"/> (single user turn wrapped with the chat template + <see cref="SystemPrompt"/>); <see cref="RawTokenIds"/> bypasses templating entirely.</summary>
public sealed record GenerationRequest
{
    /// <summary>Optional observer after prompt evaluation and first sampling, before decode; count is actual prompt tokens.</summary>
    public Action<int>? OnPrefillCompleted { get; init; }

    /// <summary>Single user prompt (templated as one user turn). Ignored when <see cref="Messages"/> is set.</summary>
    public string? Prompt { get; init; }

    /// <summary>System prompt: prepended as a system turn to <see cref="Prompt"/>, and to <see cref="Messages"/> when they do not already start with one. Null or empty adds nothing.</summary>
    public string? SystemPrompt { get; init; }

    /// <summary>Multi-turn chat messages (templated). Takes precedence over <see cref="Prompt"/>.</summary>
    public IReadOnlyList<ChatMessage>? Messages { get; init; }

    /// <summary>Tool schemas offered to the model through the chat template; null or empty renders the prompt without a tool block.</summary>
    public IReadOnlyList<ToolSpec>? Tools { get; init; }

    /// <summary>Sets the chat template's <c>enable_thinking</c> variable (Qwen3-family reasoning toggle); null falls back to the template's own default, and templates without a thinking slot (e.g. ChatML) ignore it.</summary>
    public bool? EnableThinking { get; init; }

    /// <summary>Called once with this request's place in the scheduler's waiting queue (1 is next) when an admission pass leaves it waiting behind other requests; a request
    /// admitted at once is never told. Called on the scheduler's loop, after it releases its queue lock. Not called on the pipeline.</summary>
    public Action<int>? OnQueued { get; init; }

    /// <summary>Reasoning effort in [1, 100] for a template that takes one (DeepSeek-V4.1 thinking mode); null uses the template's default. Templates without one ignore it.</summary>
    public int? ReasoningEffort { get; init; }


    /// <summary>Pre-tokenized prompt ids; when set, templating and tokenization are skipped entirely.</summary>
    public IReadOnlyList<int>? RawTokenIds { get; init; }

    /// <summary>Maximum number of new tokens to generate.</summary>
    public int MaxTokens { get; init; } = 256;

    /// <summary>Sampling configuration (defaults to greedy-equivalent: temp 1, no top-k/p).</summary>
    public SamplingOptions Sampling { get; init; } = SamplingOptions.Default;

    /// <summary>Extra stop token ids beyond the model's end-of-turn / end-of-text tokens.</summary>
    public IReadOnlyList<int>? StopTokenIds { get; init; }

    /// <summary>Overrides whether CUDA-graph decode is attempted (null defers to <c>numerics.graphDecode</c>); still requires <see cref="Sampling"/> to be greedy and the model/backend to report eligibility via <c>SupportsGraphDecode</c> — this only controls the opt-in gate itself.</summary>
    public bool? GraphDecode { get; init; }

    /// <summary>The messages the template actually renders: <see cref="Messages"/> with <see cref="SystemPrompt"/> prepended as a system turn unless they already open with one; null when <see cref="Messages"/> is null. Parsers must resolve their initial state from this view, not from <see cref="Messages"/>.</summary>
    public IReadOnlyList<ChatMessage>? EffectiveMessages() => Messages is null ? null : PromptBuilder.WithSystemPrompt(Messages, SystemPrompt);

    /// <summary>Overrides whether prompt-lookup speculative decoding is attempted (null defers to <c>numerics.specDecode</c>); requires greedy non-JSON <see cref="Sampling"/>, is skipped when <see cref="GraphDecode"/> is eligible, and uses no draft model — drafts come from n-gram matches against the prompt/generated-so-far, so it speeds up repetitive content but costs nothing extra on prose (an unmatched draft degenerates to one plain decode step).</summary>
    public bool? SpeculativeDecode { get; init; }

    /// <summary>Sizes a FRESH retained sequence's KV capacity (tokens) when
    /// <see cref="TextGenerationPipeline.Generate(GenerationRequest,RetainedSequence,Action{int},CancellationToken)"/> is
    /// given a prefix-cache entry with nothing cached yet; null sizes it to just this call's own prompt +
    /// <see cref="MaxTokens"/>. Only that first allocation: a retained sequence too small for a later call grows by
    /// copying its reusable prefix, and is shrunk back toward its length when each call ends
    /// (<see cref="PrefixCacheHeadroomTokens"/>).</summary>
    public int? PrefixCacheCapacityHint { get; init; }

    /// <summary>Spare KV capacity (tokens) a retained sequence keeps past its length when the call ends; a larger
    /// allocation is copied down to its length plus this. Null defers to <c>vram.prefixCacheHeadroomTokens</c>.</summary>
    public int? PrefixCacheHeadroomTokens { get; init; }

    /// <summary>Most KV bytes a retained sequence may keep once the call ends; one that would need more is freed
    /// instead of retained. Null defers to <c>vram.prefixCacheMaxBytes</c>.</summary>
    public long? PrefixCacheMaxBytes { get; init; }
}
