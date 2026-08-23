namespace HartsyInference.Engine.Requests;

/// <summary>Native chat/completion request. Carries the conversation, sampling knobs, tool definitions, and the decode hints the local backend honors. Per-request knobs (temperature/topP/seed/maxTokens) live here; nullable knobs fall back to the model/engine default when unset.</summary>
public sealed record TextRequest
{
    /// <summary>The conversation so far, oldest first.</summary>
    public required IReadOnlyList<TextMessage> Messages { get; init; }

    /// <summary>System prompt applied ahead of <see cref="Messages"/>; null/empty for none.</summary>
    public string? SystemPrompt { get; init; }

    /// <summary>Sampling temperature; &lt;= 0 selects greedy decoding.</summary>
    public double Temperature { get; init; } = 0.7;

    /// <summary>Nucleus (top-P) cutoff.</summary>
    public double TopP { get; init; } = 0.95;

    /// <summary>Top-K cutoff; null uses the model default.</summary>
    public int? TopK { get; init; }

    /// <summary>Min-P cutoff; null uses the model default.</summary>
    public double? MinP { get; init; }

    /// <summary>Repetition penalty; null uses the model default.</summary>
    public double? RepetitionPenalty { get; init; }

    /// <summary>Maximum tokens to generate.</summary>
    public int MaxTokens { get; init; } = 4096;

    /// <summary>Sampling seed; negative means a random seed per request.</summary>
    public long Seed { get; init; } = -1;

    /// <summary>Force greedy decoding regardless of temperature.</summary>
    public bool Greedy { get; init; }

    /// <summary>Sets the model's chat-template <c>enable_thinking</c> variable (Qwen3-family reasoning-block toggle); null leaves it undefined so the template falls back to its own default. Ignored by templates without a thinking slot.</summary>
    public bool? EnableThinking { get; init; }

    /// <summary>Target device key (e.g. "cpu", "cuda:0"); null uses the backend's primary device.</summary>
    public string? Device { get; init; }

    /// <summary>Tool definitions offered to the model; null/empty disables tool calling.</summary>
    public IReadOnlyList<ToolDefinition>? Tools { get; init; }

    /// <summary>Force the model to call this tool by name; null lets it choose.</summary>
    public string? ForceToolId { get; init; }

    /// <summary>Enable graph-mode decode when supported; null uses the engine default.</summary>
    public bool? GraphDecode { get; init; }

    /// <summary>Enable speculative decode when supported; null uses the engine default.</summary>
    public bool? SpeculativeDecode { get; init; }

    /// <summary>Low-VRAM on-the-fly quant to load weights at (e.g. "q8_0"); null loads at full precision.</summary>
    public string? LowVramQuant { get; init; }

    /// <summary>Free the model's device memory after this request completes; null uses the engine default.</summary>
    public bool? AlwaysFreeMemory { get; init; }
}
