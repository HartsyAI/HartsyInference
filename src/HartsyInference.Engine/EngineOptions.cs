namespace HartsyInference.Engine;

/// <summary>Compute/model-lifecycle configuration for the inference engine, independent of any transport. The HTTP
/// server maps its own options onto this; the CLI constructs it directly.</summary>
public sealed class EngineOptions
{
    /// <summary>Model cache directory for HuggingFace downloads (null = default <c>~/.hartsyinference/models</c>).</summary>
    public string? ModelCacheDirectory { get; set; }

    /// <summary>Tokens per KV page for each loaded chat (dense/MoE transformer) model's <c>PagedKvPool</c>.</summary>
    public int KvPageSize { get; set; } = 16;

    /// <summary>VRAM budget (bytes) for each loaded chat model's KV pool. <see cref="ModelManager"/> converts this
    /// into a page COUNT sized to the model's actual KV dimensions (numLayers × numKvHeads × headDim, F32, K+V),
    /// allocated up front — a fixed page count is unsafe across model shapes. Admission fails fast with a 429-mapped
    /// exhaustion once the budget is spent. Default 512 MB is a conservative single-GPU-dev starting point.</summary>
    public long KvPoolBytesBudget { get; set; } = 512L * 1024 * 1024;
}
