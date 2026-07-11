namespace HartsyInference.Server;

/// <summary>Configuration for the HartsyInference server. Bound from configuration in
/// <see cref="HartsyInferenceServiceExtensions.AddHartsyInference"/>.</summary>
public sealed class HartsyInferenceServerOptions
{
    /// <summary>Compute backend to run inference on.</summary>
    public BackendKind Backend { get; set; } = BackendKind.Cpu;

    /// <summary>Directory containing PTX kernels (required when <see cref="Backend"/> is CUDA).</summary>
    public string? PtxDirectory { get; set; }

    /// <summary>CUDA device ordinal.</summary>
    public int CudaDevice { get; set; }

    /// <summary>Optional API key. When set, requests must present it via <c>Authorization: Bearer</c> or <c>x-api-key</c>.</summary>
    public string? ApiKey { get; set; }

    /// <summary>Maximum concurrent inference requests.</summary>
    public int MaxConcurrency { get; set; } = 1;

    /// <summary>Maximum queued (waiting) requests before the server returns HTTP 429.</summary>
    public int MaxQueueDepth { get; set; } = 16;

    /// <summary>Model cache directory for HuggingFace downloads (null = default <c>~/.hartsyinference/models</c>).</summary>
    public string? ModelCacheDirectory { get; set; }

    /// <summary>Tokens per KV page for each loaded chat (dense/MoE transformer) model's <c>PagedKvPool</c>.</summary>
    public int KvPageSize { get; set; } = 16;

    /// <summary>VRAM budget (bytes) for each loaded chat model's KV pool — <see cref="ModelManager"/>
    /// converts this into a page COUNT sized to the specific model's KV dimensions
    /// (<c>numLayers × numKvHeads × headDim</c>, F32, K+V), not a raw page count, because a fixed page
    /// count is NOT safe across different model shapes: the same 1024 pages that comfortably fit a small
    /// model's narrow KV tensors can eagerly pre-allocate several GB for a model with wider heads/more
    /// layers (measured: a naive fixed default OOM'd loading a 1B model on a 12GB GPU with ~4GB already in
    /// use by unrelated processes — the pool alone wanted ~3.4GB). <c>PagedKvPool</c> allocates its full
    /// budget UP FRONT (not lazily), so this bound is exact, not a soft target. This is the real cap on how
    /// many concurrent chat sequences (and how much total context across them)
    /// <see cref="LLM.Generation.DynamicBatchScheduler"/> can admit at once for that model — admission fails
    /// fast with <see cref="LLM.Transformer.KvPoolExhaustedException"/> (mapped to HTTP 429) once exhausted,
    /// it does not degrade silently. Default (512 MB) is a conservative single-user-dev-box starting point;
    /// tune for your actual VRAM budget and expected concurrency.</summary>
    public long KvPoolBytesBudget { get; set; } = 512L * 1024 * 1024;
}

/// <summary>Selectable compute backends for the server.</summary>
public enum BackendKind
{
    /// <summary>CPU SIMD backend (always available).</summary>
    Cpu,

    /// <summary>NVIDIA CUDA backend (requires <see cref="HartsyInferenceServerOptions.PtxDirectory"/>).</summary>
    Cuda,
}
