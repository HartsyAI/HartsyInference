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

    /// <summary>Total pages in each loaded chat model's shared KV pool — the real cap on how many concurrent
    /// chat sequences (and how much total context across them) <see cref="LLM.Generation.DynamicBatchScheduler"/>
    /// can admit at once for that model. <c>KvPageSize * KvPoolMaxPages</c> tokens of total shared KV capacity.
    /// Size for your VRAM budget and expected concurrency; admission fails fast with
    /// <see cref="LLM.Transformer.KvPoolExhaustedException"/> (mapped to HTTP 429) once exhausted, it does
    /// not degrade silently.</summary>
    public int KvPoolMaxPages { get; set; } = 1024;
}

/// <summary>Selectable compute backends for the server.</summary>
public enum BackendKind
{
    /// <summary>CPU SIMD backend (always available).</summary>
    Cpu,

    /// <summary>NVIDIA CUDA backend (requires <see cref="HartsyInferenceServerOptions.PtxDirectory"/>).</summary>
    Cuda,
}
