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
}

/// <summary>Selectable compute backends for the server.</summary>
public enum BackendKind
{
    /// <summary>CPU SIMD backend (always available).</summary>
    Cpu,

    /// <summary>NVIDIA CUDA backend (requires <see cref="HartsyInferenceServerOptions.PtxDirectory"/>).</summary>
    Cuda,
}
