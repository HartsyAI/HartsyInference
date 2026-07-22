namespace HartsyInference.API;

/// <summary>Configuration for the HartsyInference server. Bound from configuration in
/// <see cref="HartsyInferenceServiceExtensions.AddHartsyInference"/>.</summary>
public sealed class HartsyInferenceServerOptions
{
    /// <summary>Compute backend selector passed straight through to <c>InferenceEngine</c>/<c>BackendFactory</c> —
    /// same tokens as the CLI's <c>--backend</c> (<c>auto</c>/<c>cpu</c>/<c>cuda</c>/<c>vulkan</c>). <c>auto</c>
    /// picks CUDA when a device is present, else CPU.</summary>
    public string Backend { get; set; } = "auto";

    /// <summary>Explicit PTX/SPIR-V kernel directory override, for hosts that deploy compiled kernels somewhere
    /// other than beside the engine assemblies. Applied to <c>BackendFactory.KernelDirOverride</c> at startup;
    /// leave null to use the default assembly-relative resolution.</summary>
    public string? KernelDirectory { get; set; }

    /// <summary>Optional API key. When set, requests must present it via <c>Authorization: Bearer</c> or <c>x-api-key</c>.</summary>
    public string? ApiKey { get; set; }

    /// <summary>Maximum concurrent inference requests.</summary>
    public int MaxConcurrency { get; set; } = 1;

    /// <summary>Maximum queued (waiting) requests before the server returns HTTP 429.</summary>
    public int MaxQueueDepth { get; set; } = 16;

    /// <summary>Model cache directory for HuggingFace downloads (null = default <c>~/.hartsyinference/models</c>).</summary>
    public string? ModelCacheDirectory { get; set; }
}
