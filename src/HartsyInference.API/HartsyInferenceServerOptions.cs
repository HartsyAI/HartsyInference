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

    /// <summary>Optional single shared API key. When set, requests must present it via <c>Authorization: Bearer</c>
    /// or <c>x-api-key</c>. Sugar for the common single-tenant case — folded into <see cref="ApiKeys"/> as an
    /// entry named <c>"default"</c> at startup (see <c>AddHartsyInference</c>). Prefer <see cref="ApiKeys"/>
    /// directly when multiple callers need distinct identities/rate limits.</summary>
    public string? ApiKey { get; set; }

    /// <summary>Named API keys, each resolving to its own caller identity for usage metering, tracing, and
    /// per-key rate limiting. Bindable from configuration as <c>HartsyInference:ApiKeys:0:Key</c> etc. If both
    /// this and <see cref="ApiKey"/> are set, both are honored (the legacy key is folded in as one more entry).
    /// If neither is set, auth is disabled entirely and every route (except <c>/admin/*</c>-equivalent-sensitive
    /// ones like <c>/metrics</c>) is open.</summary>
    public List<ApiKeyOptions> ApiKeys { get; set; } = [];

    /// <summary>Requests-per-minute ceiling applied to a resolved caller identity that doesn't set its own
    /// <see cref="ApiKeyOptions.RateLimitPerMinute"/>, and to unauthenticated callers (partitioned by remote IP)
    /// when auth is disabled entirely.</summary>
    public int DefaultRateLimitPerMinute { get; set; } = 300;

    /// <summary>Maximum concurrent inference requests for the "fast" modalities (image/text/speech/transcribe/
    /// voice-convert/fx/vision/mesh — everything that completes in seconds).</summary>
    public int MaxConcurrency { get; set; } = 1;

    /// <summary>Maximum queued (waiting) "fast" requests before the server returns HTTP 429.</summary>
    public int MaxQueueDepth { get; set; } = 16;

    /// <summary>Maximum concurrent long-running requests (video generation, opening a world session — both can
    /// run for minutes). Separate from <see cref="MaxConcurrency"/> so one slow video job can't starve every fast
    /// request behind it in the same queue.</summary>
    public int MaxLongRunningConcurrency { get; set; } = 1;

    /// <summary>Maximum queued (waiting) long-running requests before the server returns HTTP 429.</summary>
    public int MaxLongRunningQueueDepth { get; set; } = 4;

    /// <summary>How long an interactive world session may sit with no action/stream activity before the server
    /// evicts it and releases its backend resources.</summary>
    public int WorldSessionIdleTimeoutMinutes { get; set; } = 10;

    /// <summary>Model cache directory for HuggingFace downloads (null = default <c>~/.hartsyinference/models</c>).</summary>
    public string? ModelCacheDirectory { get; set; }

    /// <summary>Opens the always-on wake-word listener that voice satellites stream into. Off by default: it
    /// binds a second port and holds a detection thread, which a server used only for HTTP inference should
    /// not pay for.</summary>
    public bool WakeEnabled { get; set; }

    /// <summary>TCP port satellites connect to when <see cref="WakeEnabled"/>.</summary>
    public int WakePort { get; set; } = 10_800;

    /// <summary>Wake assets directory; defaults to <c>{models}/audio/wake</c>.</summary>
    public string? WakeModelRoot { get; set; }

    /// <summary>Whether a detection also transcribes the command that follows it.</summary>
    public bool WakeTranscribeOnDetection { get; set; } = true;
}

/// <summary>One entry in <see cref="HartsyInferenceServerOptions.ApiKeys"/> — a secret plus the identity/limits
/// it resolves to.</summary>
public sealed class ApiKeyOptions
{
    /// <summary>The secret value a caller presents via <c>Authorization: Bearer</c> or <c>x-api-key</c>.</summary>
    public string? Key { get; set; }

    /// <summary>Identity label for this key — used as the usage-metering key, the rate-limiter partition key, and
    /// the tracing tag on every request this key authenticates. Defaults to <c>"default"</c> when unset, matching
    /// the legacy single-<see cref="HartsyInferenceServerOptions.ApiKey"/> fold-in.</summary>
    public string Name { get; set; } = "default";

    /// <summary>Per-key requests-per-minute override. Null falls back to
    /// <see cref="HartsyInferenceServerOptions.DefaultRateLimitPerMinute"/>.</summary>
    public int? RateLimitPerMinute { get; set; }
}
