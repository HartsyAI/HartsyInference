namespace HartsyInference.Core.Backends;

/// <summary>Declarative multi-device placement for one engine. Device tokens are backend selectors
/// (<c>"cuda:0"</c>). Null/empty members mean "the engine's primary backend" — the all-defaults instance is
/// exactly today's single-device behavior, byte-for-byte (see <see cref="IsSingle"/> and <see cref="CacheKey"/>).
/// Config, not new model classes: pipelines and loaders consume this record; nothing else changes shape.</summary>
public sealed record PlacementConfig
{
    /// <summary>The single-device default; placement-disabled paths compare against this.</summary>
    public static readonly PlacementConfig Single = new();

    /// <summary>Ordered pipeline-stage devices for sharded execution (LLM layer split / DiT block split),
    /// e.g. <c>["cuda:0","cuda:1"]</c>. Empty = no sharding.</summary>
    public IReadOnlyList<string> ShardDevices { get; init; } = [];

    /// <summary>Explicit proportional layer split across <see cref="ShardDevices"/> (the llama.cpp
    /// <c>--tensor-split</c> shape); null = auto-plan by free VRAM. Length must match <see cref="ShardDevices"/>.</summary>
    public IReadOnlyList<float>? ShardRatios { get; init; }

    /// <summary>Device for diffusion text encoders (CLIP/T5/umT5/LLM-style); null = primary. Safe without any
    /// peer-copy support because the encoder→denoiser handoff host-materializes.</summary>
    public string? TextEncoderDevice { get; init; }

    /// <summary>Device for VAE encode/decode; null = primary.</summary>
    public string? VaeDevice { get; init; }

    /// <summary>Device that runs the negative/uncond CFG branch with replicated denoiser weights, concurrent with
    /// the positive branch on the primary; null = off. Requires the denoiser to fit on BOTH devices.</summary>
    public string? CfgParallelDevice { get; init; }

    /// <summary>Opt-in gate for experimental DiT block-sharding (<see cref="ShardDevices"/> then applies to the
    /// denoiser). Off = ShardDevices only affects LLMs.</summary>
    public bool EnableDitSharding { get; init; }

    /// <summary>Future tensor-parallel degree (NCCL milestone); 1 = off.</summary>
    public int TensorParallelDegree { get; init; } = 1;

    /// <summary>True when every member is at its default — the engine must then behave byte-identically to a
    /// build without placement support.</summary>
    public bool IsSingle =>
        ShardDevices.Count == 0
        && ShardRatios is null
        && TextEncoderDevice is null
        && VaeDevice is null
        && CfgParallelDevice is null
        && !EnableDitSharding
        && TensorParallelDegree == 1;

    /// <summary>Stable, order-sensitive identity for pipeline cache keys. Empty for <see cref="IsSingle"/> so
    /// existing cache keys stay byte-identical when placement is unused.</summary>
    public string CacheKey()
    {
        if (IsSingle)
        {
            return "";
        }
        string shard = ShardDevices.Count == 0
            ? ""
            : string.Join(",", ShardDevices) + (ShardRatios is null ? "" : "@" + string.Join(",", ShardRatios));
        return $"|placement:shard={shard};te={TextEncoderDevice};vae={VaeDevice};cfg={CfgParallelDevice};" +
            $"dit={(EnableDitSharding ? 1 : 0)};tp={TensorParallelDegree}";
    }
}
