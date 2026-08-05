using HartsyInference.Core.Backends;

namespace HartsyInference.Engine.Audio;

/// <summary>Load-time context for music model descriptors: the primary backend plus, when the engine placement
/// pools VRAM across GPUs, the ordered shard devices and the resolved LM precision policy. Single-device with
/// <see cref="AudioLmQuant.Q4K"/> is byte-identical to the pre-placement behavior.</summary>
internal sealed record MusicLoadContext
{
    /// <summary>The engine's primary compute backend (never disposed by loaders).</summary>
    internal required IBackend Backend { get; init; }

    /// <summary>Ordered shard stages (selector + resolved backend) when the engine placement has ≥2
    /// <c>ShardDevices</c>; null = single-device. Backends belong to the engine pool — do not dispose.</summary>
    internal IReadOnlyList<(string Selector, IBackend Backend)>? ShardStages { get; init; }

    /// <summary>Explicit proportional split across <see cref="ShardStages"/> (null = auto by free VRAM).</summary>
    internal IReadOnlyList<float>? ShardRatios { get; init; }

    /// <summary>Resolved precision for the big codec-token LMs (YuE); other families ignore it.</summary>
    internal AudioLmQuant LmQuant { get; init; } = AudioLmQuant.Q4K;

    /// <summary>True when a layer-split placement is available to loaders.</summary>
    internal bool IsSharded => ShardStages is { Count: >= 2 };

    /// <summary>Runner-cache key suffix so a quant/placement change never reuses a stale runner. Empty for the
    /// single-device Q4_K default, keeping existing cache keys byte-identical.</summary>
    internal string CacheSuffix()
    {
        if (!IsSharded && LmQuant == AudioLmQuant.Q4K)
        {
            return "";
        }
        string shard = IsSharded ? string.Join("+", ShardStages!.Select(s => s.Selector)) : "";
        return $"|lmq={LmQuant}|shard={shard}";
    }
}
