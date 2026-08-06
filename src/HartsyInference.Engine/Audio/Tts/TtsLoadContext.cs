using HartsyInference.Core.Backends;

namespace HartsyInference.Engine.Audio;

/// <summary>Load-time context for TTS model descriptors: the primary backend plus, when the engine placement pools
/// VRAM across GPUs, the ordered shard devices. Mirrors <see cref="MusicLoadContext"/> minus the audio-LM quant
/// policy field — no TTS codec-LM in this engine has an equivalent precision knob (yet).</summary>
internal sealed record TtsLoadContext
{
    /// <summary>The engine's primary compute backend (never disposed by loaders).</summary>
    internal required IBackend Backend { get; init; }

    /// <summary>Ordered shard stages (selector + resolved backend) when the engine placement has ≥2
    /// <c>ShardDevices</c>; null = single-device. Backends belong to the engine pool — do not dispose.</summary>
    internal IReadOnlyList<(string Selector, IBackend Backend)>? ShardStages { get; init; }

    /// <summary>Explicit proportional split across <see cref="ShardStages"/> (null = auto by free VRAM).</summary>
    internal IReadOnlyList<float>? ShardRatios { get; init; }

    /// <summary>True when a layer-split placement is available to loaders.</summary>
    internal bool IsSharded => ShardStages is { Count: >= 2 };

    /// <summary>Runner-cache key suffix so a sharded load never reuses a cached unsharded runner. Empty when
    /// unsharded, keeping existing cache keys byte-identical.</summary>
    internal string CacheSuffix()
    {
        if (!IsSharded)
        {
            return "";
        }
        return $"|shard={string.Join("+", ShardStages!.Select(s => s.Selector))}";
    }
}
