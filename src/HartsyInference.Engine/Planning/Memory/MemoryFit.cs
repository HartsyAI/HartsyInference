using HartsyInference.Core.MemoryManagement;

namespace HartsyInference.Engine.Planning.Memory;

/// <summary>One engine's answer to "can this generation run on my device, and how".</summary>
/// <remarks>Facts only. Which verdict a host should prefer — wait for a card that holds the model resident, or accept
/// streaming now — is a routing decision that belongs to the host; <see cref="EffectiveTier"/> is here so it can make
/// that decision under the tier the generation will actually run with.</remarks>
public sealed record MemoryFit
{
    /// <summary>How the generation fits this device.</summary>
    public required MemoryFitVerdict Verdict { get; init; }

    /// <summary>The estimate the verdict was judged from; null when the device could not be assessed.</summary>
    public MemoryEstimate? Estimate { get; init; }

    /// <summary>Usable bytes on this device (total VRAM less the engine's fixed reserve, plus pooled shard devices).
    /// Zero when unknown.</summary>
    public long CapacityBytes { get; init; }

    /// <summary>The tier after the request's overrides were applied to the backend's policy.</summary>
    public required VramTier EffectiveTier { get; init; }

    /// <summary>A one-line, user-facing explanation of the verdict.</summary>
    public required string Reason { get; init; }
}
