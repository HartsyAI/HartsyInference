namespace HartsyInference.Engine.Planning.Memory;

/// <summary>The VRAM a generation needs, per phase, before any weight is loaded.</summary>
/// <remarks>Kept per phase rather than as one total because the answer depends on the device it is asked for: a
/// policy that unloads between phases only ever holds the largest phase, one that keeps everything resident holds all
/// of them, and a component placed on another GPU does not count on this one at all. <see cref="PeakBytes"/> and
/// <see cref="FloorBytes"/> fold the phases for a given device.</remarks>
public sealed record MemoryEstimate
{
    /// <summary>The family id the estimate was resolved through.</summary>
    public required string FamilyId { get; init; }

    /// <summary>The phases the generation runs, each with its own weights and working memory.</summary>
    public required IReadOnlyList<MemoryPhase> Phases { get; init; }

    /// <summary>Where the activation numbers came from.</summary>
    public required MemoryEstimateAccuracy Accuracy { get; init; }

    /// <summary>The peak with every phase resident and phases unloading between each other: the single number to quote
    /// when no device is in view.</summary>
    public long PeakResidentBytes => PeakBytes(unloadBetweenPhases: true, _ => true);

    /// <summary>Peak bytes on one device with every counted phase fully resident.</summary>
    /// <param name="unloadBetweenPhases">True when a phase's weights are released before the next phase loads, so only
    /// the largest phase is ever held; false when every phase's weights stay resident together.</param>
    /// <param name="onDevice">Which components run on the device being asked about.</param>
    public long PeakBytes(bool unloadBetweenPhases, Func<MemoryComponent, bool> onDevice) =>
        Fold(unloadBetweenPhases, onDevice, streamed: false);

    /// <summary>The least the generation can run in on one device, streaming every phase that can stream.</summary>
    public long FloorBytes(bool unloadBetweenPhases, Func<MemoryComponent, bool> onDevice) =>
        Fold(unloadBetweenPhases, onDevice, streamed: true);

    private long Fold(bool unloadBetweenPhases, Func<MemoryComponent, bool> onDevice, bool streamed)
    {
        ArgumentNullException.ThrowIfNull(onDevice);
        long peak = 0;
        long weights = 0;
        long activations = 0;
        foreach (MemoryPhase phase in Phases)
        {
            if (!onDevice(phase.Component))
            {
                continue;
            }
            long phaseBytes = streamed ? phase.FloorBytes : phase.ResidentBytes;
            peak = Math.Max(peak, phaseBytes);
            weights += phaseBytes - phase.ActivationBytes;
            activations = Math.Max(activations, phase.ActivationBytes);
        }
        return unloadBetweenPhases ? peak : weights + activations;
    }
}
