namespace HartsyInference.Core.MemoryManagement;

/// <summary>Maps a device's TOTAL VRAM to the tier <see cref="VramTier.Auto"/> starts from.</summary>
/// <remarks>Total, not free: this picks a posture for the card, and a transient neighbour process must not make an
/// otherwise-roomy card behave like a small one for the rest of the session. Free VRAM still decides each individual
/// phase — <see cref="VramPlanner"/> measures it per phase — so the card's identity chooses the posture and the
/// measurement makes the call.</remarks>
public static class GpuVramClass
{
    /// <summary>At or below this, streaming is the normal case rather than the exception.</summary>
    public const long AggressiveCeilingBytes = 8L << 30;

    /// <summary>At or below this, phase-unloading pays for itself on most current checkpoints.</summary>
    public const long BalancedCeilingBytes = 16L << 30;

    /// <summary>Above this, a full-size checkpoint plus its activations normally fits with room to spare.</summary>
    public const long PerformanceFloorBytes = 24L << 30;

    /// <summary>The starting tier for a device with <paramref name="totalVramBytes"/>; a non-positive reading (CPU, or a backend that does not report) yields <see cref="VramTier.Balanced"/> as the safe middle.</summary>
    public static VramTier Seed(long totalVramBytes)
    {
        if (totalVramBytes <= 0)
        {
            return VramTier.Balanced;
        }
        if (totalVramBytes <= AggressiveCeilingBytes)
        {
            return VramTier.Aggressive;
        }
        if (totalVramBytes <= BalancedCeilingBytes)
        {
            return VramTier.Balanced;
        }
        if (totalVramBytes < PerformanceFloorBytes)
        {
            return VramTier.Balanced;
        }
        return VramTier.Performance;
    }
}
