using HartsyInference.Core.Memory.Vram;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Planning.Memory;

/// <summary>The device memory of a 3D request, from the peaks recorded for it. A model with no recorded peak has no estimate: the engine claims nothing it did not measure.</summary>
internal static class MeshMemoryProfile
{
    /// <summary>The recorded whole-card peak of <paramref name="request"/> on <paramref name="id"/> as a one-stage estimate, or null when none was recorded.</summary>
    public static MemoryEstimate? Estimate(string id, MeshRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        double? gib = MeshModelProfiles.Find(id)?.PeakGiB?.Invoke(request);
        if (gib is null)
        {
            return null;
        }
        long bytes = (long)(gib.Value * (1L << 30));
        return new MemoryEstimate
        {
            FamilyId = id,
            Phases = [new MemoryPhase(MemoryComponent.Stage, bytes, 0, 0, false)],
            Accuracy = MemoryEstimateAccuracy.Measured,
        };
    }

    /// <summary>Whether <paramref name="estimate"/> fits a card of <paramref name="totalBytes"/>. The recorded peak is whole-card, so it is held against the whole card, not against the reserve the image path keeps.</summary>
    public static MemoryFit Judge(MemoryEstimate estimate, long totalBytes, VramTier tier)
    {
        long peak = estimate.PeakResidentBytes;
        bool fits = peak <= totalBytes;
        return new MemoryFit
        {
            Verdict = fits ? MemoryFitVerdict.Resident : MemoryFitVerdict.Infeasible,
            Estimate = estimate,
            CapacityBytes = totalBytes,
            EffectiveTier = tier,
            Reason = fits
                ? $"A recorded run peaked at {ByteFormat.GbF1(peak)} of {ByteFormat.GbF1(totalBytes)} on the card."
                : $"A recorded run peaked at {ByteFormat.GbF1(peak)}; this card has {ByteFormat.GbF1(totalBytes)}.",
        };
    }
}
