using HartsyInference.Core.Memory.Vram;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Planning.Memory;

/// <summary>The device memory of a 3D request, from the peaks recorded for it. A model with no recorded peak has no estimate: the engine claims nothing it did not measure.</summary>
internal static class MeshMemoryProfile
{
    /// <summary>The matting checkpoint's weights (the F32 file the catalog pins). The model stays resident beside the shape model from the first matte until the 3D side is freed; its activations are not recorded. It is added for a request that mattes its image; a model an earlier request left resident is not counted (the estimate does not read engine state).</summary>
    public const long MatterWeightBytes = 884_878_856;

    /// <summary>The recorded whole-card peak of <paramref name="request"/> on <paramref name="id"/> as a one-stage estimate, plus the matting model when the request mattes its image, or null when no peak was recorded.</summary>
    public static MemoryEstimate? Estimate(string id, MeshRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        double? gib = MeshModelProfiles.Find(id)?.PeakGiB?.Invoke(request);
        if (gib is null)
        {
            return null;
        }
        long bytes = (long)(gib.Value * (1L << 30)) + (Mattes(id, request) ? MatterWeightBytes : 0);
        return new MemoryEstimate
        {
            FamilyId = id,
            Phases = [new MemoryPhase(MemoryComponent.Stage, bytes, 0, 0, false)],
            Accuracy = MemoryEstimateAccuracy.Measured,
        };
    }

    /// <summary>Whether the request makes the engine matte its image: TRELLIS.2 and Pixal3D always matte an image without a usable alpha plane, and the models whose input the engine prepares do under <see cref="MeshPreprocess.Auto"/>.</summary>
    internal static bool Mattes(string id, MeshRequest request) =>
        request.Image is { } image && !MeshPreparationService.HasUsableAlpha(image)
        && (id == MeshModelIds.Trellis2 || id == MeshModelIds.Pixal3D || (request.Preprocess == MeshPreprocess.Auto && MeshPreparationService.UsesPreparation(id)));

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
                ? $"The recorded peak, with the matting model when the engine mattes the image, is {ByteFormat.GbF1(peak)} of {ByteFormat.GbF1(totalBytes)} on the card; free memory is not checked, so other programs on the card can still make it fail."
                : $"The recorded peak, with the matting model when the engine mattes the image, is {ByteFormat.GbF1(peak)}; this card has {ByteFormat.GbF1(totalBytes)}.",
        };
    }
}
