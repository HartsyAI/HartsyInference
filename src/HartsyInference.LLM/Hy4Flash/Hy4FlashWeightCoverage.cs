using System.Globalization;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>What a checkpoint's tensors map to: model parameters, their FP8 scale companions, skipped drafter tensors, and what disagrees.</summary>
/// <param name="Mapped">Keys that feed a parameter.</param>
/// <param name="ScaleCompanions">FP8 block-scale keys whose weight is mapped.</param>
/// <param name="SkippedDrafter">Speculative-drafter tensors, skipped because the first series does not run the drafter.</param>
/// <param name="Unmapped">Keys that feed nothing; any one fails the load.</param>
/// <param name="Missing">Parameters with no key; any one fails the load.</param>
/// <param name="ShapeMismatches">Tensors whose shape differs from the config's.</param>
/// <param name="ScaleMismatches">FP8 weights without a correctly shaped scale grid, or a grid beside a weight that is not FP8.</param>
public sealed record Hy4FlashWeightCoverage(int Mapped, int ScaleCompanions, int SkippedDrafter, IReadOnlyList<string> Unmapped,
    IReadOnlyList<string> Missing, IReadOnlyList<string> ShapeMismatches, IReadOnlyList<string> ScaleMismatches)
{
    /// <summary>Every key the coverage saw.</summary>
    public int Total => Mapped + ScaleCompanions + SkippedDrafter + Unmapped.Count;

    /// <summary>Fails when any key feeds nothing, any parameter has no key, or any shape or scale grid disagrees, naming the first few of each.</summary>
    /// <exception cref="InvalidDataException">The checkpoint and the config do not match.</exception>
    public void RequireComplete()
    {
        if (Unmapped.Count == 0 && Missing.Count == 0 && ShapeMismatches.Count == 0 && ScaleMismatches.Count == 0) return;
        throw new InvalidDataException(string.Create(CultureInfo.InvariantCulture,
            $"The Hy4 Flash checkpoint does not match the config: {Unmapped.Count} unmapped key(s) [{string.Join(", ", Unmapped.Take(6))}], "
            + $"{Missing.Count} missing [{string.Join(", ", Missing.Take(6))}], {ShapeMismatches.Count} shape mismatch(es) "
            + $"[{string.Join("; ", ShapeMismatches.Take(4))}], {ScaleMismatches.Count} scale mismatch(es) [{string.Join("; ", ScaleMismatches.Take(4))}]."));
    }
}
