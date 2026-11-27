using System.Globalization;

namespace HartsyInference.LLM.Ernie5;

/// <summary>What a checkpoint's keys map to: model parameters, their FP8 scale companions, skipped MTP keys, and what is left over.</summary>
/// <param name="Mapped">Keys that feed a parameter of <see cref="Ernie5Model"/>.</param>
/// <param name="ScaleCompanions">FP8 block-scale keys whose weight is mapped.</param>
/// <param name="SkippedMtp">Keys of the multi-token-prediction layers, skipped because the model was loaded without
/// <see cref="Ernie5LoadOptions.Mtp"/>.</param>
/// <param name="Unmapped">Keys that feed nothing; any one fails the load.</param>
/// <param name="Missing">Parameters with no key; any one fails the load.</param>
public sealed record Ernie5WeightCoverage(int Mapped, int ScaleCompanions, int SkippedMtp, IReadOnlyList<string> Unmapped,
    IReadOnlyList<string> Missing)
{
    /// <summary>Every key the coverage saw.</summary>
    public int Total => Mapped + ScaleCompanions + SkippedMtp + Unmapped.Count;

    /// <summary>Fails when any key feeds nothing or any parameter has no key, naming the first few of each.</summary>
    /// <exception cref="InvalidDataException">The checkpoint and the model do not match key for key.</exception>
    public void RequireComplete()
    {
        if (Unmapped.Count == 0 && Missing.Count == 0) return;
        throw new InvalidDataException(string.Create(CultureInfo.InvariantCulture,
            $"The ERNIE-5.5 checkpoint does not match the model: {Unmapped.Count} unmapped key(s) [{string.Join(", ", Unmapped.Take(8))}], "
            + $"{Missing.Count} missing parameter(s) [{string.Join(", ", Missing.Take(8))}]."));
    }
}
