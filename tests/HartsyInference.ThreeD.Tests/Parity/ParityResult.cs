namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>One compared stage: the largest absolute and per-tensor relative difference, the count of elements outside tolerance,
/// and an optional note. A stage passes only when it has no failures.</summary>
internal readonly record struct ParityResult(string Name, int Count, double MaxAbs, double MaxRel, int Failures, string? Note)
{
    internal bool Passed => Failures == 0;

    internal static ParityResult Failed(string name, string note) => new(name, 0, double.NaN, double.NaN, 1, note);
}
