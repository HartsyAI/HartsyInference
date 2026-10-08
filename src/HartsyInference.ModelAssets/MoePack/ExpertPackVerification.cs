using HartsyInference.Core.Backends;

namespace HartsyInference.ModelAssets.MoePack;

/// <summary>What a verification pass measured against the source weights.</summary>
/// <param name="Checked">Experts whose bytes were read and compared.</param>
/// <param name="MaxAbsError">Largest absolute difference over every value checked.</param>
/// <param name="RelativeRmse">Root-mean-square error divided by the root-mean-square of the source values.</param>
/// <param name="Failures">Experts whose checksum did not match; they are not compared.</param>
public sealed record ExpertPackVerification(int Checked, double MaxAbsError, double RelativeRmse, IReadOnlyList<ExpertKey> Failures);
