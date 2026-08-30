namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>Provenance and fit result emitted by a local pruned-PDD conversion.</summary>
public sealed record MiniMaxH3PddConversionSummary
{
    /// <summary>SHA-256 of the official PDD adapter.</summary>
    public required string AdapterSha256 { get; init; }

    /// <summary>SHA-256 of the matching full H3 base used to reconstruct dense time features.</summary>
    public required string FullBaseSha256 { get; init; }

    /// <summary>SHA-256 of the target pruned H3 base whose curve table defines the output coordinates.</summary>
    public required string TargetBaseSha256 { get; init; }

    /// <summary>Relative F64 affine-fit residual.</summary>
    public required double RelativeResidual { get; init; }

    /// <summary>Number of AdaLN modules rebased with paired weight/DC-bias diffs.</summary>
    public required int RebasedModules { get; init; }
}
