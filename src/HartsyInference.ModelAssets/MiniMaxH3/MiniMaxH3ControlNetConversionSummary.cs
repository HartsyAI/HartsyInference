namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>Provenance and fit result written by a local H3 Fun ControlNet pruned-base conversion.</summary>
public sealed record MiniMaxH3ControlNetConversionSummary
{
    /// <summary>SHA-256 of the original full-width control branch.</summary>
    public required string ControlSha256 { get; init; }

    /// <summary>SHA-256 of the dense full base used to reconstruct timestep coordinates.</summary>
    public required string FullBaseSha256 { get; init; }

    /// <summary>SHA-256 of the target pruned FL2VA base.</summary>
    public required string TargetBaseSha256 { get; init; }

    /// <summary>Relative F64 residual of the dense-to-curve affine fit.</summary>
    public required double RelativeResidual { get; init; }

    /// <summary>Number of control AdaLN projections rebased.</summary>
    public required int RebasedBlocks { get; init; }
}
