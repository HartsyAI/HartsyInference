namespace HartsyInference.Core.Rope;

/// <summary>Cache-wide table settings. <see cref="RopeTableCache"/> copies them at construction.</summary>
public sealed record RopeTableSettings
{
    /// <summary>Operation order for position × frequency.</summary>
    public required RopeAngleOrder AngleOrder { get; init; }

    /// <summary>Row layout that the apply op reads.</summary>
    public required RopeTableLayout Layout { get; init; }

    /// <summary>Rotated width; 0 means the full head.</summary>
    public int RotaryDim { get; init; }

    /// <summary>Values for entries the layout does not read.</summary>
    public RopeTablePadding Padding { get; init; } = RopeTablePadding.Identity;

    /// <summary>Build device rows for each backend and preload them. Host tables are always built.</summary>
    public bool Preload { get; init; }

    /// <summary>Signatures kept per backend slot. The least recently used signature is retired when a new one exceeds this.</summary>
    public int RetainedSignatures { get; init; } = 2;
}
