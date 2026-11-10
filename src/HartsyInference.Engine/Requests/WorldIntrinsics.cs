namespace HartsyInference.Engine.Requests;

/// <summary>Pinhole intrinsics in pixels of the original seed image (frame 0 as supplied, before any resize).</summary>
public sealed record WorldIntrinsics
{
    /// <summary>Focal length along x, in pixels.</summary>
    public required double Fx { get; init; }

    /// <summary>Focal length along y, in pixels.</summary>
    public required double Fy { get; init; }

    /// <summary>Principal point x, in pixels.</summary>
    public required double Cx { get; init; }

    /// <summary>Principal point y, in pixels.</summary>
    public required double Cy { get; init; }
}
