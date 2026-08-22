namespace HartsyInference.ThreeD.Pipelines.Requests;

/// <summary>Inputs for a single-image → 3D generation; the conditioning image uses the same interleaved RGB24 row-major byte layout as the diffusion image pipelines.</summary>
public sealed record ImageTo3DRequest
{
    /// <summary>Interleaved RGB24 conditioning image, length <c>Width*Height*3</c>.</summary>
    public required byte[] ImageRgb { get; init; }

    /// <summary>Conditioning image width in pixels.</summary>
    public required int Width { get; init; }

    /// <summary>Conditioning image height in pixels.</summary>
    public required int Height { get; init; }

    /// <summary>Number of flow-match denoise steps. 0 → the model's default.</summary>
    public int Steps { get; init; }

    /// <summary>Classifier-free guidance scale. 0 → the model's default.</summary>
    public float CfgScale { get; init; }

    /// <summary>RNG seed for the initial latent noise. Null → random.</summary>
    public int? Seed { get; init; }

    /// <summary>Marching-cubes grid resolution (samples per axis) for the occupancy/SDF decode; 0 → the model's default.</summary>
    public int GridResolution { get; init; }

    /// <summary>Iso level for surface extraction; defaults to 0 (SDF zero-crossing), overridden by occupancy/logit models.</summary>
    public float IsoLevel { get; init; }
}
