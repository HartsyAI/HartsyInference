namespace HartsyInference.ThreeD.Pipelines.Requests;

/// <summary>Inputs for a single-image → 3D generation. The conditioning image is interleaved RGB24
/// (<c>Width*Height*3</c> bytes, row-major) — the same byte layout the diffusion image pipelines use.
/// Background removal (if desired) is the caller's responsibility for v1.</summary>
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

    /// <summary>Marching-cubes grid resolution (samples per axis) for the occupancy/SDF decode.
    /// 0 → the model's default (e.g. 256).</summary>
    public int GridResolution { get; init; }

    /// <summary>Iso level for surface extraction. Defaults to 0 (SDF zero-crossing); occupancy/logit
    /// models override.</summary>
    public float IsoLevel { get; init; }
}
