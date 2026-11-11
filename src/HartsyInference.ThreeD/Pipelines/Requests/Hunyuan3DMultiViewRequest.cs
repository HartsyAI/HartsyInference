namespace HartsyInference.ThreeD.Pipelines.Requests;

/// <summary>Inputs for Hunyuan3D-2mv shape generation from one to four views of the same object.</summary>
public sealed record Hunyuan3DMultiViewRequest
{
    /// <summary>The views, each at most once; any subset of front, left, back and right.</summary>
    public required IReadOnlyList<Hunyuan3DViewImage> Views { get; init; }

    /// <summary>Number of denoise steps. 0 uses the checkpoint's default.</summary>
    public int Steps { get; init; }

    /// <summary>Classifier-free guidance scale (or the embedded guidance for distilled builds). 0 uses the default.</summary>
    public float CfgScale { get; init; }

    /// <summary>RNG seed for the initial latent noise. Null is random.</summary>
    public int? Seed { get; init; }

    /// <summary>Decode grid resolution; 0 uses the default.</summary>
    public int GridResolution { get; init; }

    /// <summary>Iso level of the extracted surface on the decoder's logits; 0 uses the default.</summary>
    public float IsoLevel { get; init; }
}
