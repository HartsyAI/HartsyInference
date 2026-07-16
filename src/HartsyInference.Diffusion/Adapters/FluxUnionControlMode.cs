namespace HartsyInference.Diffusion.Adapters;

/// <summary>Control-mode ids for union Flux ControlNets (InstantX FLUX.1-dev-Controlnet-Union / Shakker
/// Union-Pro). The integer values are the checkpoint's <c>controlnet_mode_embedder</c> row indices — they must
/// match the training mapping exactly (diffusers <c>FluxControlNetPipeline</c> <c>control_mode</c>). Union-Pro-2.0
/// removed the mode embedder entirely; no mode id is passed for it.</summary>
public enum FluxUnionControlMode
{
    /// <summary>Canny edge map.</summary>
    Canny = 0,

    /// <summary>Tile / upscale conditioning.</summary>
    Tile = 1,

    /// <summary>Depth map.</summary>
    Depth = 2,

    /// <summary>Blurred image (deblur restoration).</summary>
    Blur = 3,

    /// <summary>OpenPose skeleton.</summary>
    Pose = 4,

    /// <summary>Grayscale image (recolorization).</summary>
    Gray = 5,

    /// <summary>Low-quality image restoration.</summary>
    LowQuality = 6,
}
