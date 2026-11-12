namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Whether an Iris-Z text-to-image run also returns depth, and from where.</summary>
public enum IrisZDepthOutput
{
    /// <summary>Image only: the depth grid stays noise and is never denoised.</summary>
    None,

    /// <summary>Image and depth are denoised together, then the depth is estimated again from the finished image.</summary>
    Estimated,

    /// <summary>Image and depth are denoised together and the joint depth is returned.</summary>
    Joint,
}
