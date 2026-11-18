namespace HartsyInference.Diffusion.Sampling;

/// <summary>Which prediction the CFG difference is taken from. <see cref="Uncond"/> is <c>u + w·(c − u)</c>, the form the
/// sampler's fused op applies at guidance <c>w</c>; <see cref="Cond"/> is <c>c + w·(c − u)</c> (Krea2, Chroma Radiance), which
/// the same op applies at guidance <c>w + 1</c>.</summary>
public enum GuidanceAnchor
{
    /// <summary>Classifier-free guidance as <c>w·cond + (1 − w)·uncond</c>.</summary>
    Uncond,

    /// <summary>Guidance as <c>cond + w·(cond − uncond)</c>.</summary>
    Cond,
}
