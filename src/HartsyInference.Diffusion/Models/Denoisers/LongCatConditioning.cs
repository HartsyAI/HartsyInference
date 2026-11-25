using HartsyInference.Core.Layers.Attention;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>How a LongCat forward treats the leading latent frames: how many are conditioning, and where a reference frame sits
/// in RoPE time.</summary>
/// <param name="NumCondLatents">Leading latent frames that are conditioning (image, context or reference). They attend only their own group
/// and take no text update.</param>
/// <param name="NumRefLatents">0, or 1 when the first latent frame is a reference frame placed at <paramref name="RefImgIndex"/> in RoPE time
/// (avatar models).</param>
/// <param name="RefImgIndex">Temporal RoPE index of the reference frame; the remaining frames take indices <c>0, 1, …</c>.</param>
/// <param name="MaskFrameRange">Noisy frames within this many temporal indices of <paramref name="RefImgIndex"/> do not attend the reference;
/// 0 disables the skip.</param>
public readonly record struct LongCatConditioning(int NumCondLatents = 0, int NumRefLatents = 0, int RefImgIndex = 10, int MaskFrameRange = 0)
{
    /// <summary>No conditioning: text-to-video.</summary>
    public static LongCatConditioning None => default;

    /// <summary>Throws unless the split fits <paramref name="frames"/> latent frames.</summary>
    public void Validate(int frames)
    {
        if (NumCondLatents < 0 || NumCondLatents > frames)
            throw new ArgumentOutOfRangeException(nameof(NumCondLatents), NumCondLatents, $"Conditioning latents must lie in [0, {frames}].");
        if (NumRefLatents is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(NumRefLatents), NumRefLatents, "The reference RoPE grid carries one reference slot.");
        if (NumRefLatents > NumCondLatents)
            throw new ArgumentException("The reference latent is one of the conditioning latents.", nameof(NumRefLatents));
        if (RefImgIndex < 0 || MaskFrameRange < 0)
            throw new ArgumentOutOfRangeException(nameof(RefImgIndex), "The reference index and the mask range are not negative.");
    }

    /// <summary>The timestep of each latent frame: <paramref name="timestep"/> for noisy frames, 0 for the conditioning frames.</summary>
    public float[] FrameTimesteps(float timestep, int frames)
    {
        Validate(frames);
        float[] steps = new float[frames];
        for (int i = NumCondLatents; i < frames; i++)
            steps[i] = timestep;
        return steps;
    }

    /// <summary>The self-attention plan of this split over <paramref name="frames"/> frames of <paramref name="tokensPerFrame"/> tokens.</summary>
    public AttentionSegment[] AttentionPlan(int tokensPerFrame, int frames)
    {
        Validate(frames);
        return NumRefLatents > 0
            ? ConditionedAttentionPlan.ForReferencedFrames(tokensPerFrame, frames, NumCondLatents, NumRefLatents, RefImgIndex, MaskFrameRange)
            : ConditionedAttentionPlan.ForConditionedFrames(tokensPerFrame, frames, NumCondLatents);
    }
}
