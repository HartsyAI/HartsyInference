using HartsyInference.Core.Layers.Attention;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>How a LongCat forward treats the leading latent frames: how many are conditioning, and where a reference frame sits
/// in RoPE time.</summary>
/// <param name="NumCondLatents">Leading latent frames that are conditioning (image, context or reference). They attend only their own group
/// and take no text update.</param>
/// <param name="NumRefLatents">0, or 1 when the first latent frame is a reference frame placed at <paramref name="RefImgIndex"/> in RoPE time
/// (avatar models).</param>
/// <param name="RefImgIndex">Temporal RoPE index of the reference frame; the remaining frames take indices <c>0, 1, …</c>. Required (not
/// negative) when <paramref name="NumRefLatents"/> is 1, and ignored otherwise. Use <see cref="Avatar"/> for the avatar layout.</param>
/// <param name="MaskFrameRange">Noisy frames within this many temporal indices of <paramref name="RefImgIndex"/> do not attend the reference;
/// 0 disables the skip.</param>
public readonly record struct LongCatConditioning(int NumCondLatents = 0, int NumRefLatents = 0, int RefImgIndex = -1, int MaskFrameRange = 0)
{
    /// <summary>The parameterless constructor is the no-reference split (<c>RefImgIndex</c> -1). <c>default(LongCatConditioning)</c> bypasses it and
    /// reads 0: harmless without a reference frame, but a <c>with</c> expression on it that adds one must name the index.</summary>
    public LongCatConditioning() : this(0, 0, -1, 0) { }

    /// <summary>No conditioning: text-to-video.</summary>
    public static LongCatConditioning None => new();

    /// <summary>One reference frame at RoPE index <paramref name="refIndex"/> (the shipped avatar uses 10), leading the
    /// <paramref name="numCond"/> conditioning latents.</summary>
    public static LongCatConditioning Avatar(int numCond, int refIndex = 10, int maskRange = 0) => new(numCond, 1, refIndex, maskRange);

    /// <summary>Throws unless the split fits <paramref name="frames"/> latent frames.</summary>
    public void Validate(int frames)
    {
        if (NumCondLatents < 0 || NumCondLatents > frames)
            throw new ArgumentOutOfRangeException(nameof(NumCondLatents), NumCondLatents, $"Conditioning latents must lie in [0, {frames}].");
        if (NumRefLatents is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(NumRefLatents), NumRefLatents, "The reference RoPE grid carries one reference slot.");
        if (NumRefLatents > NumCondLatents)
            throw new ArgumentException("The reference latent is one of the conditioning latents.", nameof(NumRefLatents));
        if (NumRefLatents > 0 && RefImgIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(RefImgIndex), RefImgIndex,
                "A reference frame needs its RoPE index (the avatar layout uses 10); use LongCatConditioning.Avatar or pass it explicitly.");
        if (MaskFrameRange < 0)
            throw new ArgumentOutOfRangeException(nameof(MaskFrameRange), MaskFrameRange, "The mask range is not negative.");
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
