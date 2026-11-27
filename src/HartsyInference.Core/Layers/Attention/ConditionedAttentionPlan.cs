namespace HartsyInference.Core.Layers.Attention;

/// <summary>Builds the <see cref="AttentionSegment"/> lists of conditioning-aware video self-attention: tokens of conditioning frames attend
/// only their own group, tokens of noisy frames attend everything, and a reference frame can be hidden from a window of noisy frames.
/// Tokens are laid out frame-major with <c>tokensPerFrame</c> rows per latent frame.</summary>
public static class ConditionedAttentionPlan
{
    /// <summary>One group of conditioning frames at the front (LongCat-Video, and the avatar model when the conditioning is a single frame):
    /// <c>condFrames</c> frames attend only themselves, the rest attend all.</summary>
    public static AttentionSegment[] ForConditionedFrames(int tokensPerFrame, int frames, int condFrames)
    {
        ValidateGeometry(tokensPerFrame, frames, condFrames);
        int total = tokensPerFrame * frames;
        if (condFrames == 0)
            return [AttentionSegment.Dense(total)];
        int cond = tokensPerFrame * condFrames;
        return condFrames == frames
            ? [new AttentionSegment(0, cond, 0, cond)]
            : [new AttentionSegment(0, cond, 0, cond), new AttentionSegment(cond, total - cond, 0, total)];
    }

    /// <summary>The avatar layout with one reference frame in front of the context frames (continuation): the reference attends only itself,
    /// the context frames attend only themselves, and the noisy frames attend everything, except that noisy frames whose temporal RoPE index
    /// lies within <paramref name="maskFrameRange"/> of <paramref name="refImgIndex"/> do not see the reference. A skip window that does not
    /// fit inside the noisy frames is ignored, as in the reference, and those frames attend everything.</summary>
    /// <param name="condFrames">Frames before the noisy ones, the reference included.</param>
    /// <param name="refFrames">Reference frames in front: 0 or 1.</param>
    public static AttentionSegment[] ForReferencedFrames(int tokensPerFrame, int frames, int condFrames, int refFrames, int refImgIndex,
        int maskFrameRange)
    {
        ValidateGeometry(tokensPerFrame, frames, condFrames);
        if (refFrames is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(refFrames), refFrames, "The reference RoPE grid carries one reference slot.");
        if (refFrames > condFrames)
            throw new ArgumentException("The reference is part of the conditioning frames.", nameof(refFrames));
        if (condFrames <= 1)
            return ForConditionedFrames(tokensPerFrame, frames, condFrames);
        if (refFrames == 0)
            throw new ArgumentException(
                "A multi-frame context needs a reference slot; the reference grid places the context after it.", nameof(refFrames));

        int total = tokensPerFrame * frames;
        int refRows = tokensPerFrame * refFrames;
        int cond = tokensPerFrame * condFrames;
        List<AttentionSegment> plan =
            [new AttentionSegment(0, refRows, 0, refRows), new AttentionSegment(refRows, cond - refRows, refRows, cond - refRows)];
        int noisyFrames = frames - condFrames;
        if (noisyFrames == 0)
            return [.. plan];

        int start = 0, end = 0;
        if (maskFrameRange > 0)
        {
            start = refImgIndex - maskFrameRange - condFrames + refFrames;
            end = refImgIndex + maskFrameRange - condFrames + refFrames + 1;
        }
        if (start >= 0 && end > start && end <= noisyFrames)
        {
            int startRow = cond + start * tokensPerFrame, endRow = cond + end * tokensPerFrame;
            if (startRow > cond)
                plan.Add(new AttentionSegment(cond, startRow - cond, 0, total));
            plan.Add(new AttentionSegment(startRow, endRow - startRow, refRows, total - refRows));
            if (endRow < total)
                plan.Add(new AttentionSegment(endRow, total - endRow, 0, total));
        }
        else
        {
            plan.Add(new AttentionSegment(cond, total - cond, 0, total));
        }
        return [.. plan];
    }

    private static void ValidateGeometry(int tokensPerFrame, int frames, int condFrames)
    {
        if (tokensPerFrame <= 0 || frames <= 0)
            throw new ArgumentException($"A plan needs positive geometry; got {tokensPerFrame} tokens per frame over {frames} frames.");
        if (condFrames < 0 || condFrames > frames)
            throw new ArgumentOutOfRangeException(nameof(condFrames), condFrames, $"Conditioning frames must lie in [0, {frames}].");
    }
}
