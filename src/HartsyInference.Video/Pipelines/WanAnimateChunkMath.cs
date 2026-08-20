namespace HartsyInference.Video.Pipelines;

/// <summary>Pure arithmetic of Wan-Animate's chunked extension (ComfyUI <c>WanAnimateToVideo</c>'s
/// <c>continue_motion</c> path): how many frames of the previous chunk become the motion prefix, how many latent
/// frames that prefix occupies, how many decoded frames the chunk must drop, and how the driving-video seek advances.
/// Shared by the pipeline (concat-mask range) and the recipe chunk loop so both read one set of rules.</summary>
public static class WanAnimateChunkMath
{
    /// <summary>Wan VAE temporal compression — the frame grid everything here is built on.</summary>
    public const int TemporalStep = 4;

    /// <summary>Snaps a motion-prefix frame count DOWN onto the <c>4n+1</c> grid (0 stays 0, 1..4 → 1). Off-grid
    /// prefixes make <see cref="TrimImageFrames"/> under-trim and let the background overwrite prefix frames.</summary>
    public static int SnapMotionFrames(int frames) =>
        frames <= 0 ? 0 : 1 + (frames - 1) / TemporalStep * TemporalStep;

    /// <summary>The motion prefix a continuation chunk carries: the request's count clamped by what has actually been
    /// generated so far and by <c>chunkFrames - 1</c> (a chunk must produce at least one new frame), then snapped.</summary>
    public static int MotionPrefixFrames(int requestedFrames, int availableFrames, int chunkFrames) =>
        SnapMotionFrames(Math.Min(Math.Min(requestedFrames, availableFrames), chunkFrames - 1));

    /// <summary>Latent frames the prefix occupies — ComfyUI's <c>((N - 1) // 4) + 1</c>.</summary>
    public static int RefMotionLatentLength(int motionFrames) =>
        motionFrames <= 0 ? 0 : (motionFrames - 1) / TemporalStep + 1;

    /// <summary>Leading pixel frames to drop from a chunk's decoded output — ComfyUI's <c>ref_images_num</c>,
    /// <c>max(0, ref·4 - 3)</c>. Equals the prefix length exactly when the prefix is on the <c>4n+1</c> grid.</summary>
    public static int TrimImageFrames(int refMotionLatentLength) =>
        Math.Max(0, refMotionLatentLength * TemporalStep - 3);

    /// <summary>Exclusive end, in flat pixel-frame index <c>j = (t - trim)·4 + m</c>, of the concat-mask range the
    /// prefix marks known. Without a character mask that is <c>ref·4</c> (node line 1221); with one it is
    /// <c>ref_images_num</c>, because the node then overwrites <c>[ref_images_num, ...)</c> with mask values.</summary>
    public static int MaskPrefixEnd(int refMotionLatentLength, bool hasCharacterMask) =>
        hasCharacterMask ? TrimImageFrames(refMotionLatentLength) : refMotionLatentLength * TemporalStep;

    /// <summary>Whether concat-mask channel <paramref name="maskChannel"/> of latent frame <paramref name="latentFrame"/>
    /// is known in model space (post the WAN21 <c>1 − mask</c> inversion): the leading reference frame(s), or a flat
    /// pixel-frame index <c>j = (t − trim)·4 + m</c> below <paramref name="prefixEnd"/>.</summary>
    public static bool IsKnownMaskCell(int latentFrame, int maskChannel, int trimLatent, int prefixEnd) =>
        latentFrame < trimLatent || (latentFrame - trimLatent) * TemporalStep + maskChannel < prefixEnd;

    /// <summary>Driving-video seek for a chunk: the carried offset rewound by the prefix, floored at 0. The rewind
    /// is what re-drives the prefix with the frames that produced it.</summary>
    public static int SliceOffset(int carriedOffset, int motionFrames) => Math.Max(0, carriedOffset - motionFrames);

    /// <summary>The offset handed to the next chunk — this chunk's seek plus its full length.</summary>
    public static int NextCarriedOffset(int sliceOffset, int chunkFrames) => sliceOffset + chunkFrames;

    /// <summary>Chunks needed to reach <paramref name="totalFrames"/>; each chunk after the first contributes
    /// <c>chunkFrames - trim_image</c> new frames.</summary>
    public static int ChunkCount(int totalFrames, int chunkFrames, int motionFrames)
    {
        int newPerChunk = chunkFrames - TrimImageFrames(RefMotionLatentLength(motionFrames));
        if (newPerChunk < 1)
            throw new ArgumentOutOfRangeException(nameof(motionFrames),
                $"A {motionFrames}-frame motion prefix leaves no new frames in a {chunkFrames}-frame chunk.");
        int remaining = Math.Max(0, totalFrames - chunkFrames);
        return 1 + (remaining + newPerChunk - 1) / newPerChunk;
    }
}
