using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.CrossFrame;

/// <summary>
/// Cross-frame attention without new parameters, as Depth Pro 2 runs it (upstream <c>MultiViewEncoder.cross_frame_attention</c>): the <c>T</c>
/// frames of a clip (or the views of one scene) go through an ordinary ViT block as one sequence, <c>[mean over frames of the prefix tokens, every
/// frame's patch tokens]</c>, with the per-frame patch RoPE repeated once per frame, and each frame keeps the prefix tokens it went in with. Frames
/// carry no position of their own, so the result is equivariant to the frame order. These helpers pack and unpack that sequence; the block itself
/// is the caller's.
/// </summary>
public static class CrossFrameTokens
{
    /// <summary>
    /// The joint sequence <c>[1, prefix + T·N, C]</c> of <paramref name="frames"/> <c>[T, prefix + N, C]</c>: the prefix rows averaged over the
    /// frames, then every frame's patch rows in frame order. The caller owns the result.
    /// </summary>
    public static Tensor Join(Tensor frames, int prefix)
    {
        (int count, int seq, int width) = Check(frames, prefix);
        int patches = seq - prefix;
        Tensor joint = new(new TensorShape(1, prefix + (long)count * patches, width), DType.F32);
        ReadOnlySpan<float> src = frames.AsReadOnlySpan<float>();
        Span<float> dst = joint.AsSpan<float>();
        int prefixWidth = prefix * width;
        Span<float> mean = dst[..prefixWidth];
        mean.Clear();
        for (int t = 0; t < count; t++)
        {
            ReadOnlySpan<float> rows = src.Slice(t * seq * width, prefixWidth);
            for (int i = 0; i < prefixWidth; i++) mean[i] += rows[i];
        }

        for (int i = 0; i < prefixWidth; i++) mean[i] /= count;
        int patchWidth = patches * width;
        for (int t = 0; t < count; t++)
            src.Slice(t * seq * width + prefixWidth, patchWidth).CopyTo(dst.Slice(prefixWidth + t * patchWidth, patchWidth));
        return joint;
    }

    /// <summary>
    /// The frames <c>[T, prefix + N, C]</c> of a joint block output <paramref name="joint"/> <c>[1, prefix + T·N, C]</c>: each frame's patch rows
    /// from it, and its prefix rows from <paramref name="input"/> (the frames as they went into the block). The caller owns the result.
    /// </summary>
    public static Tensor Split(Tensor joint, Tensor input, int prefix)
    {
        (int count, int seq, int width) = Check(input, prefix);
        int patches = seq - prefix, prefixWidth = prefix * width, patchWidth = patches * width;
        if (joint.DType != DType.F32 || joint.Shape.Rank != 3 || joint.Shape[0] != 1 || joint.Shape[1] != prefix + (long)count * patches
            || joint.Shape[2] != width)
            throw new ArgumentException(
                $"The joint output must be F32 [1, {prefix + (long)count * patches}, {width}]; got {joint.DType} {joint.Shape}.", nameof(joint));
        Tensor frames = new(input.Shape, DType.F32);
        ReadOnlySpan<float> j = joint.AsReadOnlySpan<float>(), x = input.AsReadOnlySpan<float>();
        Span<float> dst = frames.AsSpan<float>();
        for (int t = 0; t < count; t++)
        {
            int frame = t * seq * width;
            x.Slice(frame, prefixWidth).CopyTo(dst.Slice(frame, prefixWidth));
            j.Slice(prefixWidth + t * patchWidth, patchWidth).CopyTo(dst.Slice(frame + prefixWidth, patchWidth));
        }

        return frames;
    }

    /// <summary>
    /// A RoPE table for the joint sequence: <paramref name="prefix"/> rows of <paramref name="prefixValue"/> (1 for the cosine table, 0 for the
    /// sine table, so the prefix is not rotated), then the per-frame patch table <paramref name="patchTable"/> <c>[1, N, D]</c> repeated
    /// <paramref name="frames"/> times. The caller owns the result.
    /// </summary>
    public static Tensor TileRope(Tensor patchTable, int frames, int prefix, float prefixValue)
    {
        ArgumentNullException.ThrowIfNull(patchTable);
        ArgumentOutOfRangeException.ThrowIfLessThan(frames, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(prefix);
        if (patchTable.DType != DType.F32 || patchTable.Shape.Rank != 3 || patchTable.Shape[0] != 1)
            throw new ArgumentException($"The patch table must be F32 [1, N, D]; got {patchTable.DType} {patchTable.Shape}.", nameof(patchTable));
        int rows = (int)patchTable.Shape[1], dim = (int)patchTable.Shape[2];
        Tensor table = new(new TensorShape(1, prefix + (long)frames * rows, dim), DType.F32);
        Span<float> dst = table.AsSpan<float>();
        dst[..(prefix * dim)].Fill(prefixValue);
        ReadOnlySpan<float> src = patchTable.AsReadOnlySpan<float>();
        for (int t = 0; t < frames; t++) src.CopyTo(dst.Slice(prefix * dim + t * rows * dim, rows * dim));
        return table;
    }

    private static (int Count, int Seq, int Width) Check(Tensor frames, int prefix)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (frames.DType != DType.F32 || frames.Shape.Rank != 3 || frames.Shape[0] < 1 || frames.Shape[1] <= prefix || prefix < 0)
            throw new ArgumentException(
                $"Frames must be F32 [T, prefix + N, C] with N > 0 past {prefix} prefix rows; got {frames.DType} {frames.Shape}.", nameof(frames));
        return ((int)frames.Shape[0], (int)frames.Shape[1], (int)frames.Shape[2]);
    }
}
