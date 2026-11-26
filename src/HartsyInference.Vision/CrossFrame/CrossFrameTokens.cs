using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.CrossFrame;

/// <summary>
/// Cross-frame attention without new parameters, as Depth Pro 2 runs it (upstream <c>MultiViewEncoder.cross_frame_attention</c>): the <c>T</c>
/// frames of a clip (or the views of one scene) go through an ordinary ViT block as one sequence, <c>[mean over frames of the prefix tokens, every
/// frame's patch tokens]</c>, with the per-frame patch RoPE repeated once per frame, and each frame keeps the prefix tokens it went in with. Frames
/// carry no position of their own, so the result is equivariant to the frame order. These helpers pack and unpack that sequence; the block itself
/// is the caller's. They run as backend ops (<see cref="IBackend.CopyStrided"/>, <see cref="IBackend.AccumulateStrided"/>), so on a GPU backend a
/// resident tensor is not read back to the host.
/// </summary>
public static class CrossFrameTokens
{
    /// <summary>
    /// The joint sequence <c>[1, prefix + T·N, C]</c> of <paramref name="frames"/> <c>[T, prefix + N, C]</c>: the prefix rows averaged over the
    /// frames, then every frame's patch rows in frame order. The caller owns the result.
    /// </summary>
    public static Tensor Join(IBackend backend, Tensor frames, int prefix)
    {
        ArgumentNullException.ThrowIfNull(backend);
        (int count, int seq, int width) = Check(frames, prefix);
        long patches = seq - prefix, prefixWidth = (long)prefix * width, patchWidth = patches * width, frameWidth = (long)seq * width;

        // Every output is written whole by its first op (a gather, a scale or a concat), so a GPU backend allocates it on the device instead of
        // uploading the host buffer first.
        Tensor allPatches = new(new TensorShape(1, count * patches, width), DType.F32);
        bool keep = false;
        try
        {
            backend.CopyStrided(allPatches, frames, StridedCopySpec.Gather(new StridedView([count, patchWidth], [frameWidth, 1], prefixWidth)));
            if (prefixWidth == 0)
            {
                keep = true;
                return allPatches;
            }

            // The mean of the prefix rows over the frames, summed in frame order: the first frame's rows, each later frame added in place.
            using Tensor sum = new(new TensorShape(1, prefix, width), DType.F32);
            backend.CopyStrided(sum, frames, StridedCopySpec.Gather(Flat(0, prefixWidth)));
            for (int t = 1; t < count; t++)
                backend.AccumulateStrided(sum, frames, new StridedCopySpec(Flat(0, prefixWidth), Flat(t * frameWidth, prefixWidth)));
            using Tensor mean = new(sum.Shape, DType.F32);
            backend.Scale(mean, sum, 1f / count);

            Tensor joint = new(new TensorShape(1, prefix + count * patches, width), DType.F32);
            try
            {
                backend.Concat(joint, [mean, allPatches], 1);
                return joint;
            }
            catch
            {
                joint.Dispose();
                throw;
            }
        }
        finally
        {
            if (!keep) allPatches.Dispose();
        }
    }

    /// <summary>
    /// The frames <c>[T, prefix + N, C]</c> of a joint block output <paramref name="joint"/> <c>[1, prefix + T·N, C]</c>: each frame's patch rows
    /// from it, and its prefix rows from <paramref name="input"/> (the frames as they went into the block). The caller owns the result.
    /// </summary>
    public static Tensor Split(IBackend backend, Tensor joint, Tensor input, int prefix)
    {
        ArgumentNullException.ThrowIfNull(backend);
        (int count, int seq, int width) = Check(input, prefix);
        long patches = seq - prefix, prefixWidth = (long)prefix * width, patchWidth = patches * width, frameWidth = (long)seq * width;
        if (joint.DType != DType.F32 || joint.Shape.Rank != 3 || joint.Shape[0] != 1 || joint.Shape[1] != prefix + count * patches
            || joint.Shape[2] != width)
            throw new ArgumentException(
                $"The joint output must be F32 [1, {prefix + count * patches}, {width}]; got {joint.DType} {joint.Shape}.", nameof(joint));
        // The first write copies the whole input (its prefix rows are the ones to keep), so the output is allocated on the device, not uploaded;
        // the patch rows then overwrite it in place.
        Tensor frames = new(input.Shape, DType.F32);
        try
        {
            backend.CopyStrided(frames, input, StridedCopySpec.Gather(Flat(0, input.ElementCount)));
            StridedView inFrames = new([count, patchWidth], [frameWidth, 1], prefixWidth);
            backend.CopyStrided(frames, joint, new StridedCopySpec(inFrames, new StridedView([count, patchWidth], [patchWidth, 1], prefixWidth)));
            return frames;
        }
        catch
        {
            frames.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Puts the first <paramref name="prefixRows"/> rows of every item of <paramref name="source"/> <c>[B, S, C]</c> over those of
    /// <paramref name="target"/> (the same shape), in place: the prefix a block is to leave as it found it.
    /// </summary>
    public static void RestorePrefix(IBackend backend, Tensor target, Tensor source, int prefixRows)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(target);
        ArgumentNullException.ThrowIfNull(source);
        if (target.DType != DType.F32 || source.DType != DType.F32 || target.Shape.Rank != 3 || !target.Shape.Equals(source.Shape) || prefixRows < 0
            || prefixRows > target.Shape[1])
            throw new ArgumentException(
                $"Target and source must be F32 [B, S, C] of one shape with at most S prefix rows; got {target.Shape}, {source.Shape}, {prefixRows}.",
                nameof(target));
        if (prefixRows == 0) return;
        long batch = target.Shape[0], frameWidth = target.Shape[1] * target.Shape[2], prefixWidth = prefixRows * target.Shape[2];
        StridedView view = new([batch, prefixWidth], [frameWidth, 1], 0);
        backend.CopyStrided(target, source, new StridedCopySpec(view, view));
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

    private static StridedView Flat(long offset, long count) => new([count], [1], offset);

    private static (int Count, int Seq, int Width) Check(Tensor frames, int prefix)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (frames.DType != DType.F32 || frames.Shape.Rank != 3 || frames.Shape[0] < 1 || frames.Shape[1] <= prefix || prefix < 0)
            throw new ArgumentException(
                $"Frames must be F32 [T, prefix + N, C] with N > 0 past {prefix} prefix rows; got {frames.DType} {frames.Shape}.", nameof(frames));
        return ((int)frames.Shape[0], (int)frames.Shape[1], (int)frames.Shape[2]);
    }
}
