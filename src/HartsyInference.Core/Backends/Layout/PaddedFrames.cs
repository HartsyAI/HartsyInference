using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Layout;

/// <summary>The frame-major padded input of a batched CausalConv3d, composed from strided copies on the backend. It writes the same elements as the
/// host loop in <see cref="IBackend.BuildPaddedFrames"/>, with no host read: a backend without a device version of that op can use it, and the copies
/// run on the backend's own copy kernel. Covers edge-clamped spatial padding, with replicate or zero temporal padding. Reflected spatial padding has no
/// strided form, so it stays on the host path.
/// <para>The layouts are the host loop's: <c>input</c> and <c>cache</c> are channel-major <c>[1, cIn, T, H, W]</c>; <c>padded</c> is frame-major
/// <c>[paddedT, cIn, H + 2·padH, W + 2·padW]</c>. The output is split into a 3-by-3 grid of spatial blocks (top pad, interior, bottom pad by left pad,
/// interior, right pad) and, per temporal segment, one copy per grid cell. A pad block reads one source row or column with a zero stride, which is
/// the edge clamp.</para></summary>
public static class PaddedFrames
{
    /// <summary>Fills <paramref name="padded"/> the way the host loop does, on <paramref name="backend"/>.</summary>
    public static void Compose(IBackend backend, Tensor padded, Tensor input, Tensor? cache, int zeroPad, bool replicateFirst, int padH, int padW)
    {
        int paddedT = (int)padded.Shape[0];
        int cIn = (int)padded.Shape[1];
        int hp = (int)padded.Shape[2];
        int wp = (int)padded.Shape[3];
        int tin = (int)input.Shape[2];
        int h = (int)input.Shape[3];
        int w = (int)input.Shape[4];
        int cacheLen = cache is null ? 0 : (int)cache.Shape[2];
        Geometry g = new(paddedT, cIn, hp, wp, h, w, padH, padW);

        // The host loop writes zeros for the temporal pad when it does not replicate, and overwrites everything else, so zero first.
        using Tensor zero = new(new TensorShape(1), DType.F32);
        zero.AsSpan<float>()[0] = 0f;
        StridedCopies.FillFrom(backend, padded, zero);

        // Output frame t takes source frame (t - zeroPad) of the cache followed by the input, and the input's last frame past its end. The replicate
        // pad takes frame 0 of the cache when there is one, else of the input.
        if (replicateFirst && zeroPad > 0)
        {
            int n = Math.Min(zeroPad, paddedT);
            if (cacheLen > 0) Segment(backend, g, padded, cache!, cacheLen, srcFrame: 0, outT: 0, len: n, repeatFrame: true);
            else Segment(backend, g, padded, input, tin, srcFrame: 0, outT: 0, len: n, repeatFrame: true);
        }
        int cacheStart = zeroPad;
        int cacheCount = Math.Min(cacheLen, Math.Max(0, paddedT - cacheStart));
        if (cacheCount > 0) Segment(backend, g, padded, cache!, cacheLen, srcFrame: 0, outT: cacheStart, len: cacheCount, repeatFrame: false);

        int inputStart = zeroPad + cacheLen;
        int inputCount = Math.Min(tin, Math.Max(0, paddedT - inputStart));
        if (inputCount > 0) Segment(backend, g, padded, input, tin, srcFrame: 0, outT: inputStart, len: inputCount, repeatFrame: false);

        int clampStart = inputStart + tin;
        int clampCount = Math.Max(0, paddedT - clampStart);
        if (clampCount > 0 && tin > 0) Segment(backend, g, padded, input, tin, srcFrame: tin - 1, outT: clampStart, len: clampCount, repeatFrame: true);
    }

    /// <summary>Writes output frames <c>[outT, outT + len)</c> from source frames of <paramref name="src"/> (channel-major with
    /// <paramref name="srcT"/> frames), starting at <paramref name="srcFrame"/>. With <paramref name="repeatFrame"/> every output frame reads that one
    /// source frame.</summary>
    private static void Segment(IBackend backend, Geometry g, Tensor padded, Tensor src, int srcT, int srcFrame, int outT, int len, bool repeatFrame)
    {
        long hw = (long)g.H * g.W;
        long srcCStride = srcT * hw;
        long srcFrameStride = repeatFrame ? 0 : hw;
        long dstFrame = (long)g.CIn * g.Hp * g.Wp;

        foreach (Band rows in g.RowBands)
        {
            foreach (Band cols in g.ColBands)
            {
                if (rows.Count == 0 || cols.Count == 0) continue;
                long[] dims = [len, g.CIn, rows.Count, cols.Count];
                long[] dstStrides = [dstFrame, (long)g.Hp * g.Wp, g.Wp, 1];
                long dstOffset = (long)outT * dstFrame + (long)rows.DstStart * g.Wp + cols.DstStart;
                long[] srcStrides = [srcFrameStride, srcCStride, rows.SrcStride * g.W, cols.SrcStride];
                long srcOffset = (long)srcFrame * hw + rows.SrcStart * g.W + cols.SrcStart;
                StridedView destination = new(dims, dstStrides, dstOffset);
                StridedView source = new(dims, srcStrides, srcOffset);
                backend.CopyStrided(padded, src, new StridedCopySpec(destination, source));
            }
        }
    }

    /// <summary>One band of the spatial grid: the destination rows or columns it writes, and the source ones it reads. A zero <see cref="SrcStride"/> repeats
    /// the single source row or column <see cref="SrcStart"/> (the edge clamp).</summary>
    private readonly record struct Band(int DstStart, int Count, int SrcStart, int SrcStride);

    /// <summary>The padded and unpadded extents and the spatial bands the output splits into.</summary>
    private readonly struct Geometry
    {
        public Geometry(int paddedT, int cIn, int hp, int wp, int h, int w, int padH, int padW)
        {
            CIn = cIn; Hp = hp; Wp = wp; H = h; W = w;
            RowBands = Bands(h, padH);
            ColBands = Bands(w, padW);
        }

        public int CIn { get; }
        public int Hp { get; }
        public int Wp { get; }
        public int H { get; }
        public int W { get; }
        public Band[] RowBands { get; }
        public Band[] ColBands { get; }

        /// <summary>Top pad (the first source row), interior (rows 0 to n − 1), bottom pad (the last source row); a pad of zero rows is an empty band.</summary>
        private static Band[] Bands(int n, int pad) =>
        [
            new Band(DstStart: 0, Count: pad, SrcStart: 0, SrcStride: 0),
            new Band(DstStart: pad, Count: n, SrcStart: 0, SrcStride: 1),
            new Band(DstStart: pad + n, Count: pad, SrcStart: n - 1, SrcStride: 0),
        ];
    }
}
