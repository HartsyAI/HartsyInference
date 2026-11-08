using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Layout;

/// <summary>Depth-to-space and space-to-depth as strided copies: every pixel shuffle, patchify and unpatchify is a
/// permutation of eight axes <c>[B, C, T, iT, H, iH, W, iW]</c> (grid cell and sub-position per factor), so one
/// <see cref="IBackend.CopyStrided"/> moves it, and a dropped frame costs one more. The <see cref="ShuffleSpec"/>
/// names the layouts, factors and channel nesting; contract in <c>docs/BACKEND_OPS.md</c>.</summary>
public static class PixelShuffles
{
    private const int AxisBatch = 0, AxisChannel = 1, AxisFrame = 2, AxisSubT = 3, AxisHeight = 4, AxisSubH = 5, AxisWidth = 6, AxisSubW = 7;

    /// <summary>Unpacks <paramref name="packed"/> into <paramref name="unpacked"/>: channels become sub-positions of a grid scaled by the factors.</summary>
    public static void DepthToSpace(IBackend backend, Tensor unpacked, Tensor packed, in ShuffleSpec spec)
    {
        Validate(spec, unpacked, packed);
        StridedView source = PackedView(spec, packed);
        if (spec.DropLeadingFrames > 0 && spec.DropDuplicateFrame)
            throw new NotSupportedException("A shuffle drops leading frames or the duplicate frame, not both.");
        long firstFrame = (long)spec.ChunkStartFrame * spec.FactorT - spec.DropLeadingFrames;
        if (spec.DropDuplicateFrame && spec.ChunkStartFrame == 0)
        {
            // Frame 0 stays, frame 1 goes, frames 2.. shift down by one.
            CopyFrames(backend, unpacked, packed, spec, source, tStart: 0, tCount: 1, subLo: 0, subCount: 1, outFrame: 0);
            if (spec.FactorT > 2)
                CopyFrames(backend, unpacked, packed, spec, source, tStart: 0, tCount: 1, subLo: 2, subCount: spec.FactorT - 2, outFrame: 1);
            if (spec.Frames > 1)
                CopyFrames(backend, unpacked, packed, spec, source, tStart: 1, tCount: spec.Frames - 1, subLo: 0, subCount: spec.FactorT, outFrame: spec.FactorT - 1);
            return;
        }
        if (spec.DropDuplicateFrame) firstFrame -= 1;
        if (firstFrame < 0)
        {
            // The first packed frame loses its leading sub-frames; every later frame is whole.
            int skipped = (int)-firstFrame;
            if (skipped >= spec.FactorT)
                throw new ArgumentException($"Dropping {spec.DropLeadingFrames} leading frames leaves no frame of the first packed frame; the factor is {spec.FactorT}.", nameof(spec));
            CopyFrames(backend, unpacked, packed, spec, source, tStart: 0, tCount: 1, subLo: skipped, subCount: spec.FactorT - skipped, outFrame: 0);
            if (spec.Frames > 1)
                CopyFrames(backend, unpacked, packed, spec, source, tStart: 1, tCount: spec.Frames - 1, subLo: 0, subCount: spec.FactorT, outFrame: spec.FactorT - skipped);
            return;
        }
        CopyFrames(backend, unpacked, packed, spec, source, tStart: 0, tCount: spec.Frames, subLo: 0, subCount: spec.FactorT, outFrame: firstFrame);
    }

    /// <summary>Packs <paramref name="unpacked"/> into <paramref name="packed"/>: the inverse of <see cref="DepthToSpace"/> for a spec without repeats, drops or chunks.</summary>
    public static void SpaceToDepth(IBackend backend, Tensor packed, Tensor unpacked, in ShuffleSpec spec)
    {
        Validate(spec, unpacked, packed);
        if (spec.ChannelRepeat != 1 || spec.DropLeadingFrames != 0 || spec.DropDuplicateFrame || spec.ChunkStartFrame != 0)
            throw new NotSupportedException("SpaceToDepth takes a plain shuffle: no channel repeat, dropped frames or chunking.");
        StridedView destination = PackedView(spec, packed);
        StridedView source = UnpackedView(spec, unpacked).Slice(AxisFrame, 0, (long)spec.Frames * spec.FactorT).Split(AxisFrame, spec.Frames, spec.FactorT);
        backend.CopyStrided(packed, unpacked, new StridedCopySpec(destination, source));
    }

    /// <summary>DiT unpatchify: tokens <c>[B, hP·wP, C·p²]</c> (any leading shape with that element count) to the image <c>[B, C, hP·p, wP·p]</c>.</summary>
    public static void UnpatchifyTokens(IBackend backend, Tensor output, Tensor tokens, int channels, int hPacked, int wPacked, int patch, bool innerChannelFastest)
    {
        if (output.Shape.Rank != 4)
            throw new ArgumentException($"UnpatchifyTokens output must be [B, C, H, W]; got {output.Shape}.", nameof(output));
        long height = (long)hPacked * patch, width = (long)wPacked * patch;
        if (output.Shape[1] != channels || output.Shape[2] != height || output.Shape[3] != width)
            throw new ArgumentException($"UnpatchifyTokens output must be [B, {channels}, {height}, {width}]; got {output.Shape}.", nameof(output));
        DepthToSpace(backend, output, tokens, ShuffleSpec.TokenGrid((int)output.Shape[0], hPacked, wPacked, channels, patch, innerChannelFastest));
    }

    /// <summary>DiT patchify: the image <c>[B, C, H, W]</c> to tokens <c>[B, (H/p)·(W/p), p²·C]</c>.</summary>
    public static void PatchifyTokens(IBackend backend, Tensor output, Tensor input, int patch, bool innerChannelFastest)
    {
        if (input.Shape.Rank != 4)
            throw new ArgumentException($"PatchifyTokens input must be [B, C, H, W]; got {input.Shape}.", nameof(input));
        if (patch <= 0 || input.Shape[2] % patch != 0 || input.Shape[3] % patch != 0)
            throw new ArgumentException($"PatchifyTokens needs H and W divisible by patch {patch}; got {input.Shape}.", nameof(patch));
        SpaceToDepth(backend, output, input, ShuffleSpec.TokenGrid((int)input.Shape[0], (int)(input.Shape[2] / patch), (int)(input.Shape[3] / patch), (int)input.Shape[1], patch, innerChannelFastest));
    }

    /// <summary>One copy of packed frames <c>[tStart, tStart + tCount)</c> × sub-frames <c>[subLo, subLo + subCount)</c> into unpacked frames from <paramref name="outFrame"/>.</summary>
    private static void CopyFrames(IBackend backend, Tensor unpacked, Tensor packed, in ShuffleSpec spec, in StridedView source,
        long tStart, long tCount, long subLo, long subCount, long outFrame)
    {
        StridedView from = source.Slice(AxisFrame, tStart, tCount).Slice(AxisSubT, subLo, subCount);
        StridedView to = UnpackedView(spec, unpacked).Slice(AxisFrame, outFrame, tCount * subCount).Split(AxisFrame, tCount, subCount);
        backend.CopyStrided(unpacked, packed, new StridedCopySpec(to, from));
    }

    /// <summary>The packed tensor as the canonical eight axes, its stored channel axis split per the order and repeat.</summary>
    private static StridedView PackedView(in ShuffleSpec spec, Tensor packed)
    {
        long stored = spec.StoredChannels;
        long cells = (long)spec.Frames * spec.Height * spec.Width;
        Span<long> dims = stackalloc long[8];
        Span<long> strides = stackalloc long[8];
        long channelStride;
        if (spec.PackedLayout == ShuffleLayout.ChannelsFirst)
        {
            // [B, stored, T, H, W]
            channelStride = cells;
            Set(dims, strides, AxisBatch, spec.Batch, stored * cells);
            Set(dims, strides, AxisFrame, spec.Frames, (long)spec.Height * spec.Width);
            Set(dims, strides, AxisHeight, spec.Height, spec.Width);
            Set(dims, strides, AxisWidth, spec.Width, 1);
        }
        else
        {
            // [B·T·H·W, stored]
            channelStride = 1;
            Set(dims, strides, AxisBatch, spec.Batch, cells * stored);
            Set(dims, strides, AxisFrame, spec.Frames, (long)spec.Height * spec.Width * stored);
            Set(dims, strides, AxisHeight, spec.Height, spec.Width * stored);
            Set(dims, strides, AxisWidth, spec.Width, stored);
        }
        // Packed channel = mixed radix over the ordered axes; the stored channel is packed / repeat. An axis whose place
        // value is a multiple of the repeat steps the stored channel; one the repeat swallows whole is a broadcast.
        long place = 1;
        for (int position = 3; position >= 0; position--)
        {
            ShuffleAxis axis = spec.Order[position];
            long extent = spec.Extent(axis);
            long stride;
            if (place % spec.ChannelRepeat == 0) stride = place / spec.ChannelRepeat * channelStride;
            else if (spec.ChannelRepeat % (place * extent) == 0) stride = 0;
            else throw new NotSupportedException($"A channel repeat of {spec.ChannelRepeat} straddles the {axis} axis of extent {extent} at place value {place}; the shuffle is not one strided copy.");
            Set(dims, strides, CanonicalAxis(axis), extent, stride);
            place *= extent;
        }
        return new StridedView(dims, strides);
    }

    /// <summary>The unpacked tensor as the canonical axes with the frame axis still whole (its sub-frame axis absent); the caller slices and splits it.</summary>
    private static StridedView UnpackedView(in ShuffleSpec spec, Tensor unpacked)
    {
        long outT = spec.UnpackedFrameCount;
        long outH = (long)spec.Height * spec.FactorH, outW = (long)spec.Width * spec.FactorW;
        Span<long> dims = stackalloc long[7];
        Span<long> strides = stackalloc long[7];
        // Positions once the frame axis is split: [B, C, T, iT, H, iH, W, iW] minus iT → [B, C, F, H, iH, W, iW].
        if (spec.UnpackedLayout == ShuffleLayout.ChannelsFirst)
        {
            // [B, C, outT, outH, outW]
            long frame = outH * outW;
            Set(dims, strides, 0, spec.Batch, spec.Channels * outT * frame);
            Set(dims, strides, 1, spec.Channels, outT * frame);
            Set(dims, strides, 2, outT, frame);
            Set(dims, strides, 3, spec.Height, (long)spec.FactorH * outW);
            Set(dims, strides, 4, spec.FactorH, outW);
            Set(dims, strides, 5, spec.Width, spec.FactorW);
            Set(dims, strides, 6, spec.FactorW, 1);
        }
        else
        {
            // [B·outT·outH·outW, C]
            long c = spec.Channels;
            Set(dims, strides, 0, spec.Batch, outT * outH * outW * c);
            Set(dims, strides, 1, spec.Channels, 1);
            Set(dims, strides, 2, outT, outH * outW * c);
            Set(dims, strides, 3, spec.Height, (long)spec.FactorH * outW * c);
            Set(dims, strides, 4, spec.FactorH, outW * c);
            Set(dims, strides, 5, spec.Width, spec.FactorW * c);
            Set(dims, strides, 6, spec.FactorW, c);
        }
        return new StridedView(dims, strides);
    }

    private static int CanonicalAxis(ShuffleAxis axis) => axis switch
    {
        ShuffleAxis.Channel => AxisChannel,
        ShuffleAxis.Time => AxisSubT,
        ShuffleAxis.Height => AxisSubH,
        _ => AxisSubW,
    };

    private static void Set(Span<long> dims, Span<long> strides, int axis, long extent, long stride)
    {
        dims[axis] = extent;
        strides[axis] = stride;
    }

    private static void Validate(in ShuffleSpec spec, Tensor unpacked, Tensor packed)
    {
        if (spec.Batch <= 0 || spec.Frames <= 0 || spec.Height <= 0 || spec.Width <= 0 || spec.Channels <= 0)
            throw new ArgumentException($"A shuffle grid needs positive batch, frames, height, width and channels; got {spec}.", nameof(spec));
        if (spec.FactorT <= 0 || spec.FactorH <= 0 || spec.FactorW <= 0)
            throw new ArgumentException($"Shuffle factors must be positive; got {spec}.", nameof(spec));
        if (spec.ChannelRepeat <= 0 || spec.PackedChannels % spec.ChannelRepeat != 0)
            throw new ArgumentException($"The channel repeat {spec.ChannelRepeat} must divide the {spec.PackedChannels} packed channels.", nameof(spec));
        if (spec.DropLeadingFrames < 0 || spec.ChunkStartFrame < 0)
            throw new ArgumentException($"Dropped frames and the chunk start must not be negative; got {spec}.", nameof(spec));
        long packedCount = (long)spec.Batch * spec.StoredChannels * spec.Frames * spec.Height * spec.Width;
        long unpackedCount = (long)spec.Batch * spec.Channels * spec.UnpackedFrameCount * spec.Height * spec.FactorH * spec.Width * spec.FactorW;
        if (packed.ElementCount != packedCount)
            throw new ArgumentException($"The packed tensor {packed.Shape} holds {packed.ElementCount} elements; the shuffle expects {packedCount}.", nameof(packed));
        if (unpacked.ElementCount != unpackedCount)
            throw new ArgumentException($"The unpacked tensor {unpacked.Shape} holds {unpacked.ElementCount} elements; the shuffle expects {unpackedCount}.", nameof(unpacked));
    }
}
