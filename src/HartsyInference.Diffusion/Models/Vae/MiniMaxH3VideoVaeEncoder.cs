using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>MiniMax-H3 video VAE encoder — the 3D causal CNN half (<c>EncoderFCN3D</c>), paired with the ViT3D
/// <see cref="MiniMaxH3VideoVaeDecoder"/>: <c>conv_in(3→128) → 6 stages (2 residual blocks each, strided downsample on
/// the first four) → per-frame GroupNorm+SiLU → conv_out(1024→48) → quant_conv</c>, then the posterior mean is taken
/// and normalized by the stored latent statistics. Supplies the keyframe and reference latents the DiT conditions on;
/// the weights ship in the same file as the decoder and were simply never read before.</summary>
public sealed unsafe class MiniMaxH3VideoVaeEncoder
{
    private readonly MiniMaxH3VideoVaeConfig _config;

    private CausalConv3d? _convIn;
    private Stage[] _stages = [];
    private Tensor? _normOutWeight, _normOutBias;
    private CausalConv3d? _convOut;
    private CausalConv3d? _quantConv;

    private sealed class ResBlock
    {
        public required Tensor Norm1Weight { get; init; }
        public required Tensor Norm1Bias { get; init; }
        public required Tensor Norm2Weight { get; init; }
        public required Tensor Norm2Bias { get; init; }
        public required CausalConv3d Conv1 { get; init; }
        public required CausalConv3d Conv2 { get; init; }
        public CausalConv3d? NinShortcut { get; init; }
    }

    private sealed class Stage
    {
        public required ResBlock[] Blocks { get; init; }
        public CausalConv3d? Downsample { get; init; }
        public int SpaceStride { get; init; }
    }

    public MiniMaxH3VideoVaeEncoder(MiniMaxH3VideoVaeConfig? config = null)
    {
        _config = config ?? new MiniMaxH3VideoVaeConfig();
    }

    public MiniMaxH3VideoVaeConfig Config => _config;

    /// <summary>Loads the <c>encoder.*</c> and <c>quant_conv.*</c> tensors, which sit alongside the decoder's in the
    /// same checkpoint. Spatial padding is mirror-reflect and temporal padding is causal-zero throughout, matching the
    /// reference's <c>CausalConv3d</c>; the downsample convolutions pad spatially themselves and so take none here.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (!MiniMaxH3VideoVaeConfig.MatchesEncoder(weights))
        {
            throw new ArgumentException(
                "Not a MiniMax-H3 video VAE encoder: 'encoder.conv_in.weight', 'encoder.conv_out.weight' and "
                + "'quant_conv.weight' are all required.");
        }
        _convIn = Conv(weights, "encoder.conv_in", padT: 1, padH: 1, padW: 1);

        int stageCount = _config.EncoderChannelMult.Length;
        _stages = new Stage[stageCount];
        for (int i = 0; i < stageCount; i++)
        {
            ResBlock[] blocks = new ResBlock[_config.EncoderResBlocks];
            for (int j = 0; j < blocks.Length; j++)
            {
                string p = $"encoder.down.{i}.block.{j}";
                blocks[j] = new ResBlock
                {
                    Norm1Weight = Require(weights, $"{p}.norm1.weight"),
                    Norm1Bias = Require(weights, $"{p}.norm1.bias"),
                    Norm2Weight = Require(weights, $"{p}.norm2.weight"),
                    Norm2Bias = Require(weights, $"{p}.norm2.bias"),
                    Conv1 = Conv(weights, $"{p}.conv1", padT: 1, padH: 1, padW: 1),
                    Conv2 = Conv(weights, $"{p}.conv2", padT: 1, padH: 1, padW: 1),
                    NinShortcut = weights.ContainsKey($"{p}.nin_shortcut.weight")
                        ? Conv(weights, $"{p}.nin_shortcut", padT: 0, padH: 0, padW: 0) : null,
                };
            }
            int spaceStride = _config.EncoderSpaceDown[i], timeStride = _config.EncoderTimeDown[i];
            _stages[i] = new Stage
            {
                Blocks = blocks,
                SpaceStride = spaceStride,
                // The stage downsamples only when a stride actually reduces something; the reference gates on the
                // product, so a 1x1 stage carries no downsample weights at all.
                Downsample = spaceStride * timeStride > 1
                    ? Conv(weights, $"encoder.down.{i}.downsample.conv", padT: 1, padH: 0, padW: 0,
                        strideT: timeStride, strideH: spaceStride, strideW: spaceStride)
                    : null,
            };
        }

        _normOutWeight = Require(weights, "encoder.norm_out.weight");
        _normOutBias = Require(weights, "encoder.norm_out.bias");
        _convOut = Conv(weights, "encoder.conv_out", padT: 1, padH: 1, padW: 1);
        _quantConv = Conv(weights, "quant_conv", padT: 0, padH: 0, padW: 0);
    }

    /// <summary>Every encoder tensor, for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convIn is not null)
        {
            foreach (Tensor t in _convIn.EnumerateWeights()) yield return t;
        }
        foreach (Stage s in _stages)
        {
            foreach (ResBlock b in s.Blocks)
            {
                yield return b.Norm1Weight;
                yield return b.Norm1Bias;
                yield return b.Norm2Weight;
                yield return b.Norm2Bias;
                foreach (Tensor t in b.Conv1.EnumerateWeights()) yield return t;
                foreach (Tensor t in b.Conv2.EnumerateWeights()) yield return t;
                if (b.NinShortcut is not null)
                {
                    foreach (Tensor t in b.NinShortcut.EnumerateWeights()) yield return t;
                }
            }
            if (s.Downsample is not null)
            {
                foreach (Tensor t in s.Downsample.EnumerateWeights()) yield return t;
            }
        }
        if (_normOutWeight is not null) yield return _normOutWeight;
        if (_normOutBias is not null) yield return _normOutBias;
        if (_convOut is not null)
        {
            foreach (Tensor t in _convOut.EnumerateWeights()) yield return t;
        }
        if (_quantConv is not null)
        {
            foreach (Tensor t in _quantConv.EnumerateWeights()) yield return t;
        }
    }

    /// <summary>Encodes RGB <c>[1, 3, T, H, W]</c> in [-1, 1] to normalized latents <c>[1, 24, T', H/16, W/16]</c> —
    /// the same space the DiT denoises, so the result drops straight into a keyframe or reference block.</summary>
    public Tensor Encode(IBackend backend, Tensor rgb)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(rgb);
        if (rgb.Shape.Rank != 5 || rgb.Shape[0] != 1 || rgb.Shape[1] != _config.OutChannels)
        {
            throw new ArgumentException($"Expected RGB [1, {_config.OutChannels}, T, H, W]; got {rgb.Shape}.", nameof(rgb));
        }
        int frames = (int)rgb.Shape[2];

        Tensor normalized = NormalizePixels(rgb);
        Tensor moments;
        try
        {
            // A single frame still runs the full stack — the temporal strides collapse it to one token — but the
            // chunked path's clip padding would invent frames that are not there.
            moments = frames == 1 ? LastToken(AdaptiveEncode(backend, normalized)) : EncodeTemporal(backend, normalized);
        }
        finally
        {
            normalized.Dispose();
        }
        try
        {
            return NormalizeLatents(moments);
        }
        finally
        {
            moments.Dispose();
        }
    }

    /// <summary>Encodes one interleaved-RGB24 frame to <c>[1, 24, 1, H/16, W/16]</c> — the fl2va keyframe entry point.
    /// The caller owns the result.</summary>
    public Tensor EncodeRgbFrame(IBackend backend, ReadOnlySpan<byte> rgb24, int width, int height)
    {
        if (rgb24.Length < (long)width * height * 3)
        {
            throw new ArgumentException($"expected {(long)width * height * 3} bytes; got {rgb24.Length}.", nameof(rgb24));
        }
        using Tensor rgb = new Tensor(new TensorShape([1L, 3, 1, height, width]), DType.F32);
        float* p = (float*)rgb.DataPointer;
        long plane = (long)height * width;
        for (long pix = 0; pix < plane; pix++)
        {
            for (int c = 0; c < 3; c++)
            {
                p[c * plane + pix] = rgb24[(int)(pix * 3 + c)] / 127.5f - 1f;
            }
        }
        return Encode(backend, rgb);
    }

    /// <summary>Encodes a clip frame by frame from interleaved RGB24 — the ref2va reference-video entry point.</summary>
    public Tensor EncodeRgbClip(IBackend backend, IReadOnlyList<byte[]> frames, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (frames.Count == 0)
        {
            throw new ArgumentException("A reference clip needs at least one frame.", nameof(frames));
        }
        using Tensor rgb = new Tensor(new TensorShape([1L, 3, frames.Count, height, width]), DType.F32);
        float* p = (float*)rgb.DataPointer;
        long plane = (long)height * width;
        for (int f = 0; f < frames.Count; f++)
        {
            ReadOnlySpan<byte> src = frames[f];
            if (src.Length < plane * 3)
            {
                throw new ArgumentException($"frame {f} has {src.Length} bytes, expected {plane * 3}.", nameof(frames));
            }
            for (long pix = 0; pix < plane; pix++)
            {
                for (int c = 0; c < 3; c++)
                {
                    p[(c * frames.Count + f) * plane + pix] = src[(int)(pix * 3 + c)] / 127.5f - 1f;
                }
            }
        }
        return Encode(backend, rgb);
    }

    /// <summary>Splits the clip on the encoder's own chunk length, encoding each independently, then drops the trailing
    /// tokens the reference discards. A short tail repeats the last frame rather than zero-filling.</summary>
    private Tensor EncodeTemporal(IBackend backend, Tensor x)
    {
        int frames = (int)x.Shape[2], clip = _config.ClipLength;
        Tensor padded = frames % clip == 0 ? x : RepeatLastFrame(x, clip - frames % clip);
        try
        {
            int chunks = (int)padded.Shape[2] / clip;
            List<Tensor> encoded = new List<Tensor>(chunks);
            try
            {
                for (int i = 0; i < chunks; i++)
                {
                    using Tensor slice = SliceFrames(padded, i * clip, clip);
                    encoded.Add(AdaptiveEncode(backend, slice));
                }
                using Tensor joined = ConcatFrames(encoded);
                int keep = (int)joined.Shape[2] - _config.TokenDrop;
                if (keep <= 0)
                {
                    throw new ArgumentException(
                        $"MiniMax-H3 reference clip is too short: {frames} frame(s) yield {joined.Shape[2]} token(s), "
                        + $"all of which the encoder drops.", nameof(x));
                }
                return SliceFrames(joined, 0, keep);
            }
            finally
            {
                foreach (Tensor t in encoded)
                {
                    t.Dispose();
                }
            }
        }
        finally
        {
            if (!ReferenceEquals(padded, x))
            {
                padded.Dispose();
            }
        }
    }

    private Tensor AdaptiveEncode(IBackend backend, Tensor x) =>
        _config.DecoderTiling ? TiledEncode(backend, x) : EncodeMoments(backend, x);

    /// <summary>Encodes tile by tile, blending each against the RAW (unblended) latents of its top and left neighbours
    /// then cropping the overlap away — the encode-side mirror of the decoder's tiling, sharing its tile grid.</summary>
    private Tensor TiledEncode(IBackend backend, Tensor x)
    {
        int height = (int)x.Shape[3], width = (int)x.Shape[4], ratio = _config.VaeRatio;
        (int[] ys, int[] yLen, int[] yOverlap) = _config.SplitTiles(height);
        (int[] xs, int[] xLen, int[] xOverlap) = _config.SplitTiles(width);
        if (ys.Length == 1 && xs.Length == 1)
        {
            return EncodeMoments(backend, x);
        }

        Tensor[][] rows = new Tensor[ys.Length][];
        try
        {
            for (int i = 0; i < ys.Length; i++)
            {
                rows[i] = new Tensor[xs.Length];
                for (int j = 0; j < xs.Length; j++)
                {
                    using Tensor tile = SpatialTile(x, ys[i], xs[j], yLen[i], xLen[j]);
                    rows[i][j] = EncodeMoments(backend, tile);
                }
            }

            int channels = (int)rows[0][0].Shape[1], tokens = (int)rows[0][0].Shape[2];
            List<Tensor> rowCanvases = new List<Tensor>(ys.Length);
            try
            {
                for (int i = 0; i < ys.Length; i++)
                {
                    List<Tensor> pieces = new List<Tensor>(xs.Length);
                    try
                    {
                        for (int j = 0; j < xs.Length; j++)
                        {
                            Tensor blended = VaeOps.Clone(rows[i][j]);
                            int planes = channels * tokens;
                            // Blend against the neighbours' RAW latents, matching the reference's bookkeeping — using
                            // an already-blended neighbour would compound the ramp across a row.
                            if (i > 0)
                            {
                                int overlap = yOverlap[i - 1] / ratio;
                                using Tensor tail = LatentTail(rows[i - 1][j], planes, overlap, vertical: true);
                                using Tensor view = BlendView(blended, planes);
                                VaeTiling.BlendVertical(tail, view, overlap);
                            }
                            if (j > 0)
                            {
                                int overlap = xOverlap[j - 1] / ratio;
                                using Tensor tail = LatentTail(rows[i][j - 1], planes, overlap, vertical: false);
                                using Tensor view = BlendView(blended, planes);
                                VaeTiling.BlendHorizontal(tail, view, overlap);
                            }
                            int cropH = i < ys.Length - 1 ? (int)blended.Shape[3] - yOverlap[i] / ratio : (int)blended.Shape[3];
                            int cropW = j < xs.Length - 1 ? (int)blended.Shape[4] - xOverlap[j] / ratio : (int)blended.Shape[4];
                            pieces.Add(CropLatent(blended, cropH, cropW));
                            blended.Dispose();
                        }
                        rowCanvases.Add(ConcatWidth(pieces));
                    }
                    finally
                    {
                        foreach (Tensor t in pieces)
                        {
                            t.Dispose();
                        }
                    }
                }
                return ConcatHeight(rowCanvases);
            }
            finally
            {
                foreach (Tensor t in rowCanvases)
                {
                    t.Dispose();
                }
            }
        }
        finally
        {
            foreach (Tensor[]? row in rows)
            {
                if (row is null)
                {
                    continue;
                }
                foreach (Tensor t in row)
                {
                    t?.Dispose();
                }
            }
        }
    }

    private Tensor EncodeMoments(IBackend backend, Tensor x)
    {
        Tensor cur = _convIn!.Forward(backend, x);
        try
        {
            foreach (Stage stage in _stages)
            {
                foreach (ResBlock block in stage.Blocks)
                {
                    Tensor next = ResidualForward(backend, cur, block);
                    cur.Dispose();
                    cur = next;
                }
                if (stage.Downsample is not null)
                {
                    Tensor padded = stage.SpaceStride == 2 ? PadRightBottomReflect(cur) : cur;
                    Tensor down = stage.Downsample.Forward(backend, padded);
                    if (!ReferenceEquals(padded, cur))
                    {
                        padded.Dispose();
                    }
                    cur.Dispose();
                    cur = down;
                }
            }
            Tensor normed = GroupNormPerFrameSilu(backend, cur, _normOutWeight!, _normOutBias!);
            cur.Dispose();
            Tensor moments = _convOut!.Forward(backend, normed);
            normed.Dispose();
            cur = _quantConv!.Forward(backend, moments);
            moments.Dispose();
            return cur;
        }
        catch
        {
            cur.Dispose();
            throw;
        }
    }

    private Tensor ResidualForward(IBackend backend, Tensor x, ResBlock block)
    {
        Tensor h = GroupNormPerFrameSilu(backend, x, block.Norm1Weight, block.Norm1Bias);
        Tensor c1;
        try
        {
            c1 = block.Conv1.Forward(backend, h);
        }
        finally
        {
            h.Dispose();
        }
        Tensor h2 = GroupNormPerFrameSilu(backend, c1, block.Norm2Weight, block.Norm2Bias);
        c1.Dispose();
        Tensor c2;
        try
        {
            c2 = block.Conv2.Forward(backend, h2);
        }
        finally
        {
            h2.Dispose();
        }
        Tensor shortcut = block.NinShortcut is null ? x : block.NinShortcut.Forward(backend, x);
        try
        {
            Tensor sum = new Tensor(c2.Shape, DType.F32);
            backend.Add(sum, c2, shortcut);
            return sum;
        }
        finally
        {
            c2.Dispose();
            if (!ReferenceEquals(shortcut, x))
            {
                shortcut.Dispose();
            }
        }
    }

    /// <summary>GroupNorm whose statistics are taken per frame rather than per clip, then SiLU. Moving T into the batch
    /// slot leaves the stock op reducing over <c>(C/g, H, W)</c> only; norm and activation share one tensor identity so
    /// the CUDA activation cache cannot serve a stale parent.</summary>
    private Tensor GroupNormPerFrameSilu(IBackend backend, Tensor x, Tensor weight, Tensor bias)
    {
        int c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        int plane = checked(h * w);
        Tensor framed = new Tensor(new TensorShape(t, c, h, w), DType.F32);
        backend.Permute0213(framed, x, c, t, plane);
        Tensor normed = new Tensor(framed.Shape, DType.F32);
        try
        {
            backend.GroupNorm(normed, framed, weight, bias, _config.EncoderNormGroups, _config.EncoderNormEps);
            backend.Silu(normed, normed);
            Tensor outT = new Tensor(x.Shape, DType.F32);
            backend.Permute0213(outT, normed, t, c, plane);
            return outT;
        }
        finally
        {
            framed.Dispose();
            normed.Dispose();
        }
    }

    /// <summary>Mirror-pads one column on the right and one row on the bottom, which the reference does before every
    /// spatially strided downsample so the halved axis covers the full input.</summary>
    private static Tensor PadRightBottomReflect(Tensor x)
    {
        int c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        Tensor outT = new Tensor(new TensorShape([1L, c, t, h + 1, w + 1]), DType.F32);
        float* src = (float*)x.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (long ct = 0; ct < (long)c * t; ct++)
        {
            float* sp = src + ct * h * w;
            float* dp = dst + ct * (h + 1) * (w + 1);
            for (int y = 0; y <= h; y++)
            {
                int sy = y == h ? h - 2 : y;
                for (int xi = 0; xi <= w; xi++)
                {
                    int sx = xi == w ? w - 2 : xi;
                    dp[(long)y * (w + 1) + xi] = sp[(long)sy * w + sx];
                }
            }
        }
        return outT;
    }

    /// <summary>Applies the encoder's pixel normalization: [-1, 1] to [0, 1], then ImageNet mean/std.</summary>
    private Tensor NormalizePixels(Tensor rgb)
    {
        int c = (int)rgb.Shape[1], t = (int)rgb.Shape[2], h = (int)rgb.Shape[3], w = (int)rgb.Shape[4];
        Tensor outT = new Tensor(rgb.Shape, DType.F32);
        float* src = (float*)rgb.DataPointer;
        float* dst = (float*)outT.DataPointer;
        long plane = (long)t * h * w;
        for (int ci = 0; ci < c; ci++)
        {
            float mean = _config.PixelMean[ci], std = _config.PixelStd[ci];
            for (long i = 0; i < plane; i++)
            {
                dst[ci * plane + i] = ((src[ci * plane + i] + 1f) * 0.5f - mean) / std;
            }
        }
        return outT;
    }

    /// <summary>Takes the posterior mean (the leading half of the moments) and normalizes it by the stored per-channel
    /// latent statistics, landing in the same space the DiT's own latents live in.</summary>
    private Tensor NormalizeLatents(Tensor moments)
    {
        int t = (int)moments.Shape[2], h = (int)moments.Shape[3], w = (int)moments.Shape[4];
        int channels = _config.LatentChannels;
        Tensor outT = new Tensor(new TensorShape([1L, channels, t, h, w]), DType.F32);
        float* src = (float*)moments.DataPointer;
        float* dst = (float*)outT.DataPointer;
        long plane = (long)t * h * w;
        for (int ci = 0; ci < channels; ci++)
        {
            float mean = _config.LatentsMean[ci], std = _config.LatentsStd[ci];
            for (long i = 0; i < plane; i++)
            {
                dst[ci * plane + i] = (src[ci * plane + i] - mean) / std;
            }
        }
        return outT;
    }

    private static Tensor LastToken(Tensor moments)
    {
        try
        {
            return SliceFrames(moments, (int)moments.Shape[2] - 1, 1);
        }
        finally
        {
            moments.Dispose();
        }
    }

    private static Tensor SliceFrames(Tensor x, int start, int count)
    {
        int c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        Tensor outT = new Tensor(new TensorShape([1L, c, count, h, w]), DType.F32);
        float* src = (float*)x.DataPointer;
        float* dst = (float*)outT.DataPointer;
        long plane = (long)h * w;
        for (int ci = 0; ci < c; ci++)
        {
            Buffer.MemoryCopy(src + ((long)ci * t + start) * plane, dst + (long)ci * count * plane,
                count * plane * 4, count * plane * 4);
        }
        return outT;
    }

    private static Tensor RepeatLastFrame(Tensor x, int extra)
    {
        int c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        Tensor outT = new Tensor(new TensorShape([1L, c, t + extra, h, w]), DType.F32);
        float* src = (float*)x.DataPointer;
        float* dst = (float*)outT.DataPointer;
        long plane = (long)h * w;
        for (int ci = 0; ci < c; ci++)
        {
            Buffer.MemoryCopy(src + (long)ci * t * plane, dst + (long)ci * (t + extra) * plane,
                t * plane * 4, t * plane * 4);
            for (int e = 0; e < extra; e++)
            {
                Buffer.MemoryCopy(src + ((long)ci * t + t - 1) * plane, dst + ((long)ci * (t + extra) + t + e) * plane,
                    plane * 4, plane * 4);
            }
        }
        return outT;
    }

    private static Tensor ConcatFrames(IReadOnlyList<Tensor> parts)
    {
        int c = (int)parts[0].Shape[1], h = (int)parts[0].Shape[3], w = (int)parts[0].Shape[4];
        int total = 0;
        foreach (Tensor p in parts)
        {
            total += (int)p.Shape[2];
        }
        Tensor outT = new Tensor(new TensorShape([1L, c, total, h, w]), DType.F32);
        float* dst = (float*)outT.DataPointer;
        long plane = (long)h * w;
        int cursor = 0;
        foreach (Tensor p in parts)
        {
            int pt = (int)p.Shape[2];
            float* src = (float*)p.DataPointer;
            for (int ci = 0; ci < c; ci++)
            {
                Buffer.MemoryCopy(src + (long)ci * pt * plane, dst + ((long)ci * total + cursor) * plane,
                    pt * plane * 4, pt * plane * 4);
            }
            cursor += pt;
        }
        return outT;
    }

    private static Tensor SpatialTile(Tensor x, int y, int xStart, int tileH, int tileW)
    {
        int c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        Tensor outT = new Tensor(new TensorShape([1L, c, t, tileH, tileW]), DType.F32);
        float* src = (float*)x.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (long ct = 0; ct < (long)c * t; ct++)
        {
            for (int row = 0; row < tileH; row++)
            {
                Buffer.MemoryCopy(src + ct * h * w + (long)(y + row) * w + xStart,
                    dst + ct * tileH * tileW + (long)row * tileW, tileW * 4, tileW * 4);
            }
        }
        return outT;
    }

    /// <summary>A <c>[1, planes, H, W]</c> view of a latent tile for the blend helpers, which are pure host loops —
    /// safe to write through a view precisely because no device op binds its result to the object it was handed.</summary>
    private static Tensor BlendView(Tensor tile, int planes) =>
        new Tensor((void*)tile.DataPointer, new TensorShape(1, planes, tile.Shape[3], tile.Shape[4]), DType.F32);

    /// <summary>The overlap strip of a neighbouring tile, reshaped to the <c>[1, planes, H, W]</c> view the blend
    /// helpers expect.</summary>
    private static Tensor LatentTail(Tensor tile, int planes, int overlap, bool vertical)
    {
        int h = (int)tile.Shape[3], w = (int)tile.Shape[4];
        using Tensor view = new Tensor((void*)tile.DataPointer, new TensorShape(1, planes, h, w), DType.F32);
        return vertical ? VaeTiling.ExtractTile(view, 1, planes, h - overlap, 0, overlap, w)
            : VaeTiling.ExtractTile(view, 1, planes, 0, w - overlap, h, overlap);
    }

    private static Tensor CropLatent(Tensor x, int cropH, int cropW)
    {
        int c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        if (cropH == h && cropW == w)
        {
            return VaeOps.Clone(x);
        }
        Tensor outT = new Tensor(new TensorShape([1L, c, t, cropH, cropW]), DType.F32);
        float* src = (float*)x.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (long ct = 0; ct < (long)c * t; ct++)
        {
            for (int row = 0; row < cropH; row++)
            {
                Buffer.MemoryCopy(src + ct * h * w + (long)row * w, dst + ct * cropH * cropW + (long)row * cropW,
                    cropW * 4, cropW * 4);
            }
        }
        return outT;
    }

    private static Tensor ConcatWidth(IReadOnlyList<Tensor> parts)
    {
        int c = (int)parts[0].Shape[1], t = (int)parts[0].Shape[2], h = (int)parts[0].Shape[3];
        int total = 0;
        foreach (Tensor p in parts)
        {
            total += (int)p.Shape[4];
        }
        Tensor outT = new Tensor(new TensorShape([1L, c, t, h, total]), DType.F32);
        float* dst = (float*)outT.DataPointer;
        int cursor = 0;
        foreach (Tensor p in parts)
        {
            int pw = (int)p.Shape[4];
            float* src = (float*)p.DataPointer;
            for (long ct = 0; ct < (long)c * t; ct++)
            {
                for (int row = 0; row < h; row++)
                {
                    Buffer.MemoryCopy(src + ct * h * pw + (long)row * pw,
                        dst + ct * h * total + (long)row * total + cursor, pw * 4, pw * 4);
                }
            }
            cursor += pw;
        }
        return outT;
    }

    private static Tensor ConcatHeight(IReadOnlyList<Tensor> parts)
    {
        int c = (int)parts[0].Shape[1], t = (int)parts[0].Shape[2], w = (int)parts[0].Shape[4];
        int total = 0;
        foreach (Tensor p in parts)
        {
            total += (int)p.Shape[3];
        }
        Tensor outT = new Tensor(new TensorShape([1L, c, t, total, w]), DType.F32);
        float* dst = (float*)outT.DataPointer;
        int cursor = 0;
        foreach (Tensor p in parts)
        {
            int ph = (int)p.Shape[3];
            float* src = (float*)p.DataPointer;
            for (long ct = 0; ct < (long)c * t; ct++)
            {
                Buffer.MemoryCopy(src + ct * ph * w, dst + ct * total * w + (long)cursor * w,
                    (long)ph * w * 4, (long)ph * w * 4);
            }
            cursor += ph;
        }
        return outT;
    }

    private static CausalConv3d Conv(IReadOnlyDictionary<string, Tensor> weights, string prefix,
        int padT, int padH, int padW, int strideT = 1, int strideH = 1, int strideW = 1) =>
        new CausalConv3d(Require(weights, $"{prefix}.weight"),
            weights.TryGetValue($"{prefix}.bias", out Tensor? bias) ? bias : null,
            strideT, strideH, strideW, padT, padH, padW,
            replicateFirstPad: false, causal: true, spatialReflectPad: padH > 0 || padW > 0);

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string key) =>
        weights.TryGetValue(key, out Tensor? t) ? t
            : throw new ArgumentException($"MiniMax-H3 video VAE encoder is missing '{key}'.");
}
