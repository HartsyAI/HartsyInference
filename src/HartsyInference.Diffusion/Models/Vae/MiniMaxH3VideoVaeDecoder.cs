using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>MiniMax-H3 video VAE decoder — a <b>ViT3D transformer</b>, not a conv upsampler. Per spatial tile:
/// <c>post_quant_conv</c> (1×1×1, run as a linear over channel vectors) → <c>x_embedder</c> → the latent grid as one
/// token per latent cell, plus 4 register tokens and a zero cls row → 36 blocks
/// (<see cref="MiniMaxH3VideoVaeVitBlock"/>) under a 3-axis length-normalized rope → affine <c>norm_out</c> LayerNorm
/// → <c>proj_out</c> → unpatchify to a 4×16×16 pixel patch per token. Latents are denormalized
/// (<c>latent · std + mean</c>) before any of that, and the ImageNet pixel denormalization is folded into a private
/// copy of <c>proj_out</c> so the tail stays whole-tensor ops.</summary>
public sealed unsafe class MiniMaxH3VideoVaeDecoder : IDisposable
{
    private readonly MiniMaxH3VideoVaeConfig _config;
    private readonly List<Tensor> _owned = new List<Tensor>();
    private MiniMaxH3VideoVaeVitBlock[] _blocks = [];
    private Tensor? _postQuantW, _postQuantB, _xEmbedW, _xEmbedB;
    private Tensor? _registerTokens, _clsToken, _qkNormOnes;
    private Tensor? _normOutW, _normOutB, _projOutW, _projOutB;
    private Tensor? _latentsMean, _latentsStd;
    private float[] _invFreq = [];
    private bool _disposed;

    public MiniMaxH3VideoVaeDecoder(MiniMaxH3VideoVaeConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.RopeDim % 6 != 0)
            throw new ArgumentException($"Rope width {config.RopeDim} must be divisible by 2×3 axes.", nameof(config));
        if (config.RopeDim > config.DimHead)
            throw new ArgumentException($"Rope width {config.RopeDim} exceeds head dim {config.DimHead}.", nameof(config));
        if (config.LatentsMean.Length != config.LatentChannels || config.LatentsStd.Length != config.LatentChannels)
        {
            throw new ArgumentException(
                $"latents_mean/latents_std must have {config.LatentChannels} entries; got "
                + $"{config.LatentsMean.Length}/{config.LatentsStd.Length}.", nameof(config));
        }
        if (config.PixelMean.Length != config.OutChannels || config.PixelStd.Length != config.OutChannels)
            throw new ArgumentException($"pixel mean/std must have {config.OutChannels} entries.", nameof(config));
        _config = config;
    }

    public MiniMaxH3VideoVaeConfig Config => _config;

    /// <summary>Maps checkpoint tensors (native <c>decoder.*</c> / <c>post_quant_conv.*</c> names) onto the decoder.
    /// Weights are borrowed from <paramref name="w"/>; only the F32 casts and the folded/reshaped copies are owned.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        ArgumentNullException.ThrowIfNull(w);
        int dim = _config.Dim;

        _postQuantW = Reshape2d(MiniMaxH3VideoVaeWeights.Raw(w, "post_quant_conv.weight"), _config.ZChannels, _config.LatentChannels);
        _postQuantB = MiniMaxH3VideoVaeWeights.OptionalF32(w, "post_quant_conv.bias", _owned);
        _xEmbedW = MiniMaxH3VideoVaeWeights.Raw(w, "decoder.x_embedder.weight");
        _xEmbedB = MiniMaxH3VideoVaeWeights.OptionalF32(w, "decoder.x_embedder.bias", _owned);
        _normOutW = MiniMaxH3VideoVaeWeights.F32(w, "decoder.norm_out.weight", _owned);
        _normOutB = MiniMaxH3VideoVaeWeights.F32(w, "decoder.norm_out.bias", _owned);
        _registerTokens = Reshape2d(MiniMaxH3VideoVaeWeights.Raw(w, "decoder.register_tokens"), _config.NumRegisterTokens, dim);

        _clsToken = new Tensor(new TensorShape(1, dim), DType.F32);
        _clsToken.AsSpan<float>().Clear();
        _owned.Add(_clsToken);
        _qkNormOnes = new Tensor(new TensorShape(_config.DimHead), DType.F32);
        _qkNormOnes.AsSpan<float>().Fill(1f);
        _owned.Add(_qkNormOnes);
        _latentsMean = FromArray(_config.LatentsMean);
        _latentsStd = FromArray(_config.LatentsStd);

        FoldProjOut(w);

        _blocks = new MiniMaxH3VideoVaeVitBlock[_config.NumLayers];
        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i] = new MiniMaxH3VideoVaeVitBlock(_config);
            _blocks[i].LoadWeights(w, $"decoder.transformer_blocks.{i}", _owned);
        }
        _invFreq = MiniMaxH3Rope.DefaultInvFreq(_config.RopeInvFreqLen, _config.RopeTheta);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _postQuantW, _postQuantB, _xEmbedW, _xEmbedB, _registerTokens, _clsToken,
            _qkNormOnes, _normOutW, _normOutB, _projOutW, _projOutB, _latentsMean, _latentsStd })
        {
            if (t is not null) yield return t;
        }
        foreach (MiniMaxH3VideoVaeVitBlock b in _blocks)
        {
            foreach (Tensor t in b.EnumerateWeights()) yield return t;
        }
    }

    /// <summary>Decodes a normalized latent <c>[1, 24, T, H, W]</c> to RGB <c>[1, 3, F, H·16, W·16]</c> in [-1, 1],
    /// denormalizing with <c>latent · std + mean</c> first. Spatial tiling and temporal chunking both run internally,
    /// so 2K clips never materialize a whole-frame token sequence.</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(latent);
        if (latent.Shape.Rank != 5)
            throw new ArgumentException($"expected a [B, C, T, H, W] latent; got {latent.Shape}.", nameof(latent));
        if (latent.Shape[0] != 1)
            throw new ArgumentException($"batch must be 1; got {latent.Shape[0]}.", nameof(latent));
        if (latent.Shape[1] != _config.LatentChannels)
            throw new ArgumentException($"latent channels {latent.Shape[1]} != {_config.LatentChannels}.", nameof(latent));

        int zt = (int)latent.Shape[2], zh = (int)latent.Shape[3], zw = (int)latent.Shape[4];
        using Tensor z = Denormalize(backend, latent, zt, zh, zw);
        if (zt > 1)
        {
            return DecodeTemporal(backend, z, zt, zh, zw);
        }

        int height = zh * _config.VaeRatio, width = zw * _config.VaeRatio;
        using Tensor single = AdaptiveDecode(backend, z, 1, zh, zw);
        return SliceFrames(single, _config.VaeRatioT, height, width, _config.VaeRatioT - 1, 1, asRank5: true);
    }

    /// <summary>Frames <see cref="Decode"/> emits for a latent of <paramref name="latentFrames"/> tokens.</summary>
    public int OutputFrames(int latentFrames)
    {
        if (latentFrames <= 1)
        {
            return 1;
        }
        (int numChunks, int padTokens) = PlanChunks(latentFrames);
        return FramePlan(latentFrames + padTokens, numChunks, padTokens);
    }

    // ── Latent normalization ────────────────────────────────────────────

    /// <summary>Undoes the encoder's per-channel latent normalization. Transposing to <c>[n, C]</c> puts the channel
    /// on the last dim, which is what the broadcast affine op operates over.</summary>
    private Tensor Denormalize(IBackend backend, Tensor latent, int zt, int zh, int zw)
    {
        int c = _config.LatentChannels;
        int n = zt * zh * zw;
        using Tensor rows = new Tensor(new TensorShape(1, n, c), DType.F32);
        backend.Transpose2D(rows, latent, c, n);
        backend.AffineBroadcastLastDim(rows, rows, _latentsStd!, _latentsMean);
        Tensor outT = new Tensor(new TensorShape([1L, c, zt, zh, zw]), DType.F32);
        backend.Transpose2D(outT, rows, n, c);
        return outT;
    }

    // ── Temporal chunking ───────────────────────────────────────────────

    /// <summary>Decodes a multi-token latent chunk by chunk. With the shipped config (clip 17, drop 3, ratio_t 4):
    /// <c>tokens_chunk_size</c> 5, <c>token_overlap</c> 2, <c>frame_pre_padding</c> 3, <c>frame_overlap</c> 5 — each
    /// chunk reads 7 tokens, decodes 28 frames, emits frames 3..20 and carries frames 23..28 forward as the overlap
    /// that the next chunk's leading 5 frames are ramp-blended against.</summary>
    private Tensor DecodeTemporal(IBackend backend, Tensor z, int zt, int zh, int zw)
    {
        int ratioT = _config.VaeRatioT, chunkTokens = _config.TokensChunkSize;
        int chunkDec = chunkTokens * ratioT;
        int splitCount = _config.TokenDrop > 0 ? 2 : 1;
        int height = zh * _config.VaeRatio, width = zw * _config.VaeRatio;

        (int numChunks, int padTokens) = PlanChunks(zt);
        using Tensor padded = padTokens > 0 ? PadTokens(z, zt, zh, zw, padTokens) : Borrow(z);
        int zLen = zt + padTokens;
        int outputFrames = FramePlan(zLen, numChunks, padTokens);

        Tensor? dec = null;
        Tensor? overlap = null;
        int writePos = 0;
        try
        {
            for (int i = 0; i < numChunks; i++)
            {
                int start = Math.Min(i * chunkTokens, zLen);
                int stop = Math.Min(start + chunkTokens + _config.TokenOverlap, zLen);
                using Tensor clipZ = SliceTokens(padded, zh, zw, start, stop - start);
                using Tensor clipDec = AdaptiveDecode(backend, clipZ, stop - start, zh, zw);
                int clipFrames = (stop - start) * ratioT;

                for (int j = 0; j < splitCount; j++)
                {
                    int from = j * chunkDec + _config.FramePrePadding;
                    int to = Math.Min(j * chunkDec + chunkDec, clipFrames);
                    if (to <= from)
                    {
                        continue;
                    }
                    Tensor part = SliceFrames(clipDec, clipFrames, height, width, from, to - from, asRank5: false);
                    if (j == 0)
                    {
                        if (overlap is not null)
                        {
                            BlendFrames(overlap, part, to - from, height * width);
                            overlap.Dispose();
                            overlap = null;
                        }
                        WritePart(ref dec, part, to - from, height, width, outputFrames, ref writePos);
                        part.Dispose();
                    }
                    else
                    {
                        overlap?.Dispose();
                        overlap = part;
                    }
                }

                if (i == numChunks - 1 && overlap is not null)
                {
                    WritePart(ref dec, overlap, (int)(overlap.ElementCount / (_config.OutChannels * (long)height * width)),
                        height, width, outputFrames, ref writePos);
                    overlap.Dispose();
                    overlap = null;
                }
            }
        }
        catch
        {
            dec?.Dispose();
            throw;
        }
        finally
        {
            overlap?.Dispose();
        }

        return dec ?? throw new InvalidOperationException("MiniMax-H3 video VAE temporal decode produced no frames.");
    }

    /// <summary>Chunk count and the tokens the last chunk must be padded by; the second branch covers latents too
    /// short for a single chunk, which the reference bundle would otherwise decode to nothing.</summary>
    private (int NumChunks, int PadTokens) PlanChunks(int zt)
    {
        int chunkTokens = _config.TokensChunkSize;
        int pseudo = zt + _config.TokenDrop;
        int padTokens = 0;
        int remainder = pseudo % chunkTokens;
        if (remainder != 0)
        {
            padTokens = chunkTokens - remainder;
            pseudo += padTokens;
        }
        int numChunks = pseudo / chunkTokens - (_config.TokenDrop > 0 ? 1 : 0);
        if (numChunks < 1)
        {
            padTokens += chunkTokens;
            numChunks++;
        }
        return (numChunks, padTokens);
    }

    private int FramePlan(int zLen, int numChunks, int padTokens)
    {
        int chunkTokens = _config.TokensChunkSize, ratioT = _config.VaeRatioT;
        int chunkDec = chunkTokens * ratioT;
        int splitCount = _config.TokenDrop > 0 ? 2 : 1;
        int total = 0, finalOverlap = 0;
        for (int i = 0; i < numChunks; i++)
        {
            int start = Math.Min(i * chunkTokens, zLen);
            int stop = Math.Min(start + chunkTokens + _config.TokenOverlap, zLen);
            int clipFrames = Math.Max(0, stop - start) * ratioT;
            for (int j = 0; j < splitCount; j++)
            {
                int fEnd = Math.Min(j * chunkDec + chunkDec, clipFrames);
                int frames = Math.Max(0, fEnd - j * chunkDec - _config.FramePrePadding);
                if (j == 0) total += frames;
                else finalOverlap = frames;
            }
        }
        return total + finalOverlap - PadFrames(zLen, padTokens);
    }

    private int PadFrames(int zLen, int padTokens)
    {
        if (padTokens <= 0)
        {
            return 0;
        }
        int intraTail = _config.ClipLength % _config.VaeRatioT;
        if (intraTail == 0)
        {
            return padTokens * _config.VaeRatioT;
        }
        int before = zLen - padTokens;
        int frames = 0;
        for (int k = 0; k < padTokens; k++)
        {
            frames += (before + k) % _config.TokensChunkSize == 0 ? intraTail : _config.VaeRatioT;
        }
        return frames;
    }

    // ── Spatial tiling ──────────────────────────────────────────────────

    private Tensor AdaptiveDecode(IBackend backend, Tensor z, int zt, int zh, int zw) =>
        _config.DecoderTiling ? TiledDecode(backend, z, zt, zh, zw) : DecodePixels(backend, z, zt, zh, zw);

    /// <summary>Decodes tile by tile into one canvas, ramp-blending each tile against the RAW (unblended) overlap of
    /// its top and left neighbours, then cropping the overlap away — matching the reference's tile bookkeeping.</summary>
    private Tensor TiledDecode(IBackend backend, Tensor z, int zt, int zh, int zw)
    {
        int ratio = _config.VaeRatio;
        int height = zh * ratio, width = zw * ratio;
        (int[] ys, int[] yLen, int[] yOverlap) = SplitTiles(height);
        (int[] xs, int[] xLen, int[] xOverlap) = SplitTiles(width);
        if (ys.Length == 1 && xs.Length == 1)
        {
            return DecodePixels(backend, z, zt, zh, zw);
        }

        int planes = _config.OutChannels * zt * _config.VaeRatioT;
        int latentPlanes = _config.LatentChannels * zt;
        Tensor canvas = new Tensor(new TensorShape(1, planes, height, width), DType.F32);
        using Tensor zView = new Tensor((void*)z.DataPointer, new TensorShape(1, latentPlanes, zh, zw), DType.F32);
        Tensor?[] rowTails = new Tensor?[xs.Length];
        int outY = 0;

        try
        {
            for (int i = 0; i < ys.Length; i++)
            {
                Tensor?[] newTails = new Tensor?[xs.Length];
                Tensor? leftTail = null;
                int outX = 0, cropH = 0;
                for (int j = 0; j < xs.Length; j++)
                {
                    int tileH = yLen[i], tileW = xLen[j];
                    using Tensor zTile = VaeTiling.ExtractTile(zView, 1, latentPlanes,
                        ys[i] / ratio, xs[j] / ratio, tileH / ratio, tileW / ratio);
                    using Tensor tile = DecodePixels(backend, zTile, zt, tileH / ratio, tileW / ratio);

                    if (i < ys.Length - 1)
                    {
                        newTails[j] = VaeTiling.ExtractTile(tile, 1, planes, tileH - yOverlap[i], 0, yOverlap[i], tileW);
                    }
                    Tensor? nextLeft = j < xs.Length - 1
                        ? VaeTiling.ExtractTile(tile, 1, planes, 0, tileW - xOverlap[j], tileH, xOverlap[j])
                        : null;
                    if (i > 0) VaeTiling.BlendVertical(rowTails[j]!, tile, yOverlap[i - 1]);
                    if (j > 0) VaeTiling.BlendHorizontal(leftTail!, tile, xOverlap[j - 1]);
                    leftTail?.Dispose();
                    leftTail = nextLeft;

                    cropH = i < ys.Length - 1 ? tileH - yOverlap[i] : tileH;
                    int cropW = j < xs.Length - 1 ? tileW - xOverlap[j] : tileW;
                    Tensor cropped = cropH == tileH && cropW == tileW ? tile : VaeTiling.CropTile(tile, 1, planes, cropH, cropW);
                    WriteInto(canvas, cropped, planes, height, width, outY, outX, cropH, cropW);
                    if (!ReferenceEquals(cropped, tile)) cropped.Dispose();
                    outX += cropW;
                }
                leftTail?.Dispose();
                foreach (Tensor? t in rowTails) t?.Dispose();
                rowTails = newTails;
                outY += cropH;
                // A short row would leave an unwritten band in the canvas rather than throwing.
                if (outX != width)
                    throw new InvalidOperationException($"tile row {i} covered {outX} of {width} columns.");
            }
            if (outY != height)
                throw new InvalidOperationException($"tile rows covered {outY} of {height} rows.");
        }
        catch
        {
            canvas.Dispose();
            throw;
        }
        finally
        {
            foreach (Tensor? t in rowTails) t?.Dispose();
        }
        return canvas;
    }

    private (int[] Start, int[] Length, int[] Overlap) SplitTiles(int inputLen) => _config.SplitTiles(inputLen);

    // ── ViT3D forward ───────────────────────────────────────────────────

    /// <summary>One tile through the transformer: <c>[1, C_lat, T, H, W]</c> → <c>[1, 3·T·4, H·16, W·16]</c> (the
    /// channel and frame axes stay folded so the tiling helpers can treat it as a 4-D plane stack).</summary>
    private Tensor DecodePixels(IBackend backend, Tensor z, int zt, int zh, int zw)
    {
        int dim = _config.Dim;
        int n = zt * zh * zw;
        int seq = n + _config.NumSuffixTokens;

        using Tensor rows = new Tensor(new TensorShape(1, n, _config.LatentChannels), DType.F32);
        backend.Transpose2D(rows, z, _config.LatentChannels, n);
        using Tensor projected = new Tensor(new TensorShape(n, _config.ZChannels), DType.F32);
        backend.Linear(projected, rows, _postQuantW!, _postQuantB);
        using Tensor embedded = new Tensor(new TensorShape(n, dim), DType.F32);
        backend.Linear(embedded, projected, _xEmbedW!, _xEmbedB);

        using Tensor h = new Tensor(new TensorShape(seq, dim), DType.F32);
        backend.Concat(h, [embedded, _registerTokens!, _clsToken!], 0);

        (Tensor cos, Tensor sin) = BuildRope(zt, zh, zw);
        try
        {
            foreach (MiniMaxH3VideoVaeVitBlock block in _blocks)
            {
                block.Forward(backend, h, cos, sin, _qkNormOnes!);
            }
        }
        finally
        {
            cos.Dispose();
            sin.Dispose();
        }

        using Tensor normed = new Tensor(new TensorShape(seq, dim), DType.F32);
        backend.LayerNorm(normed, h, _normOutW!, _normOutB!, _config.Eps);
        using Tensor projFull = new Tensor(new TensorShape(seq, _config.PatchDim), DType.F32);
        backend.Linear(projFull, normed, _projOutW!, _projOutB);
        using Tensor patches = new Tensor(new TensorShape(n, _config.PatchDim), DType.F32);
        backend.SliceRows(patches, projFull, 0);
        backend.Clamp(patches, patches, 0f, 1f);
        backend.Scale(patches, patches, 2f);
        backend.AddScalar(patches, patches, -1f);
        return Unpatchify(patches, zt, zh, zw);
    }

    /// <summary>Length-normalized coordinates on (t, h, w), pre-scaled by 2π (the reference's <c>use_angle</c>);
    /// the 5 suffix rows keep position 0.</summary>
    private (Tensor Cos, Tensor Sin) BuildRope(int zt, int zh, int zw)
    {
        int n = zt * zh * zw;
        int seq = n + _config.NumSuffixTokens;
        double[] positions = new double[(long)seq * 3];
        const double TwoPi = 2.0 * Math.PI;
        int idx = 0;
        for (int t = 0; t < zt; t++)
        {
            double ct = TwoPi * Coord(t, zt);
            for (int y = 0; y < zh; y++)
            {
                double cy = TwoPi * Coord(y, zh);
                for (int x = 0; x < zw; x++)
                {
                    positions[idx++] = ct;
                    positions[idx++] = cy;
                    positions[idx++] = TwoPi * Coord(x, zw);
                }
            }
        }
        return MiniMaxH3Rope.BuildTables(positions, _invFreq, _config.DimHead);
    }

    private static double Coord(int index, int size) => 2.0 * ((index + 0.5) / size) - 1.0;

    /// <summary>Scatters each token's <c>[C, pt, ph, pw]</c> patch vector into the frame stack. No backend op covers
    /// this 8-D permute, so it is a run-length memory move (one <c>pw</c>-wide copy per row), not element math.</summary>
    private Tensor Unpatchify(Tensor patches, int zt, int zh, int zw)
    {
        int c = _config.OutChannels, pt = _config.VaeRatioT, p = _config.VaeRatio, patchDim = _config.PatchDim;
        int frames = zt * pt, height = zh * p, width = zw * p;
        Tensor outT = new Tensor(new TensorShape(1, (long)c * frames, height, width), DType.F32);
        float* src = (float*)patches.DataPointer;
        float* dst = (float*)outT.DataPointer;
        long plane = (long)height * width;
        long rowBytes = (long)p * sizeof(float);
        for (int t = 0; t < zt; t++)
        {
            for (int y = 0; y < zh; y++)
            {
                for (int x = 0; x < zw; x++)
                {
                    long row = (((long)t * zh + y) * zw + x) * patchDim;
                    for (int ci = 0; ci < c; ci++)
                    {
                        for (int f = 0; f < pt; f++)
                        {
                            long planeBase = ((long)ci * frames + t * pt + f) * plane;
                            for (int py = 0; py < p; py++)
                            {
                                long srcOff = row + (((long)ci * pt + f) * p + py) * p;
                                long dstOff = planeBase + ((long)y * p + py) * width + (long)x * p;
                                Buffer.MemoryCopy(src + srcOff, dst + dstOff, rowBytes, rowBytes);
                            }
                        }
                    }
                }
            }
        }
        return outT;
    }

    // ── Frame / token plumbing ──────────────────────────────────────────

    private Tensor SliceTokens(Tensor z, int zh, int zw, int start, int count)
    {
        int c = _config.LatentChannels;
        long cell = (long)zh * zw;
        int total = (int)(z.ElementCount / (c * cell));
        Tensor outT = new Tensor(new TensorShape([1L, c, count, zh, zw]), DType.F32);
        float* src = (float*)z.DataPointer;
        float* dst = (float*)outT.DataPointer;
        long bytes = count * cell * sizeof(float);
        for (int ci = 0; ci < c; ci++)
        {
            Buffer.MemoryCopy(src + ((long)ci * total + start) * cell, dst + (long)ci * count * cell, bytes, bytes);
        }
        return outT;
    }

    private Tensor PadTokens(Tensor z, int zt, int zh, int zw, int padTokens)
    {
        int c = _config.LatentChannels;
        long cell = (long)zh * zw;
        int total = zt + padTokens;
        Tensor outT = new Tensor(new TensorShape([1L, c, total, zh, zw]), DType.F32);
        float* src = (float*)z.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int ci = 0; ci < c; ci++)
        {
            long head = (long)zt * cell * sizeof(float);
            Buffer.MemoryCopy(src + (long)ci * zt * cell, dst + (long)ci * total * cell, head, head);
            for (int k = 0; k < padTokens; k++)
            {
                Buffer.MemoryCopy(src + ((long)ci * zt + zt - 1) * cell,
                    dst + ((long)ci * total + zt + k) * cell, cell * sizeof(float), cell * sizeof(float));
            }
        }
        return outT;
    }

    /// <summary>Frame sub-range of a folded plane stack; <paramref name="asRank5"/> emits the public
    /// <c>[1, C, F, H, W]</c> shape instead of the internal <c>[1, C·F, H, W]</c>.</summary>
    private Tensor SliceFrames(Tensor tile, int frames, int height, int width, int start, int count, bool asRank5)
    {
        int c = _config.OutChannels;
        long plane = (long)height * width;
        TensorShape shape = asRank5
            ? new TensorShape([1L, c, count, height, width])
            : new TensorShape(1, (long)c * count, height, width);
        Tensor outT = new Tensor(shape, DType.F32);
        float* src = (float*)tile.DataPointer;
        float* dst = (float*)outT.DataPointer;
        long bytes = count * plane * sizeof(float);
        for (int ci = 0; ci < c; ci++)
        {
            Buffer.MemoryCopy(src + ((long)ci * frames + start) * plane, dst + (long)ci * count * plane, bytes, bytes);
        }
        return outT;
    }

    /// <summary>Ramp-blends <paramref name="part"/>'s leading frames against <paramref name="previous"/>'s trailing
    /// frames, in place. Viewing the stack as <c>[1, C, F, H·W]</c> makes the frame axis the tiling helper's row axis.</summary>
    private void BlendFrames(Tensor previous, Tensor part, int partFrames, int plane)
    {
        int c = _config.OutChannels;
        int prevFrames = (int)(previous.ElementCount / ((long)c * plane));
        using Tensor a = new Tensor((void*)previous.DataPointer, new TensorShape(1, c, prevFrames, plane), DType.F32);
        using Tensor b = new Tensor((void*)part.DataPointer, new TensorShape(1, c, partFrames, plane), DType.F32);
        VaeTiling.BlendVertical(a, b, _config.FrameOverlap);
    }

    private void WritePart(ref Tensor? dec, Tensor part, int partFrames, int height, int width, int outputFrames, ref int writePos)
    {
        int c = _config.OutChannels;
        dec ??= new Tensor(new TensorShape([1L, c, outputFrames, height, width]), DType.F32);
        int copy = Math.Min(partFrames, Math.Max(0, outputFrames - writePos));
        if (copy <= 0)
        {
            return;
        }
        long plane = (long)height * width;
        float* src = (float*)part.DataPointer;
        float* dst = (float*)dec.DataPointer;
        long bytes = copy * plane * sizeof(float);
        for (int ci = 0; ci < c; ci++)
        {
            Buffer.MemoryCopy(src + (long)ci * partFrames * plane,
                dst + ((long)ci * outputFrames + writePos) * plane, bytes, bytes);
        }
        writePos += copy;
    }

    private static void WriteInto(Tensor canvas, Tensor tile, int planes, int height, int width,
        int outY, int outX, int tileH, int tileW)
    {
        float* src = (float*)tile.DataPointer;
        float* dst = (float*)canvas.DataPointer;
        long bytes = (long)tileW * sizeof(float);
        for (int pIdx = 0; pIdx < planes; pIdx++)
        {
            for (int y = 0; y < tileH; y++)
            {
                Buffer.MemoryCopy(src + ((long)pIdx * tileH + y) * tileW,
                    dst + ((long)pIdx * height + outY + y) * width + outX, bytes, bytes);
            }
        }
    }

    // ── Weight preparation ──────────────────────────────────────────────

    /// <summary>Folds the encoder's ImageNet pixel normalization into a private copy of <c>proj_out</c>: row
    /// <c>r</c> covers channel <c>r / (pt·p·p)</c>, so scaling the row by that channel's std and shifting the bias by
    /// its mean makes the decode tail a plain clamp/scale on whole tensors.</summary>
    private void FoldProjOut(IReadOnlyDictionary<string, Tensor> w)
    {
        Tensor weight = MiniMaxH3VideoVaeWeights.Raw(w, "decoder.proj_out.weight");
        Tensor source = weight.DType == DType.F32 ? weight : weight.CastTo(DType.F32);
        try
        {
            int rows = (int)source.Shape[0], cols = (int)source.Shape[1];
            int perChannel = _config.VaeRatioT * _config.VaeRatio * _config.VaeRatio;
            Tensor folded = new Tensor(new TensorShape(rows, cols), DType.F32);
            Tensor bias = new Tensor(new TensorShape(rows), DType.F32);
            _owned.Add(folded);
            _owned.Add(bias);

            Tensor? sourceBias = MiniMaxH3VideoVaeWeights.OptionalF32(w, "decoder.proj_out.bias", _owned);
            float* src = (float*)source.DataPointer;
            float* dst = (float*)folded.DataPointer;
            float* bDst = (float*)bias.DataPointer;
            float* bSrc = sourceBias is null ? null : (float*)sourceBias.DataPointer;
            for (int r = 0; r < rows; r++)
            {
                int channel = r / perChannel;
                float std = _config.PixelStd[channel];
                for (int k = 0; k < cols; k++)
                {
                    dst[(long)r * cols + k] = src[(long)r * cols + k] * std;
                }
                bDst[r] = (bSrc is null ? 0f : bSrc[r]) * std + _config.PixelMean[channel];
            }
            _projOutW = folded;
            _projOutB = bias;
        }
        finally
        {
            if (!ReferenceEquals(source, weight)) source.Dispose();
        }
    }

    /// <summary>Copies a conv/parameter tensor into a plain <c>[rows, cols]</c> F32 matrix for the Linear path.</summary>
    private Tensor Reshape2d(Tensor t, int rows, int cols)
    {
        if (t.ElementCount != (long)rows * cols)
            throw new ArgumentException($"cannot view {t.Shape} as [{rows}, {cols}].", nameof(t));
        Tensor source = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        try
        {
            Tensor outT = new Tensor(new TensorShape(rows, cols), DType.F32);
            new ReadOnlySpan<float>((float*)source.DataPointer, rows * cols).CopyTo(outT.AsSpan<float>());
            _owned.Add(outT);
            return outT;
        }
        finally
        {
            if (!ReferenceEquals(source, t)) source.Dispose();
        }
    }

    private Tensor FromArray(float[] values)
    {
        Tensor t = new Tensor(new TensorShape(1, values.Length), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        _owned.Add(t);
        return t;
    }

    private static Tensor Borrow(Tensor t) => new Tensor((void*)t.DataPointer, t.Shape, t.DType);

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
    }
}
