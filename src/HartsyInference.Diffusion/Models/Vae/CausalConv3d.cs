using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Causal 3D convolution (Wan2.2 / video-VAE family), ported from <c>CausalConv3d(nn.Conv3d)</c> in upstream <c>vae2_2.py</c>. Reusable by any 3D causal VAE (Wan, LTX, future video models).
///
/// <para><b>Design:</b> decomposed into existing <see cref="IBackend.Conv2D"/> calls over the temporal kernel taps, so it runs on EVERY backend (CPU/CUDA/Vulkan) with no new kernel — <c>out[t] = Σ_dt Conv2D(paddedInput[t·strideT + dt], weight[:,:,dt])</c>. Spatial padding is symmetric; temporal padding is <b>causal</b> (left-only, <c>2·padT</c> frames) so a frame never sees the future. An optional <paramref name="cacheFrames"/> prepend supports chunked/streaming decode (the caller owns the cache; <c>CACHE_T=2</c> for Wan2.2).</para>
///
/// <para>Operates on 5-D tensors <c>[B, C, T, H, W]</c>. Bias is added once after the temporal sum.</para></summary>
public sealed unsafe class CausalConv3d
{
    /// <summary>Test-only escape hatch: forces the original per-frame accumulate loop instead of the batched fast path,
    /// so a parity test can prove the two produce identical output. Never set in production.</summary>
    public static bool DisableBatchedPath;

    private readonly int _cOut;
    private readonly int _cIn;
    private readonly int _kt;
    private readonly int _kh;
    private readonly int _kw;
    private readonly int _strideT;
    private readonly int _strideH;
    private readonly int _strideW;
    private readonly int _padH;
    private readonly int _padW;
    private readonly int _padTLeft;   // causal: 2 * padT frames on the left only; non-causal: padT on the left
    private readonly int _padTRight;  // non-causal: padT frames on the right (replicate last frame)
    private readonly bool _causal;
    private readonly bool _replicateFirstPad;   // LTX/HunyuanVideo: pad with copies of the edge frame instead of zeros
    private readonly bool _spatialReplicatePad; // HunyuanVideo: F.pad(mode="replicate") spatially instead of zero-pad
    private readonly bool _spatialReflectPad;   // LTX-2: F.pad(mode="reflect") spatially instead of zero-pad
    private readonly Tensor[] _weight2d;  // kt slices of [cOut, cIn, kh, kw]
    private readonly Tensor? _bias;

    /// <summary>Builds the op from a 5-D conv weight <c>[cOut, cIn, kt, kh, kw]</c> and optional bias, pre-slicing the kt temporal taps into 2-D conv weights. <paramref name="padT"/>/<paramref name="padH"/>/<paramref name="padW"/> are the nn.Conv3d <c>padding</c> values (temporal becomes <c>2·padT</c> causal-left). <paramref name="replicateFirstPad"/> fills the leading causal frames with copies of the input's first frame (LTX-Video, HunyuanVideo) instead of zeros (Wan2.2). <paramref name="spatialReplicatePad"/> pads H/W borders by edge replication (HunyuanVideo <c>F.pad(mode="replicate")</c>) instead of zeros. <paramref name="spatialReflectPad"/> pads H/W borders by mirror-reflection (LTX-2 <c>F.pad(mode="reflect")</c>) instead of zeros — mutually exclusive with <paramref name="spatialReplicatePad"/>, and only supported on the batch-1 fast path with no streaming cache.</summary>
    public CausalConv3d(Tensor weight5d, Tensor? bias,
        int strideT = 1, int strideH = 1, int strideW = 1,
        int padT = 0, int padH = 0, int padW = 0, bool replicateFirstPad = false, bool causal = true,
        bool spatialReplicatePad = false, bool spatialReflectPad = false)
    {
        _replicateFirstPad = replicateFirstPad;
        _spatialReplicatePad = spatialReplicatePad;
        _spatialReflectPad = spatialReflectPad;
        _causal = causal;
        if (weight5d.Shape.Rank != 5)
            throw new ArgumentException($"weight must be 5-D [cOut,cIn,kt,kh,kw], got {weight5d.Shape}.", nameof(weight5d));
        _cOut = (int)weight5d.Shape[0];
        _cIn = (int)weight5d.Shape[1];
        _kt = (int)weight5d.Shape[2];
        _kh = (int)weight5d.Shape[3];
        _kw = (int)weight5d.Shape[4];
        _strideT = strideT; _strideH = strideH; _strideW = strideW;
        _padH = padH; _padW = padW;
        _padTLeft = causal ? 2 * padT : padT;     // non-causal splits the temporal pad symmetrically
        _padTRight = causal ? 0 : padT;
        _bias = bias is null ? null : (bias.DType == DType.F32 ? bias : bias.CastTo(DType.F32));
        _weight2d = SliceTemporal(weight5d);
    }

    /// <summary>Output channel count.</summary>
    public int OutChannels => _cOut;

    /// <summary>Number of temporal padding frames (causal, left). With a streaming cache of this many frames, no zero-padding is needed after the first chunk.</summary>
    public int TemporalPad => _padTLeft;

    /// <summary>Pre-slices a contiguous <c>[cOut,cIn,kt,kh,kw]</c> weight into kt tensors of <c>[cOut,cIn,kh,kw]</c>.</summary>
    private Tensor[] SliceTemporal(Tensor weight5d)
    {
        // The cast temporary must stay reachable for the whole copy loop: `sp` is a raw pointer, so once the local
        // is dead the GC can finalize the Tensor mid-loop and free the buffer under us (AccessViolation under
        // memory pressure). The explicit Dispose at the end doubles as the keep-alive AND fixes the leak (the
        // full VAE's cast slices are ~240 MB otherwise held until GC).
        Tensor? cast = weight5d.DType == DType.F32 ? null : weight5d.CastTo(DType.F32);
        Tensor src = cast ?? weight5d;
        float* sp = (float*)src.DataPointer;
        int khw = _kh * _kw;
        Tensor[] slices = new Tensor[_kt];
        for (int dt = 0; dt < _kt; dt++)
        {
            Tensor w = new Tensor(new TensorShape(_cOut, _cIn, _kh, _kw), DType.F32);
            float* dp = (float*)w.DataPointer;
            for (int co = 0; co < _cOut; co++)
                for (int ci = 0; ci < _cIn; ci++)
                {
                    long srcOff = (((long)co * _cIn + ci) * _kt + dt) * khw;
                    long dstOff = ((long)co * _cIn + ci) * khw;
                    Buffer.MemoryCopy(sp + srcOff, dp + dstOff, (long)khw * 4, (long)khw * 4);
                }
            slices[dt] = w;
        }
        cast?.Dispose();
        GC.KeepAlive(weight5d);
        return slices;
    }

    /// <summary>Enumerates the (sliced) weight + bias for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in _weight2d) yield return w;
        if (_bias is not null) yield return _bias;
    }

    /// <summary>Runs the causal 3D conv. <paramref name="input"/> is <c>[B, cIn, Tin, H, W]</c>; <paramref name="cacheFrames"/> (optional <c>[B, cIn, C, H, W]</c>) prepends streaming context, replacing that many zero-pad frames. Returns <c>[B, cOut, Tout, H', W']</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor input, Tensor? cacheFrames = null)
    {
        int batch = (int)input.Shape[0];
        int tin = (int)input.Shape[2];
        int h = (int)input.Shape[3];
        int w = (int)input.Shape[4];
        int cacheLen = cacheFrames is null ? 0 : (int)cacheFrames.Shape[2];

        int paddedT = _padTLeft + tin + _padTRight;             // cache occupies first cacheLen of the left pad
        int tout = (paddedT - _kt) / _strideT + 1;
        if (tout < 1)
            throw new ArgumentException($"CausalConv3d produced Tout={tout} (Tin={tin}, kt={_kt}, padTLeft={_padTLeft}).");
        int hOut = (h + 2 * _padH - _kh) / _strideH + 1;
        int wOut = (w + 2 * _padW - _kw) / _strideW + 1;
        int zeroPad = _padTLeft - cacheLen;                      // leading all-zero frames

        // FAST PATH — batched convolution (B=1: every video-VAE decode). The per-frame loop below fires tout·kt tiny
        // Conv2D+glue ops per layer (plus HOST replicate-pad loops on the HunyuanVideo/LTX path), leaving the GPU
        // idle ~66% on host dispatch. Instead build the frame-major padded input ONCE (temporal zero/replicate pad +
        // spatial edge-replicate baked in), run kt batched Conv2D calls over ALL frames, then a temporal gather-sum —
        // ~(2+2·kt) ops instead of tout·kt·4. Zero-pad frames convolve to 0 (no bias in the per-tap Conv2D),
        // matching the per-frame skip; replicate-pad frames contribute exactly like F.pad(mode="replicate").
        if (batch == 1 && !DisableBatchedPath)
        {
            bool reflectPre = _spatialReflectPad && (_padH > 0 || _padW > 0);
            if (reflectPre && cacheFrames is not null)
                throw new NotSupportedException("CausalConv3d spatialReflectPad does not support a streaming cacheFrames prepend (LTX-2 decode is one-shot, non-streaming).");
            Tensor? reflectPadded = reflectPre ? ReflectPadSpatial5D(input, batch, tin, h, w) : null;
            Tensor convInput = reflectPadded ?? input;
            int hSrc = reflectPre ? h + 2 * _padH : h;
            int wSrc = reflectPre ? w + 2 * _padW : w;

            bool spatialPre = !reflectPre && _spatialReplicatePad && (_padH > 0 || _padW > 0);
            int convPadH = spatialPre || reflectPre ? 0 : _padH;
            int convPadW = spatialPre || reflectPre ? 0 : _padW;
            int hp = spatialPre || reflectPre ? hSrc : h;
            int wp = spatialPre || reflectPre ? wSrc : w;
            using Tensor padded = new Tensor(new TensorShape(paddedT, _cIn, hp, wp), DType.F32);
            backend.BuildPaddedFrames(padded, convInput, cacheFrames, zeroPad, _replicateFirstPad,
                spatialPre ? _padH : 0, spatialPre ? _padW : 0);
            reflectPadded?.Dispose();
            Tensor fastOut = new Tensor(new TensorShape([1L, _cOut, tout, hOut, wOut]), DType.F32);
            backend.FillBias(fastOut, _bias);
            for (int dt = 0; dt < _kt; dt++)
            {
                using Tensor convDt = new Tensor(new TensorShape(paddedT, _cOut, hOut, wOut), DType.F32);
                backend.Conv2D(convDt, padded, _weight2d[dt], null, _strideH, _strideW, convPadH, convPadW);
                backend.AccumulateTap(fastOut, convDt, dt, _strideT);
            }
            return fastOut;
        }

        // Device-resident output: every temporal slot is written by WriteVaeFrame below (GPU), so no host clear.
        Tensor output = new Tensor(new TensorShape([(long)batch, _cOut, tout, hOut, wOut]), DType.F32);

        for (int to = 0; to < tout; to++)
        {
            Tensor? acc = null;
            for (int dt = 0; dt < _kt; dt++)
            {
                int srcT = to * _strideT + dt;
                Tensor? frame = ResolveFrame(backend, srcT, zeroPad, cacheLen, cacheFrames, input, batch, h, w);
                if (frame is null) continue;                    // zero-pad frame contributes nothing
                if (_spatialReplicatePad && (_padH > 0 || _padW > 0))
                {
                    Tensor padded = ReplicatePadSpatial(frame, batch, h, w);
                    frame.Dispose();
                    frame = padded;
                }
                Tensor conv = new Tensor(new TensorShape(batch, _cOut, hOut, wOut), DType.F32);
                backend.Conv2D(conv, frame, _weight2d[dt], null, _strideH, _strideW,
                    _spatialReplicatePad ? 0 : _padH, _spatialReplicatePad ? 0 : _padW);
                frame.Dispose();
                if (acc is null) { acc = conv; }
                else { backend.Add(acc, acc, conv); conv.Dispose(); }   // accumulate taps on-GPU (was CPU AddInPlace → D2H per tap)
            }

            // Write the accumulated (+bias) frame into output[:, :, to], on-GPU (was a host pointer loop).
            if (acc is null)
            {
                acc = new Tensor(new TensorShape(batch, _cOut, hOut, wOut), DType.F32);
                new Span<float>((float*)acc.DataPointer, checked((int)acc.Shape.ElementCount)).Clear();
            }
            backend.WriteVaeFrame(output, acc, _bias, to);
            acc.Dispose();
        }
        return output;
    }

    /// <summary>Extracts temporal frame <paramref name="srcT"/> of the (conceptually) left-padded input as a <c>[B, cIn, H, W]</c> tensor, or null for an all-zero pad frame.</summary>
    private Tensor? ResolveFrame(IBackend backend, int srcT, int zeroPad, int cacheLen, Tensor? cache, Tensor input, int batch, int h, int w)
    {
        long frame = (long)h * w;
        Tensor outF = new Tensor(new TensorShape(batch, _cIn, h, w), DType.F32);
        if (srcT < zeroPad)                                     // leading causal-pad frame
        {
            if (!_replicateFirstPad) { outF.Dispose(); return null; }   // zero (Wan): no contribution
            // LTX: replicate the input's first frame (CPU — rare replicate path).
            int tinR = (int)input.Shape[2];
            float* dp = (float*)outF.DataPointer, ipR = (float*)input.DataPointer;
            for (int b = 0; b < batch; b++)
                for (int ci = 0; ci < _cIn; ci++)
                    Buffer.MemoryCopy(ipR + ((long)b * _cIn + ci) * tinR * frame, dp + ((long)b * _cIn + ci) * frame, frame * 4, frame * 4);
            return outF;
        }
        int afterZero = srcT - zeroPad;
        if (afterZero < cacheLen)                               // cache frame → GPU strided extract
        {
            backend.ExtractVaeFrame(outF, cache!, afterZero);
        }
        else                                                    // real input frame (or trailing edge-replicate)
        {
            int ti = afterZero - cacheLen;
            int tin = (int)input.Shape[2];
            if (ti >= tin) ti = tin - 1;                        // non-causal trailing pad: replicate the last frame
            backend.ExtractVaeFrame(outF, input, ti);
        }
        return outF;
    }

    /// <summary>Mirror-reflect pads a <c>[B, cIn, T, H, W]</c> tensor's H/W borders to <c>[B, cIn, T, H+2·padH, W+2·padW]</c>
    /// (PyTorch <c>F.pad(mode="reflect")</c>: the border pixel itself is NOT repeated — index <c>-1</c> maps to source
    /// index <c>1</c>, not <c>0</c> — unlike <see cref="ReplicatePadSpatial"/>'s edge-clamp). Used by the LTX-2 VAE decoder,
    /// whose diffusers reference (<c>LTX2VideoDecoder3d</c>) defaults every conv to <c>spatial_padding_mode="reflect"</c>.</summary>
    private Tensor ReflectPadSpatial5D(Tensor input, int batch, int t, int h, int w)
    {
        int hp = h + 2 * _padH, wp = w + 2 * _padW;
        Tensor outF = new Tensor(new TensorShape([(long)batch, _cIn, t, hp, wp]), DType.F32);
        Tensor inF32 = input.DType == DType.F32 ? input : input.CastTo(DType.F32);
        float* sp = (float*)inF32.DataPointer;
        float* dp = (float*)outF.DataPointer;
        long frame = (long)h * w;
        long frameP = (long)hp * wp;
        for (int b = 0; b < batch; b++)
            for (int ci = 0; ci < _cIn; ci++)
                for (int ti = 0; ti < t; ti++)
                {
                    long srcBase = (((long)b * _cIn + ci) * t + ti) * frame;
                    long dstBase = (((long)b * _cIn + ci) * t + ti) * frameP;
                    for (int y = 0; y < hp; y++)
                    {
                        int sy = ReflectIndex(y - _padH, h);
                        for (int x = 0; x < wp; x++)
                        {
                            int sx = ReflectIndex(x - _padW, w);
                            dp[dstBase + (long)y * wp + x] = sp[srcBase + (long)sy * w + sx];
                        }
                    }
                }
        if (!ReferenceEquals(inF32, input)) inF32.Dispose();
        return outF;
    }

    /// <summary>Maps an out-of-range index to its PyTorch <c>reflect</c>-mode source index (no edge repeat):
    /// <c>-1 → 1</c>, <c>-2 → 2</c>, <c>len → len-2</c>, etc. Only correct for pad ≤ len-1 (true here: pad is always 1).</summary>
    private static int ReflectIndex(int i, int len)
    {
        if (i < 0) return -i;
        if (i >= len) return 2 * (len - 1) - i;
        return i;
    }

    /// <summary>Edge-replicate pads a <c>[B, cIn, H, W]</c> frame to <c>[B, cIn, H+2·padH, W+2·padW]</c> (HunyuanVideo <c>F.pad(mode="replicate")</c>).</summary>
    private Tensor ReplicatePadSpatial(Tensor frame, int batch, int h, int w)
    {
        int hp = h + 2 * _padH, wp = w + 2 * _padW;
        Tensor outF = new Tensor(new TensorShape(batch, _cIn, hp, wp), DType.F32);
        float* sp = (float*)frame.DataPointer;
        float* dp = (float*)outF.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int ci = 0; ci < _cIn; ci++)
            {
                long srcBase = ((long)b * _cIn + ci) * h * w;
                long dstBase = ((long)b * _cIn + ci) * hp * wp;
                for (int y = 0; y < hp; y++)
                {
                    int sy = Math.Clamp(y - _padH, 0, h - 1);
                    for (int x = 0; x < wp; x++)
                    {
                        int sx = Math.Clamp(x - _padW, 0, w - 1);
                        dp[dstBase + (long)y * wp + x] = sp[srcBase + (long)sy * w + sx];
                    }
                }
            }
        return outF;
    }

    private static void AddInPlace(Tensor acc, Tensor add)
    {
        long n = acc.Shape.ElementCount;
        float* a = (float*)acc.DataPointer;
        float* b = (float*)add.DataPointer;
        for (long i = 0; i < n; i++) a[i] += b[i];
    }
}
