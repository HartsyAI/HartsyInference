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
    private readonly Tensor[] _weight2d;  // kt slices of [cOut, cIn, kh, kw]
    private readonly Tensor? _bias;

    /// <summary>Builds the op from a 5-D conv weight <c>[cOut, cIn, kt, kh, kw]</c> and optional bias, pre-slicing the kt temporal taps into 2-D conv weights. <paramref name="padT"/>/<paramref name="padH"/>/<paramref name="padW"/> are the nn.Conv3d <c>padding</c> values (temporal becomes <c>2·padT</c> causal-left). <paramref name="replicateFirstPad"/> fills the leading causal frames with copies of the input's first frame (LTX-Video, HunyuanVideo) instead of zeros (Wan2.2). <paramref name="spatialReplicatePad"/> pads H/W borders by edge replication (HunyuanVideo <c>F.pad(mode="replicate")</c>) instead of zeros.</summary>
    public CausalConv3d(Tensor weight5d, Tensor? bias,
        int strideT = 1, int strideH = 1, int strideW = 1,
        int padT = 0, int padH = 0, int padW = 0, bool replicateFirstPad = false, bool causal = true,
        bool spatialReplicatePad = false)
    {
        _replicateFirstPad = replicateFirstPad;
        _spatialReplicatePad = spatialReplicatePad;
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
        Tensor src = weight5d.DType == DType.F32 ? weight5d : weight5d.CastTo(DType.F32);
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

        Tensor output = new Tensor(new TensorShape([(long)batch, _cOut, tout, hOut, wOut]), DType.F32);
        new Span<float>((float*)output.DataPointer, checked((int)output.Shape.ElementCount)).Clear();
        float* outPtr = (float*)output.DataPointer;
        long frameOut = (long)hOut * wOut;

        for (int to = 0; to < tout; to++)
        {
            Tensor? acc = null;
            for (int dt = 0; dt < _kt; dt++)
            {
                int srcT = to * _strideT + dt;
                Tensor? frame = ResolveFrame(srcT, zeroPad, cacheLen, cacheFrames, input, batch, h, w);
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
                else { AddInPlace(acc, conv); conv.Dispose(); }
            }

            // Write the accumulated (+bias) frame into output[:, :, to].
            if (acc is null)
            {
                acc = new Tensor(new TensorShape(batch, _cOut, hOut, wOut), DType.F32);
                new Span<float>((float*)acc.DataPointer, checked((int)acc.Shape.ElementCount)).Clear();
            }
            float* ap = (float*)acc.DataPointer;
            float* bp = _bias is null ? null : (float*)_bias.DataPointer;
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < _cOut; co++)
                {
                    float biasVal = bp is null ? 0f : bp[co];
                    long accBase = ((long)b * _cOut + co) * frameOut;
                    long outBase = (((long)b * _cOut + co) * tout + to) * frameOut;
                    for (long i = 0; i < frameOut; i++)
                        outPtr[outBase + i] = ap[accBase + i] + biasVal;
                }
            acc.Dispose();
        }
        return output;
    }

    /// <summary>Extracts temporal frame <paramref name="srcT"/> of the (conceptually) left-padded input as a <c>[B, cIn, H, W]</c> tensor, or null for an all-zero pad frame.</summary>
    private Tensor? ResolveFrame(int srcT, int zeroPad, int cacheLen, Tensor? cache, Tensor input, int batch, int h, int w)
    {
        long frame = (long)h * w;
        Tensor outF = new Tensor(new TensorShape(batch, _cIn, h, w), DType.F32);
        float* dp = (float*)outF.DataPointer;
        if (srcT < zeroPad)                                     // leading causal-pad frame
        {
            if (!_replicateFirstPad) { outF.Dispose(); return null; }   // zero (Wan): no contribution
            // LTX: replicate the input's first frame.
            int tinR = (int)input.Shape[2];
            float* ipR = (float*)input.DataPointer;
            for (int b = 0; b < batch; b++)
                for (int ci = 0; ci < _cIn; ci++)
                {
                    long srcOff = ((long)b * _cIn + ci) * tinR * frame;   // frame 0
                    long dstOff = ((long)b * _cIn + ci) * frame;
                    Buffer.MemoryCopy(ipR + srcOff, dp + dstOff, frame * 4, frame * 4);
                }
            return outF;
        }
        int afterZero = srcT - zeroPad;
        if (afterZero < cacheLen)                               // cache frame
        {
            float* cp = (float*)cache!.DataPointer;
            for (int b = 0; b < batch; b++)
                for (int ci = 0; ci < _cIn; ci++)
                {
                    long srcOff = (((long)b * _cIn + ci) * cacheLen + afterZero) * frame;
                    long dstOff = ((long)b * _cIn + ci) * frame;
                    Buffer.MemoryCopy(cp + srcOff, dp + dstOff, frame * 4, frame * 4);
                }
        }
        else                                                    // real input frame (or trailing edge-replicate)
        {
            int ti = afterZero - cacheLen;
            int tin = (int)input.Shape[2];
            if (ti >= tin) ti = tin - 1;                        // non-causal trailing pad: replicate the last frame
            float* ip = (float*)input.DataPointer;
            for (int b = 0; b < batch; b++)
                for (int ci = 0; ci < _cIn; ci++)
                {
                    long srcOff = (((long)b * _cIn + ci) * tin + ti) * frame;
                    long dstOff = ((long)b * _cIn + ci) * frame;
                    Buffer.MemoryCopy(ip + srcOff, dp + dstOff, frame * 4, frame * 4);
                }
        }
        return outF;
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
