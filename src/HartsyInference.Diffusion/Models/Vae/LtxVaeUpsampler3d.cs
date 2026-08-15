using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>LTX-Video VAE 3D upsampler (<c>LTXVideoUpsampler3d</c>), ported from diffusers. A causal conv followed by a pixel-shuffle that trades channels for spatiotemporal resolution: <c>[B, C, F, H, W] → [B, C/upscale, F·st0−(st0−1), H·st1, W·st2]</c>. With <c>residual=True</c> the input is pixel-shuffled + channel-repeated and added. Reuses <see cref="CausalConv3d"/> (replicate-first-frame padding).</summary>
public sealed unsafe class LtxVaeUpsampler3d
{
    private readonly int _inC;
    private readonly (int T, int H, int W) _stride;
    private readonly int _upscale;
    private readonly bool _residual;
    private readonly bool _isCausal;
    private readonly bool _spatialReflectPad;
    private readonly DType _computeDtype;
    private CausalConv3d? _conv;

    public LtxVaeUpsampler3d(int inC, (int T, int H, int W) stride, int upscaleFactor, bool residual, bool isCausal = true,
        bool spatialReflectPad = false, DType? computeDtype = null)
    {
        _computeDtype = computeDtype ?? DType.F32;
        _inC = inC;
        _stride = stride;
        _upscale = upscaleFactor;
        _residual = residual;
        _isCausal = isCausal;
        _spatialReflectPad = spatialReflectPad;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _conv = new CausalConv3d(w[$"{prefix}.conv.conv.weight"], Bias(w, $"{prefix}.conv.conv.bias"),
            padT: 1, padH: 1, padW: 1, replicateFirstPad: true, causal: _isCausal, spatialReflectPad: _spatialReflectPad, computeDtype: _computeDtype);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_conv is not null) foreach (Tensor t in _conv.EnumerateWeights()) yield return t;
    }

    /// <summary>Compact geometry description for a parity harness to assert against the reference's own modules.</summary>
    internal string Describe() => $"stride({_stride.T},{_stride.H},{_stride.W}),upscale{_upscale}";

    /// <summary>The two behaviour flags the geometry does NOT pin down, which the reference takes from the
    /// checkpoint's config rather than from weight shapes.</summary>
    internal string DescribeFlags() => $"residual{(_residual ? 1 : 0)},reflect{(_spatialReflectPad ? 1 : 0)}";

    public Tensor Forward(IBackend backend, Tensor x)
    {
        int b = (int)x.Shape[0], f = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        int st0 = _stride.T, st1 = _stride.H, st2 = _stride.W;
        int stProd = st0 * st1 * st2;
        int outC = (_inC * stProd) / _upscale;            // conv output channels
        int xOut = outC / stProd;                          // final channels (= inC / upscale)

        Tensor conv = _conv!.Forward(backend, x);          // [B, outC, F, H, W]
        Tensor main = b == 1
            ? PixelShuffleDevice(backend, conv, xOut, f, h, w, st0, st1, st2)
            : PixelShuffle(conv, b, outC, xOut, f, h, w, st0, st1, st2);
        conv.Dispose();

        if (!_residual) return main;

        // Residual: pixel-shuffle the INPUT (Y = inC/stProd channels), then repeat to xOut channels.
        int y = _inC / stProd;
        int repeats = stProd / _upscale;                   // y*repeats == xOut
        if (b == 1)
        {
            Tensor resShufDev = PixelShuffleDevice(backend, x, y, f, h, w, st0, st1, st2);
            Tensor residualDev = RepeatChannelsDevice(backend, resShufDev, repeats);
            resShufDev.Dispose();
            backend.Add(main, main, residualDev);
            residualDev.Dispose();
            return main;
        }
        Tensor resShuf = PixelShuffle(x, b, _inC, y, f, h, w, st0, st1, st2);   // [B, y, F*st0-(st0-1), H*st1, W*st2]
        Tensor residual = RepeatChannels(resShuf, repeats);
        resShuf.Dispose();

        long n = main.Shape.ElementCount;
        float* mp = (float*)main.DataPointer;
        float* rp = (float*)residual.DataPointer;
        for (long i = 0; i < n; i++) mp[i] += rp[i];
        residual.Dispose();
        return main;
    }

    /// <summary>B=1 device pixel-shuffle: decomposes the channel→(t,h,w) scatter <c>[x·s0·s1·s2, F, H, W] → [1, x, F·st0−(st0−1), H·st1, W·st2]</c> into batched <see cref="IBackend.Permute0213"/> adjacent-group swaps plus one <see cref="IBackend.SliceRows"/> for the leading temporal-pad drop, so the tensor never leaves the GPU (the host loop D2H-drained the conv output and re-uploaded for the next stage). Identity swaps (stride-1 axes) are skipped.</summary>
    private static Tensor PixelShuffleDevice(IBackend backend, Tensor src, int xOut, int f, int h, int w, int st0, int st1, int st2)
    {
        int outF = f * st0 - (st0 - 1);
        int outH = h * st1, outW = w * st2;
        int hw = h * w;
        int sg = st1 * st2;                                // spatial sub-channel group (s1·s2)
        int rest = sg * hw;                                // elements right of the fused temporal axes
        TensorShape finalShape = new TensorShape([1L, xOut, outF, outH, outW]);
        Tensor cur = src;
        bool owned = false;

        // (a) [x, s0, (s1·s2), F, (H·W)] → [x, s0, F, (s1·s2), (H·W)]
        if (sg > 1)
        {
            cur = PermuteStep(backend, cur, owned, sg, f, hw, new TensorShape([(long)xOut * st0, f, sg, hw]));
            owned = true;
        }
        if (st0 > 1)
        {
            // (b) [x, s0, F, rest] → [x, F, s0, rest]; (c) [x, (F·s0), rest] → [(F·s0), x, rest]
            cur = PermuteStep(backend, cur, owned, st0, f, rest, new TensorShape([(long)xOut, f, st0, rest]));
            cur = PermuteStep(backend, cur, true, xOut, f * st0, rest, new TensorShape([(long)f * st0, xOut, rest]));
            owned = true;
            // (d) drop the leading (st0−1) temporal-pad frames of the fused (F·s0) axis
            Tensor sliced = new Tensor(new TensorShape([(long)outF, xOut, (long)sg * h, w]), cur.DType);
            backend.SliceRows(sliced, cur, (st0 - 1) * xOut * sg * h);
            cur.Dispose();
            cur = sliced;
        }
        // (e1) [.., (s1·s2), H, W] → [.., H, (s1·s2), W]; (e2) [.., s2, W] → [.., W, s2]; (e3) [F', x, ..] → [x, F', ..]
        if (sg > 1)
        {
            bool isLast = st2 == 1 && st0 == 1;
            cur = PermuteStep(backend, cur, owned, sg, h, w,
                isLast ? finalShape : new TensorShape([(long)outF * xOut, h, sg, w]));
            owned = true;
        }
        if (st2 > 1)
        {
            bool isLast = st0 == 1;
            cur = PermuteStep(backend, cur, owned, st2, w, 1,
                isLast ? finalShape : new TensorShape([(long)outF * xOut, (long)h * st1, w, st2]));
            owned = true;
        }
        if (st0 > 1)
        {
            cur = PermuteStep(backend, cur, true, outF, xOut, outH * outW, finalShape);
            owned = true;
        }
        if (!owned)
        {
            // Identity stride (1,1,1): flat device copy so the caller can dispose uniformly.
            Tensor copyT = new Tensor(finalShape, cur.DType);
            backend.SliceRows(copyT, cur, 0);
            cur = copyT;
        }
        return cur;
    }

    /// <summary>Allocates <paramref name="outShape"/> and runs one <c>[B, s, h, d] → [B, h, s, d]</c> group swap, disposing the previous intermediate when owned.</summary>
    private static Tensor PermuteStep(IBackend backend, Tensor cur, bool ownCur, int s, int h, int d, TensorShape outShape)
    {
        Tensor next = new Tensor(outShape, cur.DType);
        backend.Permute0213(next, cur, s, h, d);
        if (ownCur) cur.Dispose();
        return next;
    }

    /// <summary>Device channel repeat (<c>dst[r·C+c] = src[c]</c>) via <see cref="IBackend.Concat"/> along dim 1 — the exact block layout of the host <see cref="RepeatChannels"/>.</summary>
    private static Tensor RepeatChannelsDevice(IBackend backend, Tensor x, int repeats)
    {
        if (repeats == 1)
        {
            Tensor copyT = new Tensor(x.Shape, x.DType);
            backend.SliceRows(copyT, x, 0);
            return copyT;
        }
        long c = x.Shape[1];
        Tensor outT = new Tensor(new TensorShape([x.Shape[0], c * repeats, x.Shape[2], x.Shape[3], x.Shape[4]]), x.DType);
        Tensor[] reps = new Tensor[repeats];
        Array.Fill(reps, x);
        backend.Concat(outT, reps, dim: 1);
        return outT;
    }

    /// <summary>Pixel-shuffle a <c>[B, srcC, F, H, W]</c> tensor (srcC = xOut·stProd) to <c>[B, xOut, F·st0−(st0−1), H·st1, W·st2]</c> per the upstream reshape/permute (then the leading <c>st0−1</c> temporal frames are dropped).</summary>
    private static Tensor PixelShuffle(Tensor src, int b, int srcC, int xOut, int f, int h, int w, int st0, int st1, int st2)
    {
        int outF = f * st0 - (st0 - 1);
        int outH = h * st1, outW = w * st2;
        Tensor outT = new Tensor(new TensorShape([(long)b, xOut, outF, outH, outW]), DType.F32);
        float* sp = (float*)src.DataPointer;
        float* op = (float*)outT.DataPointer;
        long srcFrame = (long)h * w;

        for (int bi = 0; bi < b; bi++)
            for (int x = 0; x < xOut; x++)
                for (int fo = 0; fo < outF; fo++)
                {
                    int full = fo + (st0 - 1);
                    int fi = full / st0, s0 = full % st0;
                    for (int ho = 0; ho < outH; ho++)
                    {
                        int hi = ho / st1, s1 = ho % st1;
                        for (int wo = 0; wo < outW; wo++)
                        {
                            int wi = wo / st2, s2 = wo % st2;
                            int ch = ((x * st0 + s0) * st1 + s1) * st2 + s2;
                            long srcOff = (((long)bi * srcC + ch) * f + fi) * srcFrame + (long)hi * w + wi;
                            long dstOff = (((long)bi * xOut + x) * outF + fo) * ((long)outH * outW) + (long)ho * outW + wo;
                            op[dstOff] = sp[srcOff];
                        }
                    }
                }
        return outT;
    }

    private static Tensor RepeatChannels(Tensor x, int repeats)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], f = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        Tensor outT = new Tensor(new TensorShape([(long)b, c * repeats, f, h, w]), DType.F32);
        float* sp = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        long chunk = (long)f * h * w;
        for (int bi = 0; bi < b; bi++)
            for (int r = 0; r < repeats; r++)
                for (int ci = 0; ci < c; ci++)
                {
                    long srcOff = ((long)bi * c + ci) * chunk;
                    long dstOff = ((long)bi * c * repeats + r * c + ci) * chunk;
                    Buffer.MemoryCopy(sp + srcOff, op + dstOff, chunk * 4, chunk * 4);
                }
        return outT;
    }

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string k) => w.TryGetValue(k, out Tensor? b) ? b : null;
}
