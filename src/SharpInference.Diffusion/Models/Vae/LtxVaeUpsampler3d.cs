using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Vae;

/// <summary>LTX-Video VAE 3D upsampler (<c>LTXVideoUpsampler3d</c>), ported from diffusers. A causal conv followed by a pixel-shuffle that trades channels for spatiotemporal resolution: <c>[B, C, F, H, W] → [B, C/upscale, F·st0−(st0−1), H·st1, W·st2]</c>. With <c>residual=True</c> the input is pixel-shuffled + channel-repeated and added. Reuses <see cref="CausalConv3d"/> (replicate-first-frame padding).</summary>
public sealed unsafe class LtxVaeUpsampler3d
{
    private readonly int _inC;
    private readonly (int T, int H, int W) _stride;
    private readonly int _upscale;
    private readonly bool _residual;
    private readonly bool _isCausal;
    private CausalConv3d? _conv;

    public LtxVaeUpsampler3d(int inC, (int T, int H, int W) stride, int upscaleFactor, bool residual, bool isCausal = true)
    {
        _inC = inC;
        _stride = stride;
        _upscale = upscaleFactor;
        _residual = residual;
        _isCausal = isCausal;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _conv = new CausalConv3d(w[$"{prefix}.conv.conv.weight"], Bias(w, $"{prefix}.conv.conv.bias"),
            padT: 1, padH: 1, padW: 1, replicateFirstPad: true, causal: _isCausal);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_conv is not null) foreach (Tensor t in _conv.EnumerateWeights()) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        int b = (int)x.Shape[0], f = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        int st0 = _stride.T, st1 = _stride.H, st2 = _stride.W;
        int stProd = st0 * st1 * st2;
        int outC = (_inC * stProd) / _upscale;            // conv output channels
        int xOut = outC / stProd;                          // final channels (= inC / upscale)

        Tensor conv = _conv!.Forward(backend, x);          // [B, outC, F, H, W]
        Tensor main = PixelShuffle(conv, b, outC, xOut, f, h, w, st0, st1, st2);
        conv.Dispose();

        if (!_residual) return main;

        // Residual: pixel-shuffle the INPUT (Y = inC/stProd channels), then repeat to xOut channels.
        int y = _inC / stProd;
        Tensor resShuf = PixelShuffle(x, b, _inC, y, f, h, w, st0, st1, st2);   // [B, y, F*st0-(st0-1), H*st1, W*st2]
        int repeats = stProd / _upscale;                   // y*repeats == xOut
        Tensor residual = RepeatChannels(resShuf, repeats);
        resShuf.Dispose();

        long n = main.Shape.ElementCount;
        float* mp = (float*)main.DataPointer;
        float* rp = (float*)residual.DataPointer;
        for (long i = 0; i < n; i++) mp[i] += rp[i];
        residual.Dispose();
        return main;
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
