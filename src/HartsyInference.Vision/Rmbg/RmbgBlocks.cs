using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Rmbg;

/// <summary>REBNCONV = Conv2d(3×3, dilation d) → BatchNorm → ReLU. At load the BatchNorm is <b>folded into the
/// conv</b> (inference BN is a per-channel affine) and the 3×3 kernel is <b>zero-inflated to (2d+1)×(2d+1)</b> so a
/// dilated conv runs as a plain <see cref="IBackend.Conv2D"/> with pad=d. Forward is one Conv2D + ReLU.</summary>
internal sealed unsafe class RebnConv
{
    private readonly int _dirate;
    private Tensor? _w, _b;   // folded + inflated conv weight [out,in,2d+1,2d+1], folded bias [out]

    public RebnConv(int dirate) => _dirate = dirate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        Tensor cw = BriaRmbg.F32(w[$"{p}.conv_s1.weight"]);   // [out,in,3,3]
        Tensor cb = BriaRmbg.F32(w[$"{p}.conv_s1.bias"]);     // [out]
        Tensor g = BriaRmbg.F32(w[$"{p}.bn_s1.weight"]), be = BriaRmbg.F32(w[$"{p}.bn_s1.bias"]);
        Tensor mean = BriaRmbg.F32(w[$"{p}.bn_s1.running_mean"]), var = BriaRmbg.F32(w[$"{p}.bn_s1.running_var"]);
        (_w, _b) = FoldAndInflate(cw, cb, g, be, mean, var, _dirate);
    }

    public IEnumerable<Tensor> EnumerateWeights() { if (_w is not null) yield return _w; if (_b is not null) yield return _b; }

    public Tensor Forward(IBackend backend, Tensor x) => RmbgOps.Conv(backend, x, _w!, _b!, stride: 1, pad: _dirate, relu: true);

    /// <summary>Folds inference BatchNorm into the conv (<c>w' = w·γ/√(σ²+ε)</c>, <c>b' = (b−μ)·γ/√(σ²+ε)+β</c>,
    /// ε=1e-5) and zero-inflates the 3×3 kernel to (2d+1)² with the 9 taps at stride-d positions.</summary>
    private static (Tensor, Tensor) FoldAndInflate(Tensor cw, Tensor cb, Tensor g, Tensor be, Tensor mean, Tensor var, int d)
    {
        int outc = (int)cw.Shape[0], inc = (int)cw.Shape[1];
        float* wp = (float*)cw.DataPointer, bp = (float*)cb.DataPointer;
        float* gp = (float*)g.DataPointer, bep = (float*)be.DataPointer, mp = (float*)mean.DataPointer, vp = (float*)var.DataPointer;
        int k = 2 * d + 1;
        Tensor fw = new(new TensorShape(outc, inc, k, k), DType.F32);   // zero-initialized
        Tensor fb = new(new TensorShape(outc), DType.F32);
        float* fwp = (float*)fw.DataPointer, fbp = (float*)fb.DataPointer;
        for (int o = 0; o < outc; o++)
        {
            float scale = gp[o] / MathF.Sqrt(vp[o] + 1e-5f);
            fbp[o] = (bp[o] - mp[o]) * scale + bep[o];
            for (int i = 0; i < inc; i++)
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                    {
                        float wv = wp[((((long)o * inc + i) * 3 + r) * 3 + c)] * scale;
                        fwp[((((long)o * inc + i) * k + r * d) * k + c * d)] = wv;
                    }
        }
        return (fw, fb);
    }
}

/// <summary>ReSidual U-block. Height-<c>L</c> pooled variant (RSU-7/6/5/4): an inner U-Net with <c>L-1</c> encoder
/// convs + <c>L-2</c> /2 pools, a dilated-2 bottleneck, and <c>L-1</c> decoder convs (upsample + concat skip);
/// or the <b>dilated</b> RSU-4F (no pooling, dilations 1/2/4/8) when <c>dilated</c>. Returns <c>rsu(x) + rebnconvin(x)</c>.</summary>
internal sealed unsafe class RsuBlock
{
    private readonly int _l;
    private readonly bool _dilated;
    private readonly RebnConv _in;
    private readonly RebnConv[] _enc;    // rebnconv1..(L-1)  (dilated: 1..4)
    private readonly RebnConv _mid;      // rebnconvL bottleneck (pooled only)
    private readonly RebnConv[] _dec;    // rebnconv(L-1)d..1d

    public RsuBlock(int l, int inCh, int midCh, int outCh, bool dilated = false)
    {
        _l = l; _dilated = dilated;
        _in = new RebnConv(1);
        if (dilated)
        {
            _enc = [new RebnConv(1), new RebnConv(2), new RebnConv(4), new RebnConv(8)];   // rebnconv1..4
            _dec = [new RebnConv(4), new RebnConv(2), new RebnConv(1)];                    // rebnconv3d,2d,1d
            _mid = null!;
        }
        else
        {
            _enc = new RebnConv[l - 1]; for (int i = 0; i < l - 1; i++) _enc[i] = new RebnConv(1);   // rebnconv1..(L-1)
            _mid = new RebnConv(2);                                                                   // rebnconvL
            _dec = new RebnConv[l - 1]; for (int i = 0; i < l - 1; i++) _dec[i] = new RebnConv(1);     // rebnconv(L-1)d..1d
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _in.LoadWeights(w, $"{p}.rebnconvin");
        if (_dilated)
        {
            for (int i = 0; i < 4; i++) _enc[i].LoadWeights(w, $"{p}.rebnconv{i + 1}");
            _dec[0].LoadWeights(w, $"{p}.rebnconv3d"); _dec[1].LoadWeights(w, $"{p}.rebnconv2d"); _dec[2].LoadWeights(w, $"{p}.rebnconv1d");
        }
        else
        {
            for (int i = 0; i < _l - 1; i++) _enc[i].LoadWeights(w, $"{p}.rebnconv{i + 1}");
            _mid.LoadWeights(w, $"{p}.rebnconv{_l}");
            for (int i = 0; i < _l - 1; i++) _dec[i].LoadWeights(w, $"{p}.rebnconv{_l - 1 - i}d");   // (L-1)d, (L-2)d, .., 1d
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _in.EnumerateWeights()) yield return t;
        foreach (RebnConv c in _enc) foreach (Tensor t in c.EnumerateWeights()) yield return t;
        if (!_dilated) foreach (Tensor t in _mid.EnumerateWeights()) yield return t;
        foreach (RebnConv c in _dec) foreach (Tensor t in c.EnumerateWeights()) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor x) => _dilated ? ForwardDilated(backend, x) : ForwardPooled(backend, x);

    private Tensor ForwardPooled(IBackend backend, Tensor x)
    {
        Tensor hxin = _in.Forward(backend, x);
        // Encoder: keep each stage's output as a skip; pool between all but the last.
        Tensor[] skips = new Tensor[_l - 1];
        Tensor hx = hxin;
        for (int i = 0; i < _l - 1; i++)
        {
            Tensor e = _enc[i].Forward(backend, hx);
            if (i > 0) hx.Dispose();       // hx was a pooled temp (i>0); i==0 hx==hxin (keep for residual)
            skips[i] = e;
            if (i < _l - 2) hx = RmbgOps.Pool2(backend, e); else hx = e;   // no pool after the last encoder conv
        }
        Tensor bott = _mid.Forward(backend, skips[_l - 2]);   // bottleneck on the last (unpooled) encoder output
        // Decoder: hx(L-1)d = dec0(cat(bott, skip[L-2])); then up+cat down to 1d.
        Tensor cur = RmbgOps.CatRun(backend, _dec[0], bott, skips[_l - 2], upsample: false); bott.Dispose();
        for (int i = 1; i < _l - 1; i++)
        {
            Tensor next = RmbgOps.CatRun(backend, _dec[i], cur, skips[_l - 2 - i], upsample: true);
            cur.Dispose(); cur = next;
        }
        for (int i = 0; i < _l - 1; i++) skips[i].Dispose();
        Tensor outp = new(cur.Shape, DType.F32); backend.Add(outp, cur, hxin); cur.Dispose(); hxin.Dispose();
        return outp;
    }

    private Tensor ForwardDilated(IBackend backend, Tensor x)
    {
        Tensor hxin = _in.Forward(backend, x);
        Tensor h1 = _enc[0].Forward(backend, hxin);
        Tensor h2 = _enc[1].Forward(backend, h1);
        Tensor h3 = _enc[2].Forward(backend, h2);
        Tensor h4 = _enc[3].Forward(backend, h3);
        Tensor h3d = RmbgOps.CatRun(backend, _dec[0], h4, h3, upsample: false); h4.Dispose(); h3.Dispose();
        Tensor h2d = RmbgOps.CatRun(backend, _dec[1], h3d, h2, upsample: false); h3d.Dispose(); h2.Dispose();
        Tensor h1d = RmbgOps.CatRun(backend, _dec[2], h2d, h1, upsample: false); h2d.Dispose(); h1.Dispose();
        Tensor outp = new(h1d.Shape, DType.F32); backend.Add(outp, h1d, hxin); h1d.Dispose(); hxin.Dispose();
        return outp;
    }
}

/// <summary>Shared conv/pool helpers for RMBG.</summary>
internal static unsafe class RmbgOps
{
    public static Tensor Conv(IBackend backend, Tensor x, Tensor w, Tensor b, int stride, int pad, bool relu = false)
    {
        int outc = (int)w.Shape[0], inH = (int)x.Shape[2], inW = (int)x.Shape[3], k = (int)w.Shape[2];
        int outH = (inH + 2 * pad - k) / stride + 1, outW = (inW + 2 * pad - k) / stride + 1;
        Tensor o = new(new TensorShape(1, outc, outH, outW), DType.F32);
        backend.Conv2D(o, x, w, b, stride, stride, pad, pad);
        if (relu) { Tensor r = new(o.Shape, DType.F32); backend.LeakyRelu(r, o, 0f); o.Dispose(); return r; }
        return o;
    }

    /// <summary>2×2 max-pool, stride 2 (spatial /2 for even sizes).</summary>
    public static Tensor Pool2(IBackend backend, Tensor x)
    {
        Tensor o = new(new TensorShape(1, (int)x.Shape[1], (int)x.Shape[2] / 2, (int)x.Shape[3] / 2), DType.F32);
        backend.MaxPool2D(o, x, 2, 2, 2, 2, 0, 0);
        return o;
    }

    /// <summary>Optionally 2× bilinear-upsamples <paramref name="prev"/> to <paramref name="skip"/>'s size, concatenates
    /// on channels, and runs <paramref name="conv"/> (a decoder REBNCONV).</summary>
    public static Tensor CatRun(IBackend backend, RebnConv conv, Tensor prev, Tensor skip, bool upsample)
    {
        int c = (int)prev.Shape[1], sh = (int)skip.Shape[2], sw = (int)skip.Shape[3];
        Tensor up = upsample ? Upsample2x(prev) : prev;
        Tensor cat = new(new TensorShape(1, c + (int)skip.Shape[1], sh, sw), DType.F32);
        backend.Concat(cat, [up, skip], 1);
        if (upsample) up.Dispose();
        Tensor outp = conv.Forward(backend, cat); cat.Dispose();
        return outp;
    }

    /// <summary>Host 2× bilinear upsample (align_corners=False), matching PyTorch <c>F.interpolate(scale=2)</c>. CUDA
    /// has no bilinear-upsample kernel; RMBG runs once per image so the host path (with a D2H sync) is acceptable.</summary>
    public static Tensor Upsample2x(Tensor x)
    {
        int cch = (int)x.Shape[1], h = (int)x.Shape[2], w = (int)x.Shape[3], oh = h * 2, ow = w * 2;
        Tensor o = new(new TensorShape(1, cch, oh, ow), DType.F32);
        float* ip = (float*)x.DataPointer; float* op = (float*)o.DataPointer;
        for (int c = 0; c < cch; c++)
        {
            float* ic = ip + (long)c * h * w; float* oc = op + (long)c * oh * ow;
            for (int oy = 0; oy < oh; oy++)
            {
                float sy = (oy + 0.5f) * 0.5f - 0.5f;
                int y0 = (int)MathF.Floor(sy); float fy = sy - y0;
                int y0c = Math.Clamp(y0, 0, h - 1), y1c = Math.Clamp(y0 + 1, 0, h - 1);
                for (int ox = 0; ox < ow; ox++)
                {
                    float sx = (ox + 0.5f) * 0.5f - 0.5f;
                    int x0 = (int)MathF.Floor(sx); float fx = sx - x0;
                    int x0c = Math.Clamp(x0, 0, w - 1), x1c = Math.Clamp(x0 + 1, 0, w - 1);
                    float a = ic[y0c * w + x0c], b2 = ic[y0c * w + x1c], cc = ic[y1c * w + x0c], d = ic[y1c * w + x1c];
                    oc[oy * ow + ox] = (a * (1 - fx) + b2 * fx) * (1 - fy) + (cc * (1 - fx) + d * fx) * fy;
                }
            }
        }
        return o;
    }
}
