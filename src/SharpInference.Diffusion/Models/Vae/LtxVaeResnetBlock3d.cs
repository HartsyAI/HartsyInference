using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Vae;

/// <summary>LTX-Video VAE 3D ResNet block (<c>LTXVideoResnetBlock3d</c>), ported from diffusers. Channel-RMS norm (no affine, eps 1e-8) → optional AdaLN timestep conditioning → SiLU → causal conv → (×2) → residual. The decoder variant is **timestep-conditioned**: a per-block <c>scale_shift_table[4, C]</c> is added to a decode-timestep embedding to produce <c>(shift1, scale1, shift2, scale2)</c>. Reuses <see cref="CausalConv3d"/> (replicate-first-frame padding). Inject-noise (`per_channel_scale`) is omitted in v1 (small stochastic perturbation; documented).</summary>
public sealed unsafe class LtxVaeResnetBlock3d
{
    private const float NormEps = 1e-8f;

    private readonly int _inC;
    private readonly int _outC;
    private readonly bool _timestepCond;
    private readonly bool _isCausal;

    private CausalConv3d? _conv1, _conv2, _convShortcut;
    private Tensor? _norm3W, _norm3B;          // LayerNorm affine (shortcut), only when in != out
    private Tensor? _scaleShift;               // [4, C], only when timestep-conditioned

    public LtxVaeResnetBlock3d(int inC, int outC, bool timestepCond, bool isCausal = true)
    {
        _inC = inC;
        _outC = outC;
        _timestepCond = timestepCond;
        _isCausal = isCausal;
    }

    public int OutChannels => _outC;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _conv1 = new CausalConv3d(w[$"{prefix}.conv1.conv.weight"], Bias(w, $"{prefix}.conv1.conv.bias"),
            padT: 1, padH: 1, padW: 1, replicateFirstPad: true, causal: _isCausal);
        _conv2 = new CausalConv3d(w[$"{prefix}.conv2.conv.weight"], Bias(w, $"{prefix}.conv2.conv.bias"),
            padT: 1, padH: 1, padW: 1, replicateFirstPad: true, causal: _isCausal);
        if (_inC != _outC)
        {
            _norm3W = LoadF32(w, $"{prefix}.norm3.weight");
            w.TryGetValue($"{prefix}.norm3.bias", out _norm3B);
            _convShortcut = new CausalConv3d(w[$"{prefix}.conv_shortcut.conv.weight"], Bias(w, $"{prefix}.conv_shortcut.conv.bias"),
                padT: 0, padH: 0, padW: 0, replicateFirstPad: true, causal: _isCausal);
        }
        if (_timestepCond) _scaleShift = LoadF32(w, $"{prefix}.scale_shift_table");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_conv1 is not null) foreach (Tensor t in _conv1.EnumerateWeights()) yield return t;
        if (_conv2 is not null) foreach (Tensor t in _conv2.EnumerateWeights()) yield return t;
        if (_convShortcut is not null) foreach (Tensor t in _convShortcut.EnumerateWeights()) yield return t;
        if (_norm3W is not null) yield return _norm3W;
        if (_norm3B is not null) yield return _norm3B;
        if (_scaleShift is not null) yield return _scaleShift;
    }

    /// <summary>Forward over <c>[B, inC, T, H, W]</c>. <paramref name="temb"/> (<c>[4·C]</c>, B=1) is required iff the block is timestep-conditioned.</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor? temb)
    {
        // Compute the 4 modulation vectors once (each [C]) when timestep-conditioned.
        float[]? shift1 = null, scale1 = null, shift2 = null, scale2 = null;
        if (_timestepCond)
        {
            (shift1, scale1, shift2, scale2) = Modulation(temb!);
        }

        Tensor h = ChannelRms(x, _inC);
        if (_timestepCond) ApplyShiftScale(h, scale1!, shift1!);
        Silu(backend, h);
        Tensor c1 = _conv1!.Forward(backend, h);
        h.Dispose();

        Tensor h2 = ChannelRms(c1, _outC);
        c1.Dispose();
        if (_timestepCond) ApplyShiftScale(h2, scale2!, shift2!);
        Silu(backend, h2);
        Tensor c2 = _conv2!.Forward(backend, h2);
        h2.Dispose();

        // Residual (with optional norm3 + shortcut conv).
        Tensor residual;
        if (_convShortcut is not null)
        {
            Tensor n3 = LayerNormChannel(x, _inC, _norm3W!, _norm3B);
            residual = _convShortcut.Forward(backend, n3);
            n3.Dispose();
        }
        else
        {
            residual = x;
        }

        Tensor outT = new Tensor(c2.Shape, DType.F32);
        backend.Add(outT, c2, residual);
        c2.Dispose();
        if (_convShortcut is not null) residual.Dispose();
        return outT;
    }

    private (float[], float[], float[], float[]) Modulation(Tensor temb)
    {
        // temb [4*C] reshaped to [4, C] + scale_shift_table[4, C].
        float* tp = (float*)temb.DataPointer;
        float* ss = (float*)_scaleShift!.DataPointer;
        float[][] o = new float[4][];
        for (int m = 0; m < 4; m++)
        {
            o[m] = new float[_inC];
            for (int c = 0; c < _inC; c++) o[m][c] = tp[m * _inC + c] + ss[m * _inC + c];
        }
        // Order matches upstream: shift_1, scale_1, shift_2, scale_2.
        return (o[0], o[1], o[2], o[3]);
    }

    /// <summary>Channel-wise RMS norm (no affine): per spatial-temporal position, <c>x / sqrt(mean_C(x²) + eps)</c>.</summary>
    private static Tensor ChannelRms(Tensor x, int c)
    {
        int b = (int)x.Shape[0];
        long spatial = x.Shape.ElementCount / ((long)b * c);
        Tensor outT = new Tensor(x.Shape, DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (long s = 0; s < spatial; s++)
            {
                double sum = 0;
                long basePos = (long)bi * c * spatial + s;
                for (int ci = 0; ci < c; ci++) { float v = xp[basePos + (long)ci * spatial]; sum += (double)v * v; }
                float inv = 1f / MathF.Sqrt((float)(sum / c) + NormEps);
                for (int ci = 0; ci < c; ci++) { long off = basePos + (long)ci * spatial; op[off] = xp[off] * inv; }
            }
        return outT;
    }

    /// <summary>Channel-wise LayerNorm (affine) for the shortcut norm3.</summary>
    private static Tensor LayerNormChannel(Tensor x, int c, Tensor weight, Tensor? bias)
    {
        int b = (int)x.Shape[0];
        long spatial = x.Shape.ElementCount / ((long)b * c);
        Tensor outT = new Tensor(x.Shape, DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        float* wp = (float*)weight.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (long s = 0; s < spatial; s++)
            {
                long basePos = (long)bi * c * spatial + s;
                double mean = 0;
                for (int ci = 0; ci < c; ci++) mean += xp[basePos + (long)ci * spatial];
                mean /= c;
                double var = 0;
                for (int ci = 0; ci < c; ci++) { double d = xp[basePos + (long)ci * spatial] - mean; var += d * d; }
                float inv = 1f / MathF.Sqrt((float)(var / c) + 1e-6f);
                for (int ci = 0; ci < c; ci++)
                {
                    long off = basePos + (long)ci * spatial;
                    float n = (float)((xp[off] - mean) * inv);
                    op[off] = n * wp[ci] + (bp is null ? 0f : bp[ci]);
                }
            }
        return outT;
    }

    private void ApplyShiftScale(Tensor x, float[] scale, float[] shift)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1];
        long spatial = x.Shape.ElementCount / ((long)b * c);
        float* xp = (float*)x.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
            {
                long basePos = ((long)bi * c + ci) * spatial;
                float sc = 1f + scale[ci], sh = shift[ci];
                for (long s = 0; s < spatial; s++) xp[basePos + s] = xp[basePos + s] * sc + sh;
            }
    }

    private static void Silu(IBackend backend, Tensor t) => backend.Silu(t, t);
    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string k) => w.TryGetValue(k, out Tensor? b) ? b : null;
    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string k) { Tensor t = w[k]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }
}
