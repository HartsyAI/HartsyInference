using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Wan-Video 3D rotary position embedding (<c>WanRotaryPosEmbed</c> + the processor's <c>apply_rotary_emb</c> in diffusers <c>transformer_wan.py</c>). Per-head (head_dim) interleaved-pair RoPE: the head dim splits into <c>t_dim / h_dim / w_dim</c> (<c>h=w=2·(head_dim//6)</c>, <c>t=head_dim−h−w</c>), each axis getting standard <c>get_1d_rotary_pos_embed</c> frequencies (<c>θ^(−2i/dim)</c>, repeat-interleaved). Distinct from <see cref="LtxRope"/> (full-dim) and <see cref="Multimodal3DRope"/> (block form).</summary>
public sealed unsafe class WanRope
{
    private readonly int _headDim;
    private readonly int _tDim, _hDim, _wDim;
    private readonly double[] _freqT, _freqH, _freqW;   // per-axis angular frequencies (length dim/2)
    private readonly int _numHeads;                     // > 0 only in per-head (sigma_theta) mode
    private readonly double[][]? _phFreqT, _phFreqH, _phFreqW;   // [head][dim/2] per-head freqs (sigma_theta mode)

    /// <param name="sigmaTheta">Matrix-Game 3.0 per-head θ spread (<c>config.json sigma_theta</c>). When non-zero (and
    /// <paramref name="numHeads"/> &gt; 0) each head <c>h</c> gets <c>θ_h = theta·(1 + sigmaTheta·ε_h)</c> with
    /// <c>ε_h = linspace(-1, 1, numHeads)</c>, and <see cref="BuildCosSin(ReadOnlySpan{int}, int, int)"/> emits per-head
    /// tables <c>[numHeads, S, headDim]</c>. Zero = a single shared θ (standard Wan behavior; byte-identical path).</param>
    public WanRope(int headDim, double theta, int maxSeqLen, double sigmaTheta = 0.0, int numHeads = 0)
    {
        _headDim = headDim;
        _hDim = _wDim = 2 * (headDim / 6);
        _tDim = headDim - _hDim - _wDim;
        _freqT = AxisFreqs(_tDim, theta);
        _freqH = AxisFreqs(_hDim, theta);
        _freqW = AxisFreqs(_wDim, theta);
        _ = maxSeqLen;   // freqs are computed on demand; max length is informational
        if (sigmaTheta != 0.0 && numHeads > 0)
        {
            _numHeads = numHeads;
            _phFreqT = new double[numHeads][];
            _phFreqH = new double[numHeads][];
            _phFreqW = new double[numHeads][];
            for (int h = 0; h < numHeads; h++)
            {
                double eps = numHeads == 1 ? -1.0 : -1.0 + 2.0 * h / (numHeads - 1);   // linspace(-1, 1, numHeads)
                double thetaH = theta * (1.0 + sigmaTheta * eps);
                _phFreqT[h] = AxisFreqs(_tDim, thetaH);
                _phFreqH[h] = AxisFreqs(_hDim, thetaH);
                _phFreqW[h] = AxisFreqs(_wDim, thetaH);
            }
        }
    }

    /// <summary>Explicit per-axis dim split (t, h, w) — Matrix-Game's ActionModule uses <c>[8, 28, 28]</c> at θ=256
    /// over a 64-wide head, instead of the main DiT's derived split.</summary>
    public WanRope(int tDim, int hDim, int wDim, double theta)
    {
        _headDim = tDim + hDim + wDim;
        _tDim = tDim; _hDim = hDim; _wDim = wDim;
        _freqT = AxisFreqs(_tDim, theta);
        _freqH = AxisFreqs(_hDim, theta);
        _freqW = AxisFreqs(_wDim, theta);
    }

    /// <summary>Head dim this rope rotates.</summary>
    public int HeadDim => _headDim;

    private static double[] AxisFreqs(int dim, double theta)
    {
        int n = dim / 2;
        double[] f = new double[n];
        for (int i = 0; i < n; i++) f[i] = 1.0 / Math.Pow(theta, (2.0 * i) / dim);
        return f;
    }

    /// <summary>Builds cos/sin <c>[S, headDim]</c> for a token grid <c>gridT × gridH × gridW</c> in <c>(f,h,w)</c> order. Each value is duplicated across its interleaved pair (so the pair shares one angle).</summary>
    public (Tensor Cos, Tensor Sin) BuildCosSin(int gridT, int gridH, int gridW)
    {
        int[] frames = new int[gridT];
        for (int i = 0; i < gridT; i++) frames[i] = i;
        return BuildCosSin(frames, gridH, gridW);
    }

    /// <summary>Builds cos/sin <c>[S, headDim]</c> with an explicit temporal index per frame slot — Matrix-Game's
    /// memory-augmented sequence assigns retrieved memory slots their <b>original historical</b> t-indices
    /// (non-contiguous gaps are fine; the model is trained for them).</summary>
    public (Tensor Cos, Tensor Sin) BuildCosSin(ReadOnlySpan<int> frameIndices, int gridH, int gridW)
    {
        int gridT = frameIndices.Length;
        int seq = gridT * gridH * gridW;
        if (_phFreqT is not null) return BuildCosSinPerHead(frameIndices, gridH, gridW, gridT, seq);

        Tensor cos = new Tensor(new TensorShape(seq, _headDim), DType.F32);
        Tensor sin = new Tensor(new TensorShape(seq, _headDim), DType.F32);
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;

        for (int fi = 0; fi < gridT; fi++)
            for (int hi = 0; hi < gridH; hi++)
                for (int wi = 0; wi < gridW; wi++)
                {
                    long token = ((long)fi * gridH + hi) * gridW + wi;
                    long off = token * _headDim;
                    int d = 0;
                    FillAxis(cp, sp, off, ref d, _freqT, frameIndices[fi]);
                    FillAxis(cp, sp, off, ref d, _freqH, hi);
                    FillAxis(cp, sp, off, ref d, _freqW, wi);
                }
        return (cos, sin);
    }

    /// <summary>Per-head (sigma_theta) cos/sin <c>[numHeads, S, headDim]</c>: token order matches the shared path, but
    /// each head uses its own θ_h frequencies. Consumed by <see cref="ApplyRotary"/>'s rank-3 branch.</summary>
    private (Tensor Cos, Tensor Sin) BuildCosSinPerHead(ReadOnlySpan<int> frameIndices, int gridH, int gridW, int gridT, int seq)
    {
        Tensor cos = new Tensor(new TensorShape(_numHeads, seq, _headDim), DType.F32);
        Tensor sin = new Tensor(new TensorShape(_numHeads, seq, _headDim), DType.F32);
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;
        int[] frames = frameIndices.ToArray();
        for (int h = 0; h < _numHeads; h++)
        {
            long headBase = (long)h * seq * _headDim;
            for (int fi = 0; fi < gridT; fi++)
                for (int hi = 0; hi < gridH; hi++)
                    for (int wi = 0; wi < gridW; wi++)
                    {
                        long token = ((long)fi * gridH + hi) * gridW + wi;
                        long off = headBase + token * _headDim;
                        int d = 0;
                        FillAxis(cp, sp, off, ref d, _phFreqT![h], frames[fi]);
                        FillAxis(cp, sp, off, ref d, _phFreqH![h], hi);
                        FillAxis(cp, sp, off, ref d, _phFreqW![h], wi);
                    }
        }
        return (cos, sin);
    }

    private static void FillAxis(float* cp, float* sp, long off, ref int d, double[] freqs, int pos)
    {
        foreach (double f in freqs)
        {
            double angle = pos * f;
            float c = (float)Math.Cos(angle), s = (float)Math.Sin(angle);
            cp[off + d] = c; sp[off + d] = s;
            cp[off + d + 1] = c; sp[off + d + 1] = s;
            d += 2;
        }
    }

    /// <summary>Applies the per-head interleaved-pair rotation in-place to <paramref name="x"/> <c>[S, heads, headDim]</c>.
    /// cos/sin are either <c>[S, headDim]</c> (shared across heads, standard Wan) or <c>[heads, S, headDim]</c>
    /// (per-head sigma_theta mode); the layout is detected from the cos rank.</summary>
    public void ApplyRotary(Tensor x, Tensor cos, Tensor sin, int heads)
    {
        int seq = (int)x.Shape[0];
        float* xp = (float*)x.DataPointer;
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;
        int pairs = _headDim / 2;
        bool perHead = cos.Shape.Rank == 3;
        for (int s = 0; s < seq; s++)
        {
            long sharedCosOff = (long)s * _headDim;
            for (int h = 0; h < heads; h++)
            {
                long cosOff = perHead ? ((long)h * seq + s) * _headDim : sharedCosOff;
                long xoff = ((long)s * heads + h) * _headDim;
                for (int i = 0; i < pairs; i++)
                {
                    int i0 = 2 * i;
                    float re = xp[xoff + i0], im = xp[xoff + i0 + 1];
                    float c = cp[cosOff + i0], sn = sp[cosOff + i0];
                    xp[xoff + i0] = re * c - im * sn;
                    xp[xoff + i0 + 1] = re * sn + im * c;
                }
            }
        }
    }
}
