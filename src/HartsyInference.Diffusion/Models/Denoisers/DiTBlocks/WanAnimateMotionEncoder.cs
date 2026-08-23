using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Wan-Animate motion encoder (<c>Generator</c> in ComfyUI <c>comfy/ldm/wan/model_animate.py</c>): a
/// StyleGAN2-style appearance encoder mapping face frames <c>[B, 3, 512, 512]</c> in [-1, 1] to motion vectors
/// <c>[B, 512]</c>. Structure: <c>enc.net_app.convs</c> = ConvLayer(3→32, k1) → 7 ResBlocks (each 2× spatial down,
/// 32→64→128→256→512→512→512→512) → EqualConv2d(512→512, k4, no bias) squeezed to <c>[B, 512]</c>; <c>enc.fc</c> =
/// 4×EqualLinear(512→512) + EqualLinear(512→20); <c>dec.direction</c> = "Linear Motion Decomposition":
/// <c>motion_vec = feat @ Qᵀ</c> where <c>Q</c> is the orthonormal basis of the learned <c>[512, 20]</c> weight
/// (+1e-8 on the diagonal), computed at inference via QR.
///
/// <para>Numerics ported exactly: EqualConv2d/EqualLinear runtime weight scale <c>1/√fan_in</c> (folded once at load;
/// <c>lr_mul</c> is 1 in this model), FusedLeakyReLU <c>leaky_relu(x+bias, 0.2)·√2</c>, Blur = depthwise FIR with the
/// normalized <c>[1,3,3,1]</c> outer-product kernel and asymmetric pad <c>((p+1)//2, p//2)</c>, ResBlock output
/// <c>/√2</c>. The QR is Householder with the LAPACK <c>geqrf</c> sign convention, matching torch.linalg.qr column
/// signs exactly (verified per-column against the real checkpoint: 12 of 20 columns flip vs positive-R Gram-Schmidt).</para></summary>
public sealed unsafe class WanAnimateMotionEncoder
{
    private EqualConv2d? _conv0;          // enc.net_app.convs.0 (ConvLayer k1 + FusedLeakyReLU)
    private ResBlock[] _resBlocks = [];   // enc.net_app.convs.1..N-2
    private EqualConv2d? _convFinal;      // enc.net_app.convs.N-1 (bare EqualConv2d k4, no bias, no act)
    private EqualLinear[] _fc = [];       // enc.fc.0..M-1
    private Tensor? _directionWeight;     // dec.direction.weight [styleDim, motionDim]

    /// <summary>Motion vector width (= style dim, 512 for the real checkpoint). Valid after <see cref="LoadWeights"/>.</summary>
    public int OutDim { get; private set; }

    /// <summary>Loads the <c>motion_encoder.*</c> subtree (original Wan-Animate names pass the converter unchanged).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _conv0 = EqualConv2d.Load(w, weightKey: $"{p}.enc.net_app.convs.0.0.weight", actBiasKey: $"{p}.enc.net_app.convs.0.1.bias",
            biasKey: null, stride: 1, padKind: EqualConv2d.PadSame, blurPad0: 0, blurPad1: 0);

        List<ResBlock> blocks = new();
        int idx = 1;
        while (w.ContainsKey($"{p}.enc.net_app.convs.{idx}.conv1.0.weight"))
        {
            ResBlock b = new();
            b.LoadWeights(w, $"{p}.enc.net_app.convs.{idx}");
            blocks.Add(b);
            idx++;
        }
        if (blocks.Count == 0)
            throw new ArgumentException($"motion encoder has no ResBlocks under {p}.enc.net_app.convs.", nameof(w));
        _resBlocks = blocks.ToArray();

        // Final bare EqualConv2d (k4, pad 0, bias=False): collapses the 4×4 map to [B, styleDim, 1, 1].
        _convFinal = EqualConv2d.Load(w, weightKey: $"{p}.enc.net_app.convs.{idx}.weight", actBiasKey: null,
            biasKey: null, stride: 1, padKind: 0, blurPad0: 0, blurPad1: 0);

        List<EqualLinear> fc = new();
        int j = 0;
        while (w.ContainsKey($"{p}.enc.fc.{j}.weight"))
        {
            fc.Add(EqualLinear.Load(w, $"{p}.enc.fc.{j}"));
            j++;
        }
        if (fc.Count == 0)
            throw new ArgumentException($"motion encoder has no fc layers under {p}.enc.fc.", nameof(w));
        _fc = fc.ToArray();

        _directionWeight = TensorCasts.LoadF32(w, $"{p}.dec.direction.weight");
        OutDim = (int)_directionWeight.Shape[0];
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_conv0 is not null) foreach (Tensor t in _conv0.EnumerateWeights()) yield return t;
        foreach (ResBlock b in _resBlocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        if (_convFinal is not null) foreach (Tensor t in _convFinal.EnumerateWeights()) yield return t;
        foreach (EqualLinear l in _fc) foreach (Tensor t in l.EnumerateWeights()) yield return t;
        if (_directionWeight is not null) yield return _directionWeight;
    }

    /// <summary>Encodes face frames <c>[B, 3, size, size]</c> in [-1, 1] → motion vectors <c>[B, styleDim]</c>
    /// (<c>Generator.get_motion</c>: enc.net_app → enc.fc → dec.direction).</summary>
    public Tensor Forward(IBackend backend, Tensor faceImage)
    {
        int b = (int)faceImage.Shape[0];
        Tensor h = _conv0!.Forward(backend, faceImage);
        foreach (ResBlock blk in _resBlocks) { Tensor n = blk.Forward(backend, h); h.Dispose(); h = n; }
        if (h.Shape[2] != _convFinal!.Kernel || h.Shape[3] != _convFinal.Kernel)
            throw new ArgumentException(
                $"motion encoder spatial map {h.Shape[2]}×{h.Shape[3]} does not match the final conv kernel " +
                $"{_convFinal.Kernel} — face frames must be {ExpectedInputSize()}×{ExpectedInputSize()}.");
        Tensor convOut = _convFinal.Forward(backend, h);   // [B, styleDim, 1, 1]
        h.Dispose();

        int styleDim = (int)convOut.Shape[1];
        Tensor feat = new Tensor(new TensorShape(b, styleDim), DType.F32);
        Buffer.MemoryCopy((float*)convOut.DataPointer, (float*)feat.DataPointer, (long)b * styleDim * 4, (long)b * styleDim * 4);
        convOut.Dispose();

        foreach (EqualLinear l in _fc) { Tensor n = l.Forward(backend, feat); feat.Dispose(); feat = n; }
        // feat is now [B, motionDim]

        // Direction: Q = QR(weight + 1e-8·eye)[0] (economy, orthonormal columns); motion_vec = feat @ Qᵀ.
        int motionDim = (int)_directionWeight!.Shape[1];
        Tensor qt = OrthonormalBasisTransposed(_directionWeight, OutDim, motionDim);   // [motionDim, styleDim]
        Tensor motionVec = new Tensor(new TensorShape(b, OutDim), DType.F32);
        backend.MatMul(motionVec, feat, qt);
        feat.Dispose();
        qt.Dispose();
        return motionVec;
    }

    /// <summary>Pixel input size the loaded conv stack expects (final-kernel × 2^numResBlocks).</summary>
    public int ExpectedInputSize() => _convFinal!.Kernel << _resBlocks.Length;

    /// <summary>Householder QR on the columns of <paramref name="weight"/> <c>[m, n]</c> (+1e-8 on the diagonal,
    /// matching <c>torch.eye(512, motion_dim)</c>), returning <c>Qᵀ</c> <c>[n, m]</c> for the <c>x @ Qᵀ</c> product.
    /// Column signs follow the LAPACK <c>geqrf</c> convention (reflector <c>α = -sign(x₀)·‖x‖</c>), which is what
    /// <c>torch.linalg.qr</c> produces — classical Gram-Schmidt (positive-R) flips 12 of 20 columns on the real
    /// checkpoint, negating those motion components.</summary>
    private static Tensor OrthonormalBasisTransposed(Tensor weight, int m, int n)
    {
        Tensor wf = weight.DType == DType.F32 ? weight : weight.CastTo(DType.F32);
        float* wp = (float*)wf.DataPointer;
        double[][] a = new double[n][];   // column j of the stabilized weight
        for (int j = 0; j < n; j++)
        {
            a[j] = new double[m];
            for (int r = 0; r < m; r++) a[j][r] = wp[(long)r * n + j] + (r == j ? 1e-8 : 0.0);
        }
        // Factor: reflector k zeroes column k below the diagonal; H_k = I - 2·v·vᵀ/‖v‖², v = x - α·e₀ over rows k…m-1.
        double[][] v = new double[n][];
        double[] vNorm2 = new double[n];
        for (int k = 0; k < n; k++)
        {
            int len = m - k;
            double[] vk = new double[len];
            for (int r = 0; r < len; r++) vk[r] = a[k][k + r];
            double norm = 0; for (int r = 0; r < len; r++) norm += vk[r] * vk[r];
            double alpha = vk[0] >= 0 ? -Math.Sqrt(norm) : Math.Sqrt(norm);   // LAPACK dlarfg sign choice
            vk[0] -= alpha;
            double vn = 0; for (int r = 0; r < len; r++) vn += vk[r] * vk[r];
            v[k] = vk; vNorm2[k] = vn;
            if (vn == 0) continue;   // column already zero below the diagonal
            for (int j = k; j < n; j++)
            {
                double dot = 0; for (int r = 0; r < len; r++) dot += vk[r] * a[j][k + r];
                double coef = 2.0 * dot / vn;
                for (int r = 0; r < len; r++) a[j][k + r] -= coef * vk[r];
            }
        }
        // Q's column j = H₀·H₁·…·H_{n-1}·e_j — apply reflectors to the identity column in reverse order.
        Tensor qt = new Tensor(new TensorShape(n, m), DType.F32);
        float* qp = (float*)qt.DataPointer;
        double[] col = new double[m];
        for (int j = 0; j < n; j++)
        {
            Array.Clear(col);
            col[j] = 1.0;
            for (int k = n - 1; k >= 0; k--)
            {
                if (vNorm2[k] == 0) continue;
                int len = m - k;
                double[] vk = v[k];
                double dot = 0; for (int r = 0; r < len; r++) dot += vk[r] * col[k + r];
                double coef = 2.0 * dot / vNorm2[k];
                for (int r = 0; r < len; r++) col[k + r] -= coef * vk[r];
            }
            for (int r = 0; r < m; r++) qp[(long)j * m + r] = (float)col[r];
        }
        if (!ReferenceEquals(wf, weight)) wf.Dispose();
        return qt;
    }

    /// <summary>Fused leaky-ReLU with channel-wise bias: <c>leaky_relu(x + bias, 0.2) · √2</c> over <c>[B, C, …]</c>.</summary>
    private static void FusedLeakyReLU(Tensor x, Tensor bias, int channels)
    {
        const float slope = 0.2f;
        const float scale = 1.4142135623730951f;
        int b = (int)x.Shape[0];
        long spatial = x.Shape.ElementCount / ((long)b * channels);
        float* xp = (float*)x.DataPointer;
        float* bp = (float*)bias.DataPointer;   // stored [1,C,1,1] or [C] — C contiguous floats either way
        for (int bi = 0; bi < b; bi++)
            for (int c = 0; c < channels; c++)
            {
                float biasVal = bp[c];
                long basePos = ((long)bi * channels + c) * spatial;
                for (long s = 0; s < spatial; s++)
                {
                    float v = xp[basePos + s] + biasVal;
                    xp[basePos + s] = (v >= 0f ? v : v * slope) * scale;
                }
            }
    }

    /// <summary>StyleGAN2 <c>EqualConv2d</c> (weight runtime-scaled by <c>1/√(in·k²)</c>, folded at load) with the
    /// ConvLayer wiring: optional leading Blur (downsample), optional trailing FusedLeakyReLU.</summary>
    private sealed class EqualConv2d
    {
        /// <summary>Sentinel for "same" padding <c>k // 2</c> (non-downsample ConvLayer).</summary>
        public const int PadSame = -1;

        private Tensor? _scaledWeight;   // [outC, inC, k, k], scale folded
        private Tensor? _convBias;       // EqualConv2d bias (bare conv only)
        private Tensor? _actBias;        // FusedLeakyReLU bias (activated ConvLayer)
        private int _inC, _outC, _stride, _pad, _blurPad0, _blurPad1;
        private bool _blur;

        public int Kernel { get; private set; }

        public static EqualConv2d Load(IReadOnlyDictionary<string, Tensor> w, string weightKey, string? actBiasKey,
            string? biasKey, int stride, int padKind, int blurPad0, int blurPad1)
        {
            EqualConv2d c = new();
            Tensor raw = TensorCasts.LoadF32(w, weightKey);
            c._outC = (int)raw.Shape[0]; c._inC = (int)raw.Shape[1]; c.Kernel = (int)raw.Shape[2];
            float scale = 1f / MathF.Sqrt(c._inC * c.Kernel * c.Kernel);
            c._scaledWeight = new Tensor(raw.Shape, DType.F32);
            long n = raw.Shape.ElementCount;
            float* sp = (float*)c._scaledWeight.DataPointer, rp = (float*)raw.DataPointer;
            for (long i = 0; i < n; i++) sp[i] = rp[i] * scale;
            if (!ReferenceEquals(raw, w[weightKey])) raw.Dispose();
            if (actBiasKey is not null) c._actBias = TensorCasts.LoadF32(w, actBiasKey);
            if (biasKey is not null && w.TryGetValue(biasKey, out Tensor? cb)) c._convBias = cb.DType == DType.F32 ? cb : cb.CastTo(DType.F32);
            c._stride = stride;
            c._pad = padKind == PadSame ? c.Kernel / 2 : padKind;
            c._blur = blurPad0 > 0 || blurPad1 > 0;
            c._blurPad0 = blurPad0; c._blurPad1 = blurPad1;
            return c;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            if (_scaledWeight is not null) yield return _scaledWeight;
            if (_convBias is not null) yield return _convBias;
            if (_actBias is not null) yield return _actBias;
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            Tensor cur = x;
            bool ownCur = false;
            if (_blur)
            {
                cur = Blur(x, _blurPad0, _blurPad1);
                ownCur = true;
            }
            int h = (int)cur.Shape[2], wd = (int)cur.Shape[3];
            int oh = (h + 2 * _pad - Kernel) / _stride + 1, ow = (wd + 2 * _pad - Kernel) / _stride + 1;
            Tensor o = new Tensor(new TensorShape((int)cur.Shape[0], _outC, oh, ow), DType.F32);
            backend.Conv2D(o, cur, _scaledWeight!, _convBias, _stride, _stride, _pad, _pad);
            if (ownCur) cur.Dispose();
            if (_actBias is not null) FusedLeakyReLU(o, _actBias, _outC);
            return o;
        }

        /// <summary>upfirdn2d with the normalized <c>[1,3,3,1]</c> outer-product kernel (up=down=1): asymmetric zero
        /// pad, depthwise correlation with the flipped kernel (symmetric here, so a plain correlation).</summary>
        private static Tensor Blur(Tensor x, int pad0, int pad1)
        {
            ReadOnlySpan<float> k1 = [1f, 3f, 3f, 1f];
            Span<float> kern = stackalloc float[16];
            float sum = 0;
            for (int a = 0; a < 4; a++) for (int bb = 0; bb < 4; bb++) { kern[a * 4 + bb] = k1[a] * k1[bb]; sum += kern[a * 4 + bb]; }
            for (int i = 0; i < 16; i++) kern[i] /= sum;

            int b = (int)x.Shape[0], c = (int)x.Shape[1], h = (int)x.Shape[2], wd = (int)x.Shape[3];
            int oh = h + pad0 + pad1 - 4 + 1, ow = wd + pad0 + pad1 - 4 + 1;
            Tensor o = new Tensor(new TensorShape(b, c, oh, ow), DType.F32);
            float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
            for (int bi = 0; bi < b; bi++)
                for (int ci = 0; ci < c; ci++)
                {
                    long inBase = ((long)bi * c + ci) * h * wd;
                    long outBase = ((long)bi * c + ci) * oh * ow;
                    for (int y = 0; y < oh; y++)
                        for (int xo = 0; xo < ow; xo++)
                        {
                            float acc = 0;
                            for (int ky = 0; ky < 4; ky++)
                            {
                                int iy = y + ky - pad0;
                                if (iy < 0 || iy >= h) continue;
                                for (int kx = 0; kx < 4; kx++)
                                {
                                    int ix = xo + kx - pad0;
                                    if (ix < 0 || ix >= wd) continue;
                                    acc += xp[inBase + (long)iy * wd + ix] * kern[ky * 4 + kx];
                                }
                            }
                            op[outBase + (long)y * ow + xo] = acc;
                        }
                }
            return o;
        }
    }

    /// <summary>StyleGAN2 <c>EqualLinear</c> (no activation in this model): <c>x @ (w·1/√in)ᵀ + bias</c>, scale folded at load.</summary>
    private sealed class EqualLinear
    {
        private Tensor? _scaledWeight, _bias;
        private int _outDim;

        public static EqualLinear Load(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            EqualLinear l = new();
            Tensor raw = TensorCasts.LoadF32(w, $"{p}.weight");
            l._outDim = (int)raw.Shape[0];
            float scale = 1f / MathF.Sqrt((int)raw.Shape[1]);
            l._scaledWeight = new Tensor(raw.Shape, DType.F32);
            long n = raw.Shape.ElementCount;
            float* sp = (float*)l._scaledWeight.DataPointer, rp = (float*)raw.DataPointer;
            for (long i = 0; i < n; i++) sp[i] = rp[i] * scale;
            if (!ReferenceEquals(raw, w[$"{p}.weight"])) raw.Dispose();
            if (w.TryGetValue($"{p}.bias", out Tensor? b)) l._bias = b;
            return l;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            if (_scaledWeight is not null) yield return _scaledWeight;
            if (_bias is not null) yield return _bias;
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            Tensor o = new Tensor(new TensorShape((int)x.Shape[0], _outDim), DType.F32);
            backend.Linear(o, x, _scaledWeight!, _bias);
            return o;
        }
    }

    /// <summary>StyleGAN2 <c>ResBlock</c>: <c>(conv2(conv1(x)) + skip(x)) / √2</c> — conv1 k3 same-pad activated,
    /// conv2 k3 downsample (blur pad (2,2) → stride 2) activated, skip k1 downsample (blur pad (1,1)) bare.</summary>
    private sealed class ResBlock
    {
        private EqualConv2d? _conv1, _conv2, _skip;

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            // ConvLayer Sequential indices: non-downsample = [EqualConv2d, FusedLeakyReLU] → .0.weight/.1.bias;
            // downsample = [Blur, EqualConv2d, FusedLeakyReLU] → .1.weight/.2.bias (the .0.kernel blur buffer is
            // recomputed, not loaded). Blur pad = ((p+1)//2, p//2) with p = (4-2)+(k-1) → k3: (2,2), k1: (1,1).
            _conv1 = EqualConv2d.Load(w, $"{p}.conv1.0.weight", $"{p}.conv1.1.bias", biasKey: null,
                stride: 1, padKind: EqualConv2d.PadSame, blurPad0: 0, blurPad1: 0);
            _conv2 = EqualConv2d.Load(w, $"{p}.conv2.1.weight", $"{p}.conv2.2.bias", biasKey: null,
                stride: 2, padKind: 0, blurPad0: 2, blurPad1: 2);
            _skip = EqualConv2d.Load(w, $"{p}.skip.1.weight", actBiasKey: null, biasKey: null,
                stride: 2, padKind: 0, blurPad0: 1, blurPad1: 1);
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor t in _conv1!.EnumerateWeights()) yield return t;
            foreach (Tensor t in _conv2!.EnumerateWeights()) yield return t;
            foreach (Tensor t in _skip!.EnumerateWeights()) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            Tensor a = _conv1!.Forward(backend, x);
            Tensor b = _conv2!.Forward(backend, a); a.Dispose();
            Tensor skip = _skip!.Forward(backend, x);
            long n = b.Shape.ElementCount;
            float* bp = (float*)b.DataPointer, sp = (float*)skip.DataPointer;
            const float inv = 0.7071067811865476f;
            for (long i = 0; i < n; i++) bp[i] = (bp[i] + sp[i]) * inv;
            skip.Dispose();
            return b;
        }
    }
}
