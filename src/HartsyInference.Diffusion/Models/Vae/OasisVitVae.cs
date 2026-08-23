using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Oasis-500m's continuous Gaussian ViT-VAE (<c>vit-l-20-shallow-encoder</c> in <c>vae.py</c>) — a pure
/// transformer autoencoder, NOT a VQ tokenizer and structurally simpler than the conv VAEs: patch-embed (20×20,
/// stride 20) → 6 ViT blocks → LayerNorm → linear Gaussian bottleneck (1024 → 2·16, μ taken deterministically) →
/// linear 16 → 1024 → 12 ViT blocks → LayerNorm → linear patch predictor (1024 → 1200) → unpatchify. ViT blocks use
/// affine LayerNorm, fused-QKV self-attention (bias) with 2-D axial "pixel" RoPE over the first 32 of 64 head dims
/// (<see cref="AxialRope2D"/> dimPerAxis 16), and a 4× MLP. The latent scaling factor (20/255) is applied by
/// the pipeline, not here. 360×640 is the only supported resolution upstream; any H/W divisible by the patch works
/// structurally. Numerics validation-pending. See <c>docs/Research/OASIS_ARCHITECTURE.md</c> § 5.
/// <para><b>Known follow-up:</b> upstream's <c>vae.py</c> uses timm's default <c>nn.GELU</c> (exact/erf) in the MLP,
/// but this path calls <c>backend.Gelu</c> which is the tanh approximation (≤1.5e-4 abs difference). A faithful fix
/// needs a backend erf-GELU kernel (CPU + CUDA); deferred to avoid breaking GPU residency with a CPU fallback. The DiT
/// MLP correctly uses tanh-GELU. Tracked in <c>docs/Checklists/PARITY_VERIFICATION.md</c>.</para></summary>
public sealed unsafe class OasisVitVae
{
    private readonly int _latentDim;
    private readonly int _patchSize;
    private readonly int _dim;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _encDepth;
    private readonly int _decDepth;
    private readonly int _mlpHidden;
    private readonly AxialRope2D _rope;

    private Tensor? _patchW2d, _patchB;            // Conv2d(3→dim, 20, stride 20) as linear [dim, 3·p²]
    private VitBlock[] _encoder = [];
    private Tensor? _encNormW, _encNormB;
    private Tensor? _quantW, _quantB;              // Linear(dim → 2·latent)
    private Tensor? _postQuantW, _postQuantB;      // Linear(latent → dim)
    private VitBlock[] _decoder = [];
    private Tensor? _decNormW, _decNormB;
    private Tensor? _predictorW, _predictorB;      // Linear(dim → 3·p²)

    public OasisVitVae(int latentDim = 16, int patchSize = 20, int dim = 1024, int heads = 16,
        int encDepth = 6, int decDepth = 12, float mlpRatio = 4.0f)
    {
        _latentDim = latentDim;
        _patchSize = patchSize;
        _dim = dim;
        _heads = heads;
        _headDim = dim / heads;
        _encDepth = encDepth;
        _decDepth = decDepth;
        _mlpHidden = (int)(dim * mlpRatio);
        // rotary dim = head_dim // 4 per axis, max_freq = 18·32 = 576 (the trained grid).
        _rope = new AxialRope2D(_headDim / 4, maxFreq: 576);
    }

    /// <summary>Latent channels (16).</summary>
    public int LatentDim => _latentDim;

    /// <summary>Spatial compression factor (20).</summary>
    public int PatchSize => _patchSize;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        int patchVec = 3 * _patchSize * _patchSize;
        _patchW2d = WanDitOps.Reshape2d(w["patch_embed.proj.weight"], _dim, patchVec);
        w.TryGetValue("patch_embed.proj.bias", out _patchB);

        _encoder = new VitBlock[_encDepth];
        for (int i = 0; i < _encDepth; i++)
        {
            _encoder[i] = new VitBlock(this);
            _encoder[i].LoadWeights(w, $"encoder.{i}");
        }
        _encNormW = TensorCasts.LoadF32(w, "enc_norm.weight"); w.TryGetValue("enc_norm.bias", out _encNormB);
        _quantW = w["quant_conv.weight"]; w.TryGetValue("quant_conv.bias", out _quantB);
        _postQuantW = w["post_quant_conv.weight"]; w.TryGetValue("post_quant_conv.bias", out _postQuantB);

        _decoder = new VitBlock[_decDepth];
        for (int i = 0; i < _decDepth; i++)
        {
            _decoder[i] = new VitBlock(this);
            _decoder[i].LoadWeights(w, $"decoder.{i}");
        }
        _decNormW = TensorCasts.LoadF32(w, "dec_norm.weight"); w.TryGetValue("dec_norm.bias", out _decNormB);
        _predictorW = w["predictor.weight"]; w.TryGetValue("predictor.bias", out _predictorB);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _patchW2d, _patchB, _encNormW, _encNormB, _quantW, _quantB,
            _postQuantW, _postQuantB, _decNormW, _decNormB, _predictorW, _predictorB })
            if (t is not null) yield return t;
        foreach (VitBlock b in _encoder) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (VitBlock b in _decoder) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Encodes RGB <c>[1, 3, H, W]</c> in [-1, 1] to the deterministic posterior mean <c>[1, 16, H/20, W/20]</c>
    /// (unscaled — the pipeline multiplies by the scaling factor).</summary>
    public Tensor Encode(IBackend backend, Tensor rgb)
    {
        (int gh, int gw) = ValidateGrid(rgb, channels: 3);
        int s = gh * gw;
        Tensor tokens = PatchifyRgb(backend, rgb, gh, gw);
        (Tensor cos, Tensor sin) = _rope.Build(gh, gw);
        Tensor cur = tokens;
        foreach (VitBlock b in _encoder)
        {
            Tensor next = b.Forward(backend, cur, cos, sin, s);
            cur.Dispose();
            cur = next;
        }
        cos.Dispose(); sin.Dispose();
        LayerNormRows(cur, _encNormW!, _encNormB, s, _dim);
        Tensor moments = new Tensor(new TensorShape(s, 2 * _latentDim), DType.F32);
        backend.Linear(moments, cur, _quantW!, _quantB);
        cur.Dispose();

        // μ = first latentDim of (mean, logvar); inference uses the mean, not a sample.
        Tensor latent = new Tensor(new TensorShape([1L, _latentDim, gh, gw]), DType.F32);
        float* mp = (float*)moments.DataPointer;
        float* lp = (float*)latent.DataPointer;
        for (int i = 0; i < s; i++)
            for (int c = 0; c < _latentDim; c++)
                lp[(long)c * s + i] = mp[(long)i * 2 * _latentDim + c];
        moments.Dispose();
        return latent;
    }

    /// <summary>Decodes a latent <c>[1, 16, h, w]</c> (already unscaled) to RGB <c>[1, 3, h·20, w·20]</c> in [-1, 1].</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        (int gh, int gw) = (
            (int)latent.Shape[2],
            (int)latent.Shape[3]);
        if ((int)latent.Shape[1] != _latentDim)
            throw new ArgumentException($"latent channels {latent.Shape[1]} != {_latentDim}.", nameof(latent));
        int s = gh * gw;

        // Channel-major latent → token rows [S, latentDim].
        Tensor rows = new Tensor(new TensorShape(s, _latentDim), DType.F32);
        float* lp = (float*)latent.DataPointer;
        float* rp = (float*)rows.DataPointer;
        for (int i = 0; i < s; i++)
            for (int c = 0; c < _latentDim; c++)
                rp[(long)i * _latentDim + c] = lp[(long)c * s + i];

        Tensor cur = new Tensor(new TensorShape(s, _dim), DType.F32);
        backend.Linear(cur, rows, _postQuantW!, _postQuantB);
        rows.Dispose();

        (Tensor cos, Tensor sin) = _rope.Build(gh, gw);
        foreach (VitBlock b in _decoder)
        {
            Tensor next = b.Forward(backend, cur, cos, sin, s);
            cur.Dispose();
            cur = next;
        }
        cos.Dispose(); sin.Dispose();
        LayerNormRows(cur, _decNormW!, _decNormB, s, _dim);
        int patchVec = 3 * _patchSize * _patchSize;
        Tensor pixels = new Tensor(new TensorShape(s, patchVec), DType.F32);
        backend.Linear(pixels, cur, _predictorW!, _predictorB);
        cur.Dispose();

        Tensor rgb = UnpatchifyRgb(pixels, gh, gw);
        pixels.Dispose();
        return rgb;
    }

    private (int gh, int gw) ValidateGrid(Tensor rgb, int channels)
    {
        if (rgb.Shape.Rank != 4 || rgb.Shape[1] != channels)
            throw new ArgumentException($"expected [1,{channels},H,W]; got {rgb.Shape}.", nameof(rgb));
        int h = (int)rgb.Shape[2], w = (int)rgb.Shape[3];
        if (h % _patchSize != 0 || w % _patchSize != 0)
            throw new ArgumentException($"H/W must be divisible by the patch size {_patchSize}.", nameof(rgb));
        return (h / _patchSize, w / _patchSize);
    }

    /// <summary>Conv2d(3→dim, p, stride p) as a linear over flattened patches (channel-major patch vector, matching the conv weight layout).</summary>
    private Tensor PatchifyRgb(IBackend backend, Tensor rgb, int gh, int gw)
    {
        int p = _patchSize, s = gh * gw, patchVec = 3 * p * p;
        int h = gh * p, w = gw * p;
        Tensor patches = new Tensor(new TensorShape(s, patchVec), DType.F32);
        float* xp = (float*)rgb.DataPointer;
        float* pp = (float*)patches.DataPointer;
        for (int ty = 0; ty < gh; ty++)
            for (int tx = 0; tx < gw; tx++)
            {
                long baseIdx = ((long)ty * gw + tx) * patchVec;
                int d = 0;
                for (int c = 0; c < 3; c++)
                    for (int py = 0; py < p; py++)
                        for (int px = 0; px < p; px++)
                            pp[baseIdx + d++] = xp[((long)c * h + ty * p + py) * w + tx * p + px];
            }
        Tensor tokens = new Tensor(new TensorShape(s, _dim), DType.F32);
        backend.Linear(tokens, patches, _patchW2d!, _patchB);
        patches.Dispose();
        return tokens;
    }

    private Tensor UnpatchifyRgb(Tensor pixels, int gh, int gw)
    {
        int p = _patchSize, h = gh * p, w = gw * p, patchVec = 3 * p * p;
        Tensor rgb = new Tensor(new TensorShape([1L, 3, h, w]), DType.F32);
        float* sp = (float*)pixels.DataPointer;
        float* op = (float*)rgb.DataPointer;
        for (int ty = 0; ty < gh; ty++)
            for (int tx = 0; tx < gw; tx++)
            {
                long baseIdx = ((long)ty * gw + tx) * patchVec;
                int d = 0;
                for (int c = 0; c < 3; c++)
                    for (int py = 0; py < p; py++)
                        for (int px = 0; px < p; px++)
                            op[((long)c * h + ty * p + py) * w + tx * p + px] = sp[baseIdx + d++];
            }
        return rgb;
    }

    private void LayerNormRows(Tensor x, Tensor weight, Tensor? bias, int rows, int dim)
    {
        float* xp = (float*)x.DataPointer;
        float* wp = (float*)weight.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;
        for (int r = 0; r < rows; r++)
        {
            long off = (long)r * dim;
            double mean = 0;
            for (int d = 0; d < dim; d++) mean += xp[off + d];
            mean /= dim;
            double var = 0;
            for (int d = 0; d < dim; d++) { double dd = xp[off + d] - mean; var += dd * dd; }
            float inv = 1f / MathF.Sqrt((float)(var / dim) + 1e-6f);
            for (int d = 0; d < dim; d++)
            {
                float n = (float)((xp[off + d] - mean) * inv);
                xp[off + d] = n * wp[d] + (bp is null ? 0f : bp[d]);
            }
        }
    }

    /// <summary>One ViT block: affine LN → fused-QKV axial-RoPE self-attention → residual; affine LN → GELU-tanh MLP → residual.</summary>
    private sealed class VitBlock
    {
        private readonly OasisVitVae _vae;
        private Tensor? _norm1W, _norm1B, _norm2W, _norm2B;
        private Tensor? _qkvW, _qkvB, _projW, _projB;
        private Tensor? _fc1W, _fc1B, _fc2W, _fc2B;

        public VitBlock(OasisVitVae vae) => _vae = vae;

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _norm1W = TensorCasts.LoadF32(w, $"{p}.norm1.weight"); w.TryGetValue($"{p}.norm1.bias", out _norm1B);
            _norm2W = TensorCasts.LoadF32(w, $"{p}.norm2.weight"); w.TryGetValue($"{p}.norm2.bias", out _norm2B);
            _qkvW = w[$"{p}.attn.qkv.weight"]; w.TryGetValue($"{p}.attn.qkv.bias", out _qkvB);
            _projW = w[$"{p}.attn.proj.weight"]; w.TryGetValue($"{p}.attn.proj.bias", out _projB);
            _fc1W = w[$"{p}.mlp.fc1.weight"]; w.TryGetValue($"{p}.mlp.fc1.bias", out _fc1B);
            _fc2W = w[$"{p}.mlp.fc2.weight"]; w.TryGetValue($"{p}.mlp.fc2.bias", out _fc2B);
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor? t in new[] { _norm1W, _norm1B, _norm2W, _norm2B, _qkvW, _qkvB, _projW, _projB, _fc1W, _fc1B, _fc2W, _fc2B })
                if (t is not null) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor x, Tensor cos, Tensor sin, int s)
        {
            int dim = _vae._dim, heads = _vae._heads, hd = _vae._headDim;
            Tensor normed = VaeOps.Clone(x);
            _vae.LayerNormRows(normed, _norm1W!, _norm1B, s, dim);
            Tensor qkv = new Tensor(new TensorShape(s, 3 * dim), DType.F32);
            backend.Linear(qkv, normed, _qkvW!, _qkvB);
            normed.Dispose();

            Tensor q = SliceQkv(qkv, s, dim, 0);
            Tensor k = SliceQkv(qkv, s, dim, 1);
            Tensor v = SliceQkv(qkv, s, dim, 2);
            qkv.Dispose();
            _vae._rope.ApplyRotary(q, cos, sin, heads, hd);
            _vae._rope.ApplyRotary(k, cos, sin, heads, hd);

            Tensor qMh = ToBhsd(q, s, heads, hd); q.Dispose();
            Tensor kMh = ToBhsd(k, s, heads, hd); k.Dispose();
            Tensor vMh = ToBhsd(v, s, heads, hd); v.Dispose();
            Tensor attn = new Tensor(new TensorShape([1L, heads, s, hd]), DType.F32);
            backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, null, 1.0f / MathF.Sqrt(hd));
            qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

            Tensor flat = FromBhsd(attn, s, heads, hd);
            attn.Dispose();
            Tensor proj = new Tensor(new TensorShape(s, dim), DType.F32);
            backend.Linear(proj, flat, _projW!, _projB);
            flat.Dispose();
            Tensor h1 = AddRows(x, proj, s, dim);
            proj.Dispose();

            Tensor n2 = VaeOps.Clone(h1);
            _vae.LayerNormRows(n2, _norm2W!, _norm2B, s, dim);
            Tensor mid = new Tensor(new TensorShape(s, _vae._mlpHidden), DType.F32);
            backend.Linear(mid, n2, _fc1W!, _fc1B);
            n2.Dispose();
            Tensor act = new Tensor(mid.Shape, DType.F32);
            backend.Gelu(act, mid);
            mid.Dispose();
            Tensor mlpOut = new Tensor(new TensorShape(s, dim), DType.F32);
            backend.Linear(mlpOut, act, _fc2W!, _fc2B);
            act.Dispose();
            Tensor outT = AddRows(h1, mlpOut, s, dim);
            h1.Dispose();
            mlpOut.Dispose();
            return outT;
        }

        private static Tensor SliceQkv(Tensor qkv, int s, int dim, int part)
        {
            Tensor o = new Tensor(new TensorShape(s, dim), DType.F32);
            float* sp = (float*)qkv.DataPointer;
            float* op = (float*)o.DataPointer;
            for (int i = 0; i < s; i++)
                Buffer.MemoryCopy(sp + (long)i * 3 * dim + part * dim, op + (long)i * dim, (long)dim * 4, (long)dim * 4);
            return o;
        }

        private static Tensor ToBhsd(Tensor x, int s, int heads, int hd)
        {
            Tensor o = new Tensor(new TensorShape([1L, heads, s, hd]), DType.F32);
            float* xp = (float*)x.DataPointer;
            float* op = (float*)o.DataPointer;
            for (int i = 0; i < s; i++)
                for (int h = 0; h < heads; h++)
                    Buffer.MemoryCopy(xp + ((long)i * heads + h) * hd, op + ((long)h * s + i) * hd, (long)hd * 4, (long)hd * 4);
            return o;
        }

        private static Tensor FromBhsd(Tensor x, int s, int heads, int hd)
        {
            Tensor o = new Tensor(new TensorShape(s, heads * hd), DType.F32);
            float* xp = (float*)x.DataPointer;
            float* op = (float*)o.DataPointer;
            for (int h = 0; h < heads; h++)
                for (int i = 0; i < s; i++)
                    Buffer.MemoryCopy(xp + ((long)h * s + i) * hd, op + ((long)i * heads + h) * hd, (long)hd * 4, (long)hd * 4);
            return o;
        }

        private static Tensor AddRows(Tensor a, Tensor b, int s, int dim)
        {
            Tensor o = new Tensor(new TensorShape(s, dim), DType.F32);
            long n = (long)s * dim;
            float* ap = (float*)a.DataPointer;
            float* bp = (float*)b.DataPointer;
            float* op = (float*)o.DataPointer;
            for (long i = 0; i < n; i++) op[i] = ap[i] + bp[i];
            return o;
        }
    }
}
