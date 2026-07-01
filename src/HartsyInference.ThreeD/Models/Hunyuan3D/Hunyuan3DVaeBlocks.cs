using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>ShapeVAE transformer self-attention block (CLIP-style <c>ResidualAttentionBlock</c>): affine-LN →
/// fused <c>c_qkv</c> (head-interleaved) → per-head LayerNorm QK-norm → SDPA → <c>c_proj</c> residual; then
/// affine-LN → erf-GELU MLP residual. All Linears are F32; <c>c_qkv</c> has no bias.</summary>
internal sealed unsafe class Hunyuan3DVaeResBlock
{
    private readonly int _width, _heads, _headDim, _mlpDim;
    private Tensor? _ln1W, _ln1B, _qkvW, _projW, _projB, _qnW, _qnB, _knW, _knB;
    private Tensor? _ln2W, _ln2B, _fcW, _fcB, _fcpW, _fcpB;

    public Hunyuan3DVaeResBlock(int width, int heads)
    {
        _width = width; _heads = heads; _headDim = width / heads; _mlpDim = 4 * width;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _ln1W = F(w, $"{p}.ln_1.weight"); _ln1B = F(w, $"{p}.ln_1.bias");
        _qkvW = F(w, $"{p}.attn.c_qkv.weight");
        _projW = F(w, $"{p}.attn.c_proj.weight"); _projB = F(w, $"{p}.attn.c_proj.bias");
        _qnW = F(w, $"{p}.attn.attention.q_norm.weight"); _qnB = F(w, $"{p}.attn.attention.q_norm.bias");
        _knW = F(w, $"{p}.attn.attention.k_norm.weight"); _knB = F(w, $"{p}.attn.attention.k_norm.bias");
        _ln2W = F(w, $"{p}.ln_2.weight"); _ln2B = F(w, $"{p}.ln_2.bias");
        _fcW = F(w, $"{p}.mlp.c_fc.weight"); _fcB = F(w, $"{p}.mlp.c_fc.bias");
        _fcpW = F(w, $"{p}.mlp.c_proj.weight"); _fcpB = F(w, $"{p}.mlp.c_proj.bias");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_ln1W, _ln1B, _qkvW, _projW, _projB, _qnW, _qnB, _knW, _knB, _ln2W, _ln2B, _fcW, _fcB, _fcpW, _fcpB];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        int b = (int)x.Shape[0], n = (int)x.Shape[1];

        Tensor n1 = new(x.Shape, DType.F32); backend.LayerNorm(n1, x, _ln1W!, _ln1B!, 1e-6f);
        Tensor qkv = new(new TensorShape(b, n, 3 * _width), DType.F32); backend.Linear(qkv, n1, _qkvW!, null); n1.Dispose();
        Tensor q = new(new TensorShape(b, n, _width), DType.F32), k = new(new TensorShape(b, n, _width), DType.F32), v = new(new TensorShape(b, n, _width), DType.F32);
        Hunyuan3DVaeOps.DeinterleaveQkv(qkv, q, k, v, b, n, _heads, _headDim); qkv.Dispose();
        Hunyuan3DVaeOps.HeadLayerNorm(q, _qnW!, _qnB!, b, n, _heads, _headDim);
        Hunyuan3DVaeOps.HeadLayerNorm(k, _knW!, _knB!, b, n, _heads, _headDim);
        Tensor a = Hunyuan3DAttention.Attend(backend, q, k, v, _heads); q.Dispose(); k.Dispose(); v.Dispose();
        Tensor proj = new(new TensorShape(b, n, _width), DType.F32); backend.Linear(proj, a, _projW!, _projB!); a.Dispose();
        Tensor x1 = new(x.Shape, DType.F32); backend.Add(x1, x, proj); proj.Dispose();

        Tensor n2 = new(x1.Shape, DType.F32); backend.LayerNorm(n2, x1, _ln2W!, _ln2B!, 1e-6f);
        Tensor f1 = new(new TensorShape(b, n, _mlpDim), DType.F32); backend.Linear(f1, n2, _fcW!, _fcB!); n2.Dispose();
        Tensor act = new(f1.Shape, DType.F32); backend.Gelu(act, f1); f1.Dispose();
        Tensor f2 = new(new TensorShape(b, n, _width), DType.F32); backend.Linear(f2, act, _fcpW!, _fcpB!); act.Dispose();
        Tensor x2 = new(x1.Shape, DType.F32); backend.Add(x2, x1, f2); x1.Dispose(); f2.Dispose();
        return x2;
    }

    private static Tensor F(IReadOnlyDictionary<string, Tensor> w, string k) => Hunyuan3DShapeVae.F32(w[k]);
}

/// <summary>ShapeVAE <c>geo_decoder</c>: Fourier-embed query points → <c>query_proj</c> → one
/// <c>ResidualCrossAttentionBlock</c> (queries cross-attend to the latents) → <c>ln_post</c> → <c>output_proj</c>
/// (Width→1 occupancy). K/V (from the latents, query-independent) are precomputed once via <see cref="PrepareKv"/>.</summary>
internal sealed unsafe class Hunyuan3DGeoDecoder
{
    private readonly int _width, _heads, _headDim, _bands, _fourierDim, _mlpDim;
    private Tensor? _qProjW, _qProjB;
    private Tensor? _ln1W, _ln1B, _ln2W, _ln2B, _ln3W, _ln3B;
    private Tensor? _cqW, _ckvW, _cprojW, _cprojB, _qnW, _qnB, _knW, _knB;
    private Tensor? _fcW, _fcB, _fcpW, _fcpB;
    private Tensor? _lnPostW, _lnPostB, _outW, _outB;

    public Hunyuan3DGeoDecoder(int width, int heads, int bands, int fourierDim)
    {
        _width = width; _heads = heads; _headDim = width / heads; _bands = bands; _fourierDim = fourierDim; _mlpDim = 4 * width;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _qProjW = F(w, $"{p}.query_proj.weight"); _qProjB = F(w, $"{p}.query_proj.bias");
        string c = $"{p}.cross_attn_decoder";
        _ln1W = F(w, $"{c}.ln_1.weight"); _ln1B = F(w, $"{c}.ln_1.bias");
        _ln2W = F(w, $"{c}.ln_2.weight"); _ln2B = F(w, $"{c}.ln_2.bias");
        _ln3W = F(w, $"{c}.ln_3.weight"); _ln3B = F(w, $"{c}.ln_3.bias");
        _cqW = F(w, $"{c}.attn.c_q.weight");
        _ckvW = F(w, $"{c}.attn.c_kv.weight");
        _cprojW = F(w, $"{c}.attn.c_proj.weight"); _cprojB = F(w, $"{c}.attn.c_proj.bias");
        _qnW = F(w, $"{c}.attn.attention.q_norm.weight"); _qnB = F(w, $"{c}.attn.attention.q_norm.bias");
        _knW = F(w, $"{c}.attn.attention.k_norm.weight"); _knB = F(w, $"{c}.attn.attention.k_norm.bias");
        _fcW = F(w, $"{c}.mlp.c_fc.weight"); _fcB = F(w, $"{c}.mlp.c_fc.bias");
        _fcpW = F(w, $"{c}.mlp.c_proj.weight"); _fcpB = F(w, $"{c}.mlp.c_proj.bias");
        _lnPostW = F(w, $"{p}.ln_post.weight"); _lnPostB = F(w, $"{p}.ln_post.bias");
        _outW = F(w, $"{p}.output_proj.weight"); _outB = F(w, $"{p}.output_proj.bias");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_qProjW, _qProjB, _ln1W, _ln1B, _ln2W, _ln2B, _ln3W, _ln3B, _cqW, _ckvW, _cprojW, _cprojB,
            _qnW, _qnB, _knW, _knB, _fcW, _fcB, _fcpW, _fcpB, _lnPostW, _lnPostB, _outW, _outB];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    /// <summary>Precomputes the query-independent cross-attn K (LayerNorm'd) and V from the processed latents:
    /// <c>ln_2(latents)</c> → <c>c_kv</c> → de-interleave → <c>k_norm(k)</c>. Returns (kNorm [1,Nl,W], v [1,Nl,W]).</summary>
    public (Tensor kNorm, Tensor v) PrepareKv(IBackend backend, Tensor latents)
    {
        int b = (int)latents.Shape[0], nl = (int)latents.Shape[1];
        Tensor n2 = new(latents.Shape, DType.F32); backend.LayerNorm(n2, latents, _ln2W!, _ln2B!, 1e-6f);
        Tensor kvFused = new(new TensorShape(b, nl, 2 * _width), DType.F32); backend.Linear(kvFused, n2, _ckvW!, null); n2.Dispose();
        Tensor k = new(new TensorShape(b, nl, _width), DType.F32), v = new(new TensorShape(b, nl, _width), DType.F32);
        Hunyuan3DVaeOps.DeinterleaveKv(kvFused, k, v, b, nl, _heads, _headDim); kvFused.Dispose();
        Hunyuan3DVaeOps.HeadLayerNorm(k, _knW!, _knB!, b, nl, _heads, _headDim);
        return (k, v);
    }

    /// <summary>Occupancy for query points <c>[count,3]</c> given precomputed (kNorm, v). Returns <c>[count]</c>.</summary>
    public Tensor Query(IBackend backend, (Tensor kNorm, Tensor v) kv, ReadOnlySpan<float> coords, int count)
    {
        // Fourier embed → query_proj → query_embeddings (also the residual base).
        Tensor feat = new(new TensorShape(1, count, _fourierDim), DType.F32);
        Hunyuan3DVaeOps.FourierEmbed(feat, coords, count, _bands);
        Tensor qEmb = new(new TensorShape(1, count, _width), DType.F32); backend.Linear(qEmb, feat, _qProjW!, _qProjB!); feat.Dispose();

        // cross_attn_decoder: x = qEmb + c_proj(attn(q_norm(c_q(ln_1(qEmb))), kNorm, v)).
        Tensor n1 = new(qEmb.Shape, DType.F32); backend.LayerNorm(n1, qEmb, _ln1W!, _ln1B!, 1e-6f);
        Tensor q = new(new TensorShape(1, count, _width), DType.F32); backend.Linear(q, n1, _cqW!, null); n1.Dispose();
        Hunyuan3DVaeOps.HeadLayerNorm(q, _qnW!, _qnB!, 1, count, _heads, _headDim);
        Tensor a = Hunyuan3DAttention.Attend(backend, q, kv.kNorm, kv.v, _heads); q.Dispose();
        Tensor proj = new(new TensorShape(1, count, _width), DType.F32); backend.Linear(proj, a, _cprojW!, _cprojB!); a.Dispose();
        Tensor x1 = new(qEmb.Shape, DType.F32); backend.Add(x1, qEmb, proj); qEmb.Dispose(); proj.Dispose();

        // x = x + mlp(ln_3(x)).
        Tensor n3 = new(x1.Shape, DType.F32); backend.LayerNorm(n3, x1, _ln3W!, _ln3B!, 1e-6f);
        Tensor f1 = new(new TensorShape(1, count, _mlpDim), DType.F32); backend.Linear(f1, n3, _fcW!, _fcB!); n3.Dispose();
        Tensor act = new(f1.Shape, DType.F32); backend.Gelu(act, f1); f1.Dispose();
        Tensor f2 = new(new TensorShape(1, count, _width), DType.F32); backend.Linear(f2, act, _fcpW!, _fcpB!); act.Dispose();
        Tensor x2 = new(x1.Shape, DType.F32); backend.Add(x2, x1, f2); x1.Dispose(); f2.Dispose();

        // ln_post → output_proj (Width → 1).
        Tensor post = new(x2.Shape, DType.F32); backend.LayerNorm(post, x2, _lnPostW!, _lnPostB!, 1e-5f); x2.Dispose();
        Tensor occ = new(new TensorShape(1, count, 1), DType.F32); backend.Linear(occ, post, _outW!, _outB!); post.Dispose();
        Tensor flat = new(new TensorShape(count), DType.F32);
        new ReadOnlySpan<float>((float*)occ.DataPointer, count).CopyTo(new Span<float>((float*)flat.DataPointer, count));
        occ.Dispose();
        return flat;
    }

    private static Tensor F(IReadOnlyDictionary<string, Tensor> w, string k) => Hunyuan3DShapeVae.F32(w[k]);
}

/// <summary>Host tensor glue for the ShapeVAE: fused-QKV/KV de-interleave (per-head [q,k,v]/[k,v]), per-head
/// affine LayerNorm QK-norm, and the FourierEmbedder (<c>[x, sin(x·2^i), cos(x·2^i)]</c>, include_pi=false).</summary>
internal static unsafe class Hunyuan3DVaeOps
{
    /// <summary>Fused c_qkv [B,N,3W] head-interleaved (per head [q,k,v]·D) → q,k,v [B,N,W] head-major ([h·D+d]).</summary>
    public static void DeinterleaveQkv(Tensor qkv, Tensor q, Tensor k, Tensor v, int b, int n, int h, int d)
    {
        float* sp = (float*)qkv.DataPointer; float* qp = (float*)q.DataPointer; float* kp = (float*)k.DataPointer; float* vp = (float*)v.DataPointer;
        int w = h * d; long rows = (long)b * n;
        for (long r = 0; r < rows; r++)
        {
            float* src = sp + r * 3 * w;
            for (int hh = 0; hh < h; hh++)
            {
                float* hb = src + hh * 3 * d;
                new ReadOnlySpan<float>(hb, d).CopyTo(new Span<float>(qp + r * w + hh * d, d));
                new ReadOnlySpan<float>(hb + d, d).CopyTo(new Span<float>(kp + r * w + hh * d, d));
                new ReadOnlySpan<float>(hb + 2 * d, d).CopyTo(new Span<float>(vp + r * w + hh * d, d));
            }
        }
    }

    /// <summary>Fused c_kv [B,N,2W] head-interleaved (per head [k,v]·D) → k,v [B,N,W] head-major.</summary>
    public static void DeinterleaveKv(Tensor kv, Tensor k, Tensor v, int b, int n, int h, int d)
    {
        float* sp = (float*)kv.DataPointer; float* kp = (float*)k.DataPointer; float* vp = (float*)v.DataPointer;
        int w = h * d; long rows = (long)b * n;
        for (long r = 0; r < rows; r++)
        {
            float* src = sp + r * 2 * w;
            for (int hh = 0; hh < h; hh++)
            {
                float* hb = src + hh * 2 * d;
                new ReadOnlySpan<float>(hb, d).CopyTo(new Span<float>(kp + r * w + hh * d, d));
                new ReadOnlySpan<float>(hb + d, d).CopyTo(new Span<float>(vp + r * w + hh * d, d));
            }
        }
    }

    /// <summary>In-place per-head affine LayerNorm over the head dim (eps 1e-6). x is [B,N,H·D] head-major.</summary>
    public static void HeadLayerNorm(Tensor x, Tensor weight, Tensor bias, int b, int n, int h, int d)
    {
        float* xp = (float*)x.DataPointer; float* wp = (float*)weight.DataPointer; float* bp = (float*)bias.DataPointer;
        long vecs = (long)b * n * h;
        for (long v = 0; v < vecs; v++)
        {
            float* row = xp + v * d;
            float mean = 0f; for (int i = 0; i < d; i++) mean += row[i]; mean /= d;
            float var = 0f; for (int i = 0; i < d; i++) { float df = row[i] - mean; var += df * df; } var /= d;
            float inv = 1f / MathF.Sqrt(var + 1e-6f);
            for (int i = 0; i < d; i++) row[i] = (row[i] - mean) * inv * wp[i] + bp[i];
        }
    }

    /// <summary>FourierEmbedder (num_freqs bands, logspace freqs 2^i, include_pi=false, include_input=true):
    /// out = [x(3), sin(x⊗freqs)(3·bands), cos(x⊗freqs)(3·bands)]; the sin/cos halves are coord-major.</summary>
    public static void FourierEmbed(Tensor dst, ReadOnlySpan<float> coords, int count, int bands)
    {
        float* dp = (float*)dst.DataPointer;
        int dim = 3 * (2 * bands + 1);
        int sinBase = 3, cosBase = 3 + 3 * bands;
        for (int i = 0; i < count; i++)
        {
            long o = (long)i * dim;
            for (int c = 0; c < 3; c++)
            {
                float x = coords[i * 3 + c];
                dp[o + c] = x;
                for (int band = 0; band < bands; band++)
                {
                    float a = x * (1 << band);   // freq = 2^band (include_pi=false → no π)
                    dp[o + sinBase + c * bands + band] = MathF.Sin(a);
                    dp[o + cosBase + c * bands + band] = MathF.Cos(a);
                }
            }
        }
    }
}
