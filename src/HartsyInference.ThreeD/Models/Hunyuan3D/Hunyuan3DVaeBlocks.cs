using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>ShapeVAE transformer self-attention block (CLIP-style <c>ResidualAttentionBlock</c>), <b>GPU-resident</b>:
/// affine-LN → q/k/v (the fused head-interleaved <c>c_qkv</c> weight is split into head-major q/k/v weights at load
/// so each is a plain Linear) → per-head LayerNorm QK-norm → SDPA → <c>c_proj</c> residual; then affine-LN → erf-GELU
/// MLP residual. All glue routes through <see cref="IBackend"/> — no mid-forward host <c>DataPointer</c> reads.</summary>
internal sealed unsafe class Hunyuan3DVaeResBlock
{
    private readonly int _width, _heads, _headDim, _mlpDim;
    private Tensor? _ln1W, _ln1B, _qW, _kW, _vW, _projW, _projB, _qnW, _qnB, _knW, _knB;
    private Tensor? _ln2W, _ln2B, _fcW, _fcB, _fcpW, _fcpB;

    public Hunyuan3DVaeResBlock(int width, int heads)
    {
        _width = width; _heads = heads; _headDim = width / heads; _mlpDim = 4 * width;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _ln1W = F(w, $"{p}.ln_1.weight"); _ln1B = F(w, $"{p}.ln_1.bias");
        (_qW, _kW, _vW) = Hunyuan3DVaeOps.SplitInterleavedWeight(F(w, $"{p}.attn.c_qkv.weight"), 3, _heads, _headDim);
        _projW = F(w, $"{p}.attn.c_proj.weight"); _projB = F(w, $"{p}.attn.c_proj.bias");
        _qnW = F(w, $"{p}.attn.attention.q_norm.weight"); _qnB = F(w, $"{p}.attn.attention.q_norm.bias");
        _knW = F(w, $"{p}.attn.attention.k_norm.weight"); _knB = F(w, $"{p}.attn.attention.k_norm.bias");
        _ln2W = F(w, $"{p}.ln_2.weight"); _ln2B = F(w, $"{p}.ln_2.bias");
        _fcW = F(w, $"{p}.mlp.c_fc.weight"); _fcB = F(w, $"{p}.mlp.c_fc.bias");
        _fcpW = F(w, $"{p}.mlp.c_proj.weight"); _fcpB = F(w, $"{p}.mlp.c_proj.bias");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_ln1W, _ln1B, _qW, _kW, _vW, _projW, _projB, _qnW, _qnB, _knW, _knB, _ln2W, _ln2B, _fcW, _fcB, _fcpW, _fcpB];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        int n = (int)x.Shape[1];
        float scale = 1f / MathF.Sqrt(_headDim);
        TensorShape flat = new(1, n, _width), heads = new(1, n, _heads, _headDim), mh = new(1, _heads, n, _headDim);

        Tensor n1 = new(flat, DType.F32); backend.LayerNorm(n1, x, _ln1W!, _ln1B!, 1e-6f);
        Tensor q = new(heads, DType.F32); backend.Linear(q, n1, _qW!, null);
        Tensor k = new(heads, DType.F32); backend.Linear(k, n1, _kW!, null);
        Tensor v = new(heads, DType.F32); backend.Linear(v, n1, _vW!, null); n1.Dispose();
        Tensor qn = new(heads, DType.F32); backend.LayerNorm(qn, q, _qnW!, _qnB!, 1e-6f); q.Dispose();
        Tensor kn = new(heads, DType.F32); backend.LayerNorm(kn, k, _knW!, _knB!, 1e-6f); k.Dispose();
        Tensor qP = new(mh, DType.F32); backend.Permute0213(qP, qn, n, _heads, _headDim); qn.Dispose();
        Tensor kP = new(mh, DType.F32); backend.Permute0213(kP, kn, n, _heads, _headDim); kn.Dispose();
        Tensor vP = new(mh, DType.F32); backend.Permute0213(vP, v, n, _heads, _headDim); v.Dispose();
        Tensor attn = new(mh, DType.F32); backend.ScaledDotProductAttention(attn, qP, kP, vP, null, scale, allowF16: true); qP.Dispose(); kP.Dispose(); vP.Dispose();
        Tensor attnFlat = new(flat, DType.F32); backend.Permute0213(attnFlat, attn, _heads, n, _headDim); attn.Dispose();
        Tensor proj = new(flat, DType.F32); backend.Linear(proj, attnFlat, _projW!, _projB!); attnFlat.Dispose();
        Tensor x1 = new(flat, DType.F32); backend.Add(x1, x, proj); proj.Dispose();

        Tensor n2 = new(flat, DType.F32); backend.LayerNorm(n2, x1, _ln2W!, _ln2B!, 1e-6f);
        Tensor f1 = new(new TensorShape(1, n, _mlpDim), DType.F32); backend.Linear(f1, n2, _fcW!, _fcB!); n2.Dispose();
        Tensor act = new(f1.Shape, DType.F32); backend.Gelu(act, f1); f1.Dispose();
        Tensor f2 = new(flat, DType.F32); backend.Linear(f2, act, _fcpW!, _fcpB!); act.Dispose();
        Tensor x2 = new(flat, DType.F32); backend.Add(x2, x1, f2); x1.Dispose(); f2.Dispose();
        return x2;
    }

    private static Tensor F(IReadOnlyDictionary<string, Tensor> w, string k) => Hunyuan3DShapeVae.F32(w[k]);
}

/// <summary>ShapeVAE <c>geo_decoder</c> (GPU-resident): Fourier-embed query points → <c>query_proj</c> → one
/// cross-attn block (queries attend to the latents via precomputed K/V) → <c>ln_post</c> → <c>output_proj</c>
/// (Width→1 occupancy). The fused head-interleaved <c>c_kv</c> weight is split into head-major k/v weights at load.</summary>
internal sealed unsafe class Hunyuan3DGeoDecoder
{
    private readonly int _width, _heads, _headDim, _bands, _fourierDim, _mlpDim;
    private Tensor? _qProjW, _qProjB;
    private Tensor? _ln1W, _ln1B, _ln2W, _ln2B, _ln3W, _ln3B;
    private Tensor? _cqW, _ckW, _cvW, _cprojW, _cprojB, _qnW, _qnB, _knW, _knB;
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
        (_ckW, _cvW, _) = Hunyuan3DVaeOps.SplitInterleavedWeight(F(w, $"{c}.attn.c_kv.weight"), 2, _heads, _headDim);
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
        Tensor?[] all = [_qProjW, _qProjB, _ln1W, _ln1B, _ln2W, _ln2B, _ln3W, _ln3B, _cqW, _ckW, _cvW, _cprojW, _cprojB,
            _qnW, _qnB, _knW, _knB, _fcW, _fcB, _fcpW, _fcpB, _lnPostW, _lnPostB, _outW, _outB];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    /// <summary>Precomputes the query-independent cross-attn K/V (head-major, permuted to [1,H,Nl,D]) from the
    /// processed latents: <c>ln_2(latents)</c> → k/v Linears → k_norm(k). Returns (kP, vP) [1,H,Nl,D].</summary>
    public (Tensor kP, Tensor vP) PrepareKv(IBackend backend, Tensor latents)
    {
        int nl = (int)latents.Shape[1];
        TensorShape flat = new(1, nl, _width), heads = new(1, nl, _heads, _headDim), mh = new(1, _heads, nl, _headDim);
        Tensor n2 = new(flat, DType.F32); backend.LayerNorm(n2, latents, _ln2W!, _ln2B!, 1e-6f);
        Tensor k = new(heads, DType.F32); backend.Linear(k, n2, _ckW!, null);
        Tensor v = new(heads, DType.F32); backend.Linear(v, n2, _cvW!, null); n2.Dispose();
        Tensor kn = new(heads, DType.F32); backend.LayerNorm(kn, k, _knW!, _knB!, 1e-6f); k.Dispose();
        Tensor kP = new(mh, DType.F32); backend.Permute0213(kP, kn, nl, _heads, _headDim); kn.Dispose();
        Tensor vP = new(mh, DType.F32); backend.Permute0213(vP, v, nl, _heads, _headDim); v.Dispose();
        return (kP, vP);
    }

    /// <summary>Occupancy for query points <c>[count,3]</c> given precomputed (kP, vP). Returns <c>[count]</c>.</summary>
    public Tensor Query(IBackend backend, (Tensor kP, Tensor vP) kv, Tensor coords, int count)
    {
        int nl = (int)kv.kP.Shape[2];
        float scale = 1f / MathF.Sqrt(_headDim);
        TensorShape flat = new(1, count, _width), heads = new(1, count, _heads, _headDim), qmh = new(1, _heads, count, _headDim);

        // Fourier embed on-device (coords [count,3] → feat stays GPU-resident) → query_proj → query_embeddings.
        Tensor feat = new(new TensorShape(1, count, _fourierDim), DType.F32);
        backend.FourierEmbed(feat, coords, count, _bands);
        Tensor qEmb = new(flat, DType.F32); backend.Linear(qEmb, feat, _qProjW!, _qProjB!); feat.Dispose();

        // cross-attn: q = q_norm(c_q(ln_1(qEmb))); attn(q, kP, vP); x = qEmb + c_proj(attn).
        Tensor n1 = new(flat, DType.F32); backend.LayerNorm(n1, qEmb, _ln1W!, _ln1B!, 1e-6f);
        Tensor q = new(heads, DType.F32); backend.Linear(q, n1, _cqW!, null); n1.Dispose();
        Tensor qn = new(heads, DType.F32); backend.LayerNorm(qn, q, _qnW!, _qnB!, 1e-6f); q.Dispose();
        Tensor qP = new(qmh, DType.F32); backend.Permute0213(qP, qn, count, _heads, _headDim); qn.Dispose();
        Tensor attn = new(qmh, DType.F32); backend.ScaledDotProductAttention(attn, qP, kv.kP, kv.vP, null, scale, allowF16: true); qP.Dispose();
        Tensor attnFlat = new(flat, DType.F32); backend.Permute0213(attnFlat, attn, _heads, count, _headDim); attn.Dispose();
        Tensor proj = new(flat, DType.F32); backend.Linear(proj, attnFlat, _cprojW!, _cprojB!); attnFlat.Dispose();
        Tensor x1 = new(flat, DType.F32); backend.Add(x1, qEmb, proj); qEmb.Dispose(); proj.Dispose();

        // x = x + mlp(ln_3(x)).
        Tensor n3 = new(flat, DType.F32); backend.LayerNorm(n3, x1, _ln3W!, _ln3B!, 1e-6f);
        Tensor f1 = new(new TensorShape(1, count, _mlpDim), DType.F32); backend.Linear(f1, n3, _fcW!, _fcB!); n3.Dispose();
        Tensor act = new(f1.Shape, DType.F32); backend.Gelu(act, f1); f1.Dispose();
        Tensor f2 = new(flat, DType.F32); backend.Linear(f2, act, _fcpW!, _fcpB!); act.Dispose();
        Tensor x2 = new(flat, DType.F32); backend.Add(x2, x1, f2); x1.Dispose(); f2.Dispose();

        // ln_post (eps 1e-5) → output_proj (Width → 1).
        Tensor post = new(flat, DType.F32); backend.LayerNorm(post, x2, _lnPostW!, _lnPostB!, 1e-5f); x2.Dispose();
        Tensor occ = new(new TensorShape(1, count, 1), DType.F32); backend.Linear(occ, post, _outW!, _outB!); post.Dispose();
        Tensor flatOut = new(new TensorShape(count), DType.F32);
        new ReadOnlySpan<float>((float*)occ.DataPointer, count).CopyTo(new Span<float>((float*)flatOut.DataPointer, count));
        occ.Dispose();
        return flatOut;
    }

    private static Tensor F(IReadOnlyDictionary<string, Tensor> w, string k) => Hunyuan3DShapeVae.F32(w[k]);
}

/// <summary>Host helpers for the ShapeVAE: one-time head-interleaved weight split (so fused <c>c_qkv</c>/<c>c_kv</c>
/// become plain head-major Linears). The FourierEmbedder is now a device op (<see cref="IBackend.FourierEmbed"/>).</summary>
internal static unsafe class Hunyuan3DVaeOps
{
    /// <summary>Splits a fused, head-interleaved projection weight <c>[k·W, W]</c> (row layout per head [t=0..k-1]·D)
    /// into <paramref name="k"/> head-major weights <c>[W, W]</c> (row = h·D+d ← fused row h·k·D + t·D + d). One-time.</summary>
    public static (Tensor, Tensor, Tensor) SplitInterleavedWeight(Tensor fused, int k, int heads, int headDim)
    {
        int w = heads * headDim, inDim = (int)fused.Shape[1];
        Tensor[] outs = new Tensor[k];
        float* fp = (float*)fused.DataPointer;
        for (int t = 0; t < k; t++)
        {
            Tensor o = new(new TensorShape(w, inDim), DType.F32);
            float* op = (float*)o.DataPointer;
            for (int h = 0; h < heads; h++)
                for (int d = 0; d < headDim; d++)
                {
                    long src = ((long)(h * k * headDim + t * headDim + d)) * inDim;
                    long dst = ((long)(h * headDim + d)) * inDim;
                    new ReadOnlySpan<float>(fp + src, inDim).CopyTo(new Span<float>(op + dst, inDim));
                }
            outs[t] = o;
        }
        return (outs[0], outs[1], k > 2 ? outs[2] : outs[1]);
    }

}
