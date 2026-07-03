using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.LanguageModels.Gpt;

/// <summary>One GPT-2 pre-norm block: <c>x + Attn(LN1(x))</c> then <c>x + MLP(LN2(x))</c>. Fused QKV
/// projection, multi-head SDPA, 4× GELU MLP, all linears <c>bias=False</c>; LayerNorms carry weight+bias.
/// Key scheme follows HF Bark (<c>layernorm_1</c>, <c>attn.att_proj</c>, <c>attn.out_proj</c>,
/// <c>layernorm_2</c>, <c>mlp.in_proj</c>, <c>mlp.out_proj</c>).</summary>
public sealed unsafe class GptBlock : IDisposable
{
    private readonly GptConfig _cfg;
    private int _disposed;

    private Tensor? _ln1G, _ln1B, _ln2G, _ln2B;
    private Tensor? _attW, _outW;         // att_proj [3H, H], out_proj [H, H]
    private Tensor? _mlpInW, _mlpOutW;    // [4H, H], [H, 4H]

    public GptBlock(GptConfig cfg) => _cfg = cfg;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _ln1G = WhisperOps.EnsureF32(w[$"{prefix}.layernorm_1.weight"]);
        _ln1B = LoadBiasOrZero(w, $"{prefix}.layernorm_1.bias", _ln1G);
        _attW = WhisperOps.EnsureF32(w[$"{prefix}.attn.att_proj.weight"]);
        _outW = WhisperOps.EnsureF32(w[$"{prefix}.attn.out_proj.weight"]);
        _ln2G = WhisperOps.EnsureF32(w[$"{prefix}.layernorm_2.weight"]);
        _ln2B = LoadBiasOrZero(w, $"{prefix}.layernorm_2.bias", _ln2G);
        _mlpInW = WhisperOps.EnsureF32(w[$"{prefix}.mlp.in_proj.weight"]);
        _mlpOutW = WhisperOps.EnsureF32(w[$"{prefix}.mlp.out_proj.weight"]);
    }

    public Tensor Forward(IBackend backend, Tensor x, Tensor? causalMask, GptKvCache.GptKvLayer? cacheOut = null, int cacheOffset = 0)
    {
        int t = (int)x.Shape[1];
        int h = _cfg.Hidden;
        int nh = _cfg.NumHeads;
        int d = _cfg.HeadDim;

        // ── Attention ──
        Tensor ln1 = new(x.Shape, DType.F32);
        backend.LayerNorm(ln1, x, _ln1G!, _ln1B!, 1e-5f);
        Tensor qkv = WhisperOps.ProjectLinear(backend, ln1, _attW!, bias: null, 1, t, h, 3 * h);
        ln1.Dispose();

        (Tensor q, Tensor k, Tensor v) = SplitQkv(qkv, t, h);
        qkv.Dispose();

        // Prefill capture: stash the prompt's K/V rows so generation can continue incrementally (ForwardStep).
        if (cacheOut is not null)
        {
            new ReadOnlySpan<float>((void*)k.DataPointer, t * h).CopyTo(cacheOut.K.AsSpan(cacheOffset * h));
            new ReadOnlySpan<float>((void*)v.DataPointer, t * h).CopyTo(cacheOut.V.AsSpan(cacheOffset * h));
        }
        Tensor qMh = ToHeads(q, t, nh, d); q.Dispose();
        Tensor kMh = ToHeads(k, t, nh, d); k.Dispose();
        Tensor vMh = ToHeads(v, t, nh, d); v.Dispose();

        Tensor attn = new(new TensorShape(1, nh, t, d), DType.F32);
        backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, causalMask, 1f / MathF.Sqrt(d));
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor attnFlat = new(new TensorShape(1, t, h), DType.F32);
        WhisperOps.ReshapeFromMultiHead4D(attnFlat, attn, 1, t, nh, d);
        attn.Dispose();
        Tensor attnOut = WhisperOps.ProjectLinear(backend, attnFlat, _outW!, bias: null, 1, t, h, h);
        attnFlat.Dispose();

        Tensor res1 = new(x.Shape, DType.F32);
        AddInto(res1, x, attnOut);
        attnOut.Dispose();

        // ── MLP ──
        Tensor ln2 = new(res1.Shape, DType.F32);
        backend.LayerNorm(ln2, res1, _ln2G!, _ln2B!, 1e-5f);
        Tensor fc = WhisperOps.ProjectLinear(backend, ln2, _mlpInW!, bias: null, 1, t, h, _cfg.MlpDim);
        ln2.Dispose();
        backend.Gelu(fc, fc);
        Tensor proj = WhisperOps.ProjectLinear(backend, fc, _mlpOutW!, bias: null, 1, t, _cfg.MlpDim, h);
        fc.Dispose();

        Tensor res2 = new(res1.Shape, DType.F32);
        AddInto(res2, res1, proj);
        res1.Dispose();
        proj.Dispose();
        return res2;
    }

    /// <summary>Single-token decode step against a K/V cache: projections run on the backend (weights stay
    /// device-resident), the 1×L attention runs on CPU over the cached prefix (K/V never cross PCIe). The new
    /// token's K/V rows are appended at <paramref name="pos"/>; attention covers rows <c>[0, pos]</c> (causal by
    /// construction, no mask). Input and output are <c>[1, 1, hidden]</c>.</summary>
    public Tensor ForwardStep(IBackend backend, Tensor x, GptKvCache.GptKvLayer cache, int pos)
    {
        int h = _cfg.Hidden;
        int nh = _cfg.NumHeads;
        int d = _cfg.HeadDim;

        // ── Attention (incremental) ──
        Tensor ln1 = new(x.Shape, DType.F32);
        backend.LayerNorm(ln1, x, _ln1G!, _ln1B!, 1e-5f);
        Tensor qkv = WhisperOps.ProjectLinear(backend, ln1, _attW!, bias: null, 1, 1, h, 3 * h);
        ln1.Dispose();

        float* qkvP = (float*)qkv.DataPointer;
        new ReadOnlySpan<float>(qkvP + h, h).CopyTo(cache.K.AsSpan(pos * h));
        new ReadOnlySpan<float>(qkvP + 2 * h, h).CopyTo(cache.V.AsSpan(pos * h));

        int len = pos + 1;
        float scale = 1f / MathF.Sqrt(d);
        Tensor attnFlat = new(new TensorShape(1, 1, h), DType.F32);
        float* outP = (float*)attnFlat.DataPointer;
        Span<float> scores = len <= 4096 ? stackalloc float[len] : new float[len];
        for (int head = 0; head < nh; head++)
        {
            int hOff = head * d;
            float* qh = qkvP + hOff;
            // scores = q · K_head over the cached prefix, softmax in place
            float max = float.NegativeInfinity;
            for (int l = 0; l < len; l++)
            {
                float dot = 0f;
                int rowOff = l * h + hOff;
                for (int j = 0; j < d; j++) dot += qh[j] * cache.K[rowOff + j];
                dot *= scale;
                scores[l] = dot;
                if (dot > max) max = dot;
            }
            float sum = 0f;
            for (int l = 0; l < len; l++)
            {
                float e = MathF.Exp(scores[l] - max);
                scores[l] = e;
                sum += e;
            }
            float inv = 1f / sum;
            float* oh = outP + hOff;
            for (int j = 0; j < d; j++) oh[j] = 0f;
            for (int l = 0; l < len; l++)
            {
                float w = scores[l] * inv;
                int rowOff = l * h + hOff;
                for (int j = 0; j < d; j++) oh[j] += w * cache.V[rowOff + j];
            }
        }
        qkv.Dispose();

        Tensor attnOut = WhisperOps.ProjectLinear(backend, attnFlat, _outW!, bias: null, 1, 1, h, h);
        attnFlat.Dispose();
        Tensor res1 = new(x.Shape, DType.F32);
        AddInto(res1, x, attnOut);
        attnOut.Dispose();

        // ── MLP ──
        Tensor ln2 = new(res1.Shape, DType.F32);
        backend.LayerNorm(ln2, res1, _ln2G!, _ln2B!, 1e-5f);
        Tensor fc = WhisperOps.ProjectLinear(backend, ln2, _mlpInW!, bias: null, 1, 1, h, _cfg.MlpDim);
        ln2.Dispose();
        backend.Gelu(fc, fc);
        Tensor proj = WhisperOps.ProjectLinear(backend, fc, _mlpOutW!, bias: null, 1, 1, _cfg.MlpDim, h);
        fc.Dispose();

        Tensor res2 = new(res1.Shape, DType.F32);
        AddInto(res2, res1, proj);
        res1.Dispose();
        proj.Dispose();
        return res2;
    }

    /// <summary>Loads a LayerNorm bias, or returns a zero tensor (matching <paramref name="weightLike"/>'s length)
    /// when the checkpoint has none — HF Bark's <c>BarkLayerNorm</c> uses <c>bias=False</c>.</summary>
    internal static Tensor LoadBiasOrZero(IReadOnlyDictionary<string, Tensor> w, string key, Tensor weightLike)
    {
        if (w.TryGetValue(key, out Tensor? b)) return WhisperOps.EnsureF32(b);
        Tensor zero = new(weightLike.Shape, DType.F32);
        float* p = (float*)zero.DataPointer;
        for (long i = 0; i < zero.ElementCount; i++) p[i] = 0f;
        return zero;
    }

    private static (Tensor q, Tensor k, Tensor v) SplitQkv(Tensor qkv, int t, int h)
    {
        Tensor q = new(new TensorShape(1, t, h), DType.F32);
        Tensor k = new(new TensorShape(1, t, h), DType.F32);
        Tensor v = new(new TensorShape(1, t, h), DType.F32);
        float* src = (float*)qkv.DataPointer;
        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer;
        for (int s = 0; s < t; s++)
        {
            long row = (long)s * 3 * h;
            long dst = (long)s * h;
            for (int c = 0; c < h; c++)
            {
                qp[dst + c] = src[row + c];
                kp[dst + c] = src[row + h + c];
                vp[dst + c] = src[row + 2 * h + c];
            }
        }
        return (q, k, v);
    }

    private static Tensor ToHeads(Tensor seq, int t, int nh, int d)
    {
        Tensor outT = new(new TensorShape(1, nh, t, d), DType.F32);
        WhisperOps.ReshapeToMultiHead4D(outT, seq, 1, t, nh, d);
        return outT;
    }

    private static void AddInto(Tensor dst, Tensor a, Tensor b)
    {
        float* dp = (float*)dst.DataPointer;
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        long n = dst.ElementCount;
        for (long i = 0; i < n; i++) dp[i] = ap[i] + bp[i];
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_ln1G, _ln1B, _attW, _outW, _ln2G, _ln2B, _mlpInW, _mlpOutW];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
