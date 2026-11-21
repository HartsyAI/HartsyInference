using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.BreezeTts;

/// <summary>Port of the T5Gemma2 text encoder bundled with Breeze TTS 2 (the repo's <c>t5gemma2_compat.py</c>). Each
/// layer is <c>x + post_attn_norm(attn(pre_attn_norm(x)))</c> then <c>x + post_ffn_norm(mlp(pre_ffn_norm(x)))</c> with
/// Gemma RMSNorm (<c>1 + w</c>), per-head Q/K RMSNorm, split-half RoPE, GQA, scale <c>query_pre_attn_scalar^-½</c> and a
/// tanh-GELU gated MLP. Attention is bidirectional: sliding layers let a token see the previous
/// <c>(window+1)/2</c> and the next <c>window/2+1</c> positions (itself included), full layers see everything.
/// <para>The attention itself runs through the shared <see cref="AttentionLayer"/> on the backend, grouped heads native;
/// a sliding layer's window is an additive bias, and none is passed where the window covers the whole segment.</para></summary>
public sealed unsafe class T5Gemma2TextEncoder : IDisposable
{
    private readonly T5Gemma2TextEncoderConfig _cfg;
    private readonly AttentionLayer _attention;
    private Tensor? _embed, _eoi, _finalNorm;
    private Layer[] _layers = [];
    private int _disposed;

    private sealed record Layer(Tensor PreAttn, Tensor PostAttn, Tensor PreFfn, Tensor PostFfn, Tensor QNorm, Tensor KNorm,
        Tensor Wq, Tensor Wk, Tensor Wv, Tensor Wo, Tensor Gate, Tensor Up, Tensor Down);

    public T5Gemma2TextEncoder(T5Gemma2TextEncoderConfig cfg)
    {
        _cfg = cfg;
        _attention = AttentionFor(cfg);
    }

    /// <summary>The weightless attention layer of this encoder's geometry: grouped key/value heads, F32, and the scale
    /// <c>query_pre_attn_scalar^-½</c> in place of <c>1/√headDim</c>.</summary>
    internal static AttentionLayer AttentionFor(T5Gemma2TextEncoderConfig cfg) => new(new AttentionLayerConfig
    {
        Heads = cfg.NumHeads, KvHeads = cfg.NumKvHeads, HeadDim = cfg.HeadDim, Scale = 1f / MathF.Sqrt(cfg.QueryPreAttnScalar),
        ComputeDType = DType.F32,
    });

    public int HiddenSize => _cfg.HiddenSize;

    /// <summary>Loads <c>text_encoder.*</c>-prefixed (or bare, via <paramref name="prefix"/>) weights. Norm weights are
    /// stored as <c>w</c> and applied as <c>1 + w</c>, so they are converted once here.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "text_encoder")
    {
        _embed = WhisperOps.EnsureF32(w[$"{prefix}.embed_tokens.weight"]);
        _eoi = w.TryGetValue($"{prefix}.embed_tokens.eoi_embedding", out Tensor? eoi) ? WhisperOps.EnsureF32(eoi) : null;
        _finalNorm = OnePlus(w[$"{prefix}.norm.weight"]);
        _layers = new Layer[_cfg.NumLayers];
        for (int i = 0; i < _layers.Length; i++)
        {
            string p = $"{prefix}.layers.{i}";
            _layers[i] = new Layer(
                OnePlus(w[$"{p}.pre_self_attn_layernorm.weight"]), OnePlus(w[$"{p}.post_self_attn_layernorm.weight"]),
                OnePlus(w[$"{p}.pre_feedforward_layernorm.weight"]), OnePlus(w[$"{p}.post_feedforward_layernorm.weight"]),
                OnePlus(w[$"{p}.self_attn.q_norm.weight"]), OnePlus(w[$"{p}.self_attn.k_norm.weight"]),
                w[$"{p}.self_attn.q_proj.weight"], w[$"{p}.self_attn.k_proj.weight"], w[$"{p}.self_attn.v_proj.weight"],
                w[$"{p}.self_attn.o_proj.weight"], w[$"{p}.mlp.gate_proj.weight"], w[$"{p}.mlp.up_proj.weight"],
                w[$"{p}.mlp.down_proj.weight"]);
        }
    }

    private static Tensor OnePlus(Tensor w) => NormWeights.OnePlus(w);

    /// <summary>Encodes one text segment (positions 0..n-1) to its <c>[n, hidden]</c> final hidden states.</summary>
    public float[] Encode(IBackend backend, ReadOnlySpan<int> tokens, Action<string, float[]>? tap = null)
    {
        int n = tokens.Length, h = _cfg.HiddenSize, heads = _cfg.NumHeads, kvHeads = _cfg.NumKvHeads, hd = _cfg.HeadDim;
        float embedScale = MathF.Sqrt(h);
        float[] x = new float[n * h];
        float* emb = (float*)_embed!.DataPointer;
        for (int i = 0; i < n; i++)
        {
            if (tokens[i] == _cfg.EoiTokenIndex && _eoi is not null)
                new ReadOnlySpan<float>((void*)_eoi.DataPointer, h).CopyTo(x.AsSpan(i * h, h));
            else
                for (int c = 0; c < h; c++) x[i * h + c] = emb[(long)tokens[i] * h + c] * embedScale;
        }
        tap?.Invoke("embed", (float[])x.Clone());

        float[,] localRope = RopeTable(n, hd, _cfg.LocalRopeTheta, 1f);
        float[,] globalRope = RopeTable(n, hd, _cfg.GlobalRopeTheta, _cfg.GlobalRopeLinearFactor);
        int left = (_cfg.SlidingWindow + 1) / 2, right = _cfg.SlidingWindow / 2 + 1;
        // One mask for every sliding layer of this segment; none where the window takes in all of it or no layer slides.
        bool slides = false;
        for (int li = 0; li < _layers.Length && !slides; li++) slides = !_cfg.IsFullAttention(li);
        using Tensor? window = slides ? WindowBias(n, left, right) : null;

        for (int li = 0; li < _layers.Length; li++)
        {
            Layer l = _layers[li];
            bool full = _cfg.IsFullAttention(li);
            float[,] rope = full ? globalRope : localRope;

            float[] a = Norm(backend, x, l.PreAttn, n, h);
            float[] q = DacOps.Linear(backend, a, l.Wq, n, h, heads * hd);
            float[] k = DacOps.Linear(backend, a, l.Wk, n, h, kvHeads * hd);
            float[] v = DacOps.Linear(backend, a, l.Wv, n, h, kvHeads * hd);
            q = Norm(backend, q, l.QNorm, n * heads, hd);
            k = Norm(backend, k, l.KNorm, n * kvHeads, hd);
            ApplyRope(q, rope, n, heads, hd);
            ApplyRope(k, rope, n, kvHeads, hd);

            float[] ctx = Attend(backend, _attention, q, k, v, n, heads, kvHeads, hd, full ? null : window);
            float[] attn = DacOps.Linear(backend, ctx, l.Wo, n, heads * hd, h);
            attn = Norm(backend, attn, l.PostAttn, n, h);
            for (int i = 0; i < x.Length; i++) x[i] += attn[i];

            float[] f = Norm(backend, x, l.PreFfn, n, h);
            float[] gate = DacOps.Linear(backend, f, l.Gate, n, h, _cfg.IntermediateSize);
            float[] up = DacOps.Linear(backend, f, l.Up, n, h, _cfg.IntermediateSize);
            for (int i = 0; i < gate.Length; i++) gate[i] = GeluTanh(gate[i]) * up[i];
            float[] ff = DacOps.Linear(backend, gate, l.Down, n, _cfg.IntermediateSize, h);
            ff = Norm(backend, ff, l.PostFfn, n, h);
            for (int i = 0; i < x.Length; i++) x[i] += ff[i];
            tap?.Invoke($"layer{li}", (float[])x.Clone());
        }
        return Norm(backend, x, _finalNorm!, n, h);
    }

    /// <summary>What a sliding layer adds to its scores: zero where token <c>i</c> sees token <c>j</c> (the previous
    /// <paramref name="left"/> and the next <paramref name="right"/> positions, each counting <c>i</c> itself) and −∞
    /// elsewhere, F32 <c>[n, n]</c>. Null where every token sees the whole segment, which needs no mask.</summary>
    internal static Tensor? WindowBias(int n, int left, int right)
    {
        if (left >= n && right >= n) return null;
        Tensor bias = new(new TensorShape(n, n), DType.F32);
        Span<float> rows = bias.AsSpan<float>();
        rows.Fill(float.NegativeInfinity);
        for (int i = 0; i < n; i++)
        {
            int lo = Math.Max(0, i - left + 1), hi = Math.Min(n - 1, i + right - 1);
            rows.Slice(i * n + lo, hi - lo + 1).Clear();
        }
        return bias;
    }

    /// <summary>Attention over one segment: <paramref name="q"/> <c>[n, heads · hd]</c>, <paramref name="k"/> and
    /// <paramref name="v"/> <c>[n, kvHeads · hd]</c>, already normed and rotated, with <paramref name="bias"/> added to the
    /// scores when given. Returns <c>[n, heads · hd]</c>.</summary>
    internal static float[] Attend(IBackend backend, AttentionLayer layer, float[] q, float[] k, float[] v, int n, int heads,
        int kvHeads, int hd, Tensor? bias)
    {
        Tensor? qT = null, kT = null, vT = null;
        try
        {
            qT = DacOps.FromHost(q, 1, n, heads * hd);
            kT = DacOps.FromHost(k, 1, n, kvHeads * hd);
            vT = DacOps.FromHost(v, 1, n, kvHeads * hd);
        }
        catch
        {
            qT?.Dispose();
            kT?.Dispose();
            vT?.Dispose();
            throw;
        }
        // The layer takes the three operands over and disposes them.
        using Tensor attended = layer.AttendFlat(backend, qT, kT, vT, new TensorShape(1, n, heads * hd),
            new AttentionCallContext { Bias = bias });
        return DacOps.ToHost(attended);
    }

    private float[] Norm(IBackend backend, float[] x, Tensor weight, int rows, int dim)
        => DacOps.RmsNorm(backend, x, weight, rows, dim, _cfg.RmsNormEps);

    internal static float GeluTanh(float x)
        => 0.5f * x * (1f + MathF.Tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));

    internal static float[,] RopeTable(int n, int hd, float theta, float linearFactor)
    {
        float[,] t = new float[n, hd];   // [pos, i] = cos(angle_i); [pos, hd/2 + i] = sin(angle_i)
        for (int p = 0; p < n; p++)
            for (int i = 0; i < hd / 2; i++)
            {
                double f = 1.0 / Math.Pow(theta, 2.0 * i / hd) / linearFactor, a = p * f;
                t[p, i] = (float)Math.Cos(a); t[p, hd / 2 + i] = (float)Math.Sin(a);
            }
        return t;
    }

    // HF rotate_half: pairs dimension i with i + hd/2.
    internal static void ApplyRope(float[] v, float[,] table, int n, int heads, int hd)
    {
        int half = hd / 2;
        for (int p = 0; p < n; p++)
            for (int hh = 0; hh < heads; hh++)
            {
                int b = (p * heads + hh) * hd;
                for (int i = 0; i < half; i++)
                {
                    float c = table[p, i], s = table[p, half + i];
                    float x0 = v[b + i], x1 = v[b + half + i];
                    v[b + i] = x0 * c - x1 * s;
                    v[b + half + i] = x1 * c + x0 * s;
                }
            }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        // The norm weights are this object's own copies (1 + w), made in LoadWeights; the projections are the caller's.
        _finalNorm?.Dispose();
        foreach (Layer l in _layers)
            foreach (Tensor norm in (ReadOnlySpan<Tensor>)[l.PreAttn, l.PostAttn, l.PreFfn, l.PostFfn, l.QNorm, l.KNorm])
                norm.Dispose();
        GC.SuppressFinalize(this);
    }
}
