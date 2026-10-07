using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.BreezeTts;

/// <summary>Port of the T5Gemma2 text encoder bundled with Breeze TTS 2 (the repo's <c>t5gemma2_compat.py</c>). Each
/// layer is <c>x + post_attn_norm(attn(pre_attn_norm(x)))</c> then <c>x + post_ffn_norm(mlp(pre_ffn_norm(x)))</c> with
/// Gemma RMSNorm (<c>1 + w</c>), per-head Q/K RMSNorm, split-half RoPE, GQA, scale <c>query_pre_attn_scalar^-½</c> and a
/// tanh-GELU gated MLP. Attention is bidirectional: sliding layers let a token see the previous
/// <c>(window+1)/2</c> and the next <c>window/2+1</c> positions (itself included), full layers see everything.</summary>
public sealed unsafe class T5Gemma2TextEncoder : IDisposable
{
    private readonly T5Gemma2TextEncoderConfig _cfg;
    private Tensor? _embed, _eoi, _finalNorm;
    private Layer[] _layers = [];
    private int _disposed;

    private sealed record Layer(Tensor PreAttn, Tensor PostAttn, Tensor PreFfn, Tensor PostFfn, Tensor QNorm, Tensor KNorm,
        Tensor Wq, Tensor Wk, Tensor Wv, Tensor Wo, Tensor Gate, Tensor Up, Tensor Down);

    public T5Gemma2TextEncoder(T5Gemma2TextEncoderConfig cfg) { _cfg = cfg; }

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

    private static Tensor OnePlus(Tensor w)
    {
        Tensor f = WhisperOps.EnsureF32(w);
        Tensor o = new(f.Shape, DType.F32);
        float* s = (float*)f.DataPointer, d = (float*)o.DataPointer;
        for (long i = 0; i < f.ElementCount; i++) d[i] = 1f + s[i];
        return o;
    }

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
        float scale = 1f / MathF.Sqrt(_cfg.QueryPreAttnScalar);
        int left = (_cfg.SlidingWindow + 1) / 2, right = _cfg.SlidingWindow / 2 + 1;

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

            float[] ctx = new float[n * heads * hd];
            float[] sc = new float[n];
            int group = heads / kvHeads;
            for (int head = 0; head < heads; head++)
            {
                int kv = head / group;
                for (int i = 0; i < n; i++)
                {
                    int lo = full ? 0 : Math.Max(0, i - left + 1), hi = full ? n - 1 : Math.Min(n - 1, i + right - 1);
                    float max = float.NegativeInfinity;
                    for (int j = lo; j <= hi; j++)
                    {
                        float s = 0f;
                        for (int e = 0; e < hd; e++) s += q[(i * heads + head) * hd + e] * k[(j * kvHeads + kv) * hd + e];
                        sc[j] = s * scale; max = MathF.Max(max, sc[j]);
                    }
                    float sum = 0f;
                    for (int j = lo; j <= hi; j++) { sc[j] = MathF.Exp(sc[j] - max); sum += sc[j]; }
                    for (int e = 0; e < hd; e++)
                    {
                        float o = 0f;
                        for (int j = lo; j <= hi; j++) o += sc[j] * v[(j * kvHeads + kv) * hd + e];
                        ctx[(i * heads + head) * hd + e] = o / sum;
                    }
                }
            }
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

    private float[] Norm(IBackend backend, float[] x, Tensor weight, int rows, int dim)
        => DacOps.RmsNorm(backend, x, weight, rows, dim, _cfg.RmsNormEps);

    private static float GeluTanh(float x)
        => 0.5f * x * (1f + MathF.Tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));

    private static float[,] RopeTable(int n, int hd, float theta, float linearFactor)
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
    private static void ApplyRope(float[] v, float[,] table, int n, int heads, int hd)
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
        GC.SuppressFinalize(this);
    }
}
