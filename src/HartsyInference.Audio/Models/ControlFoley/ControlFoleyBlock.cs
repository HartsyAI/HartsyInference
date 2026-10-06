using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Port of <c>MMDitSingleBlock</c>: adaLN-modulated self-attention (per-head RMS-normed q/k, interleaved RoPE)
/// followed by a gated SwiGLU feed-forward. A pre-only block stops after the q/k/v projection and keeps only the
/// shift/scale modulation. Activations are updated in place.</summary>
internal sealed unsafe class ControlFoleyBlock
{
    private readonly int _heads;
    private readonly Tensor _qkvW, _qkvB, _qNorm, _kNorm, _adaW, _adaB;
    private readonly Tensor? _linear1W, _linear1B, _w1, _w2, _w3;

    private ControlFoleyBlock(int heads, bool preOnly, IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _heads = heads;
        PreOnly = preOnly;
        _qkvW = F(w, $"{p}.attn.qkv.weight");
        _qkvB = F(w, $"{p}.attn.qkv.bias");
        _qNorm = F(w, $"{p}.attn.q_norm.weight");
        _kNorm = F(w, $"{p}.attn.k_norm.weight");
        _adaW = F(w, $"{p}.adaLN_modulation.1.weight");
        _adaB = F(w, $"{p}.adaLN_modulation.1.bias");
        if (!preOnly)
        {
            _linear1W = F(w, $"{p}.linear1.weight");
            _linear1B = F(w, $"{p}.linear1.bias");
            _w1 = F(w, $"{p}.ffn.w1.weight");
            _w2 = F(w, $"{p}.ffn.w2.weight");
            _w3 = F(w, $"{p}.ffn.w3.weight");
        }

        int dim = (int)_qkvW.Shape[1];
        int expectedMod = (preOnly ? 2 : 6) * dim;
        if (_qkvW.Shape[0] != 3L * dim || _adaW.Shape[0] != expectedMod || dim % heads != 0 || _qNorm.Shape[0] != dim / heads)
        {
            throw new InvalidDataException($"Block '{p}' weights do not match {heads} heads at dim {dim} (preOnly={preOnly}).");
        }
    }

    /// <summary>True when only the attention-input half of the block exists.</summary>
    internal bool PreOnly { get; }

    internal static ControlFoleyBlock Load(IReadOnlyDictionary<string, Tensor> w, string prefix, int heads, bool preOnly)
        => new(heads, preOnly, w, prefix);

    private static Tensor F(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue(key, out Tensor? t) ? WhisperOps.EnsureF32(t) : throw new KeyNotFoundException($"Missing ControlFoley weight '{key}'.");

    internal IEnumerable<Tensor> Weights()
    {
        foreach (Tensor? t in new[] { _qkvW, _qkvB, _qNorm, _kNorm, _adaW, _adaB, _linear1W, _linear1B, _w1, _w2, _w3 })
        {
            if (t is not null)
            {
                yield return t;
            }
        }
    }

    /// <summary>Queries, keys and values (<c>[B, H, T, headDim]</c>) plus the adaLN modulation they were built with.</summary>
    internal sealed class PreAttention(Tensor q, Tensor k, Tensor v, Tensor modulation) : IDisposable
    {
        internal Tensor Q { get; } = q;
        internal Tensor K { get; } = k;
        internal Tensor V { get; } = v;
        internal Tensor Modulation { get; } = modulation;

        public void Dispose()
        {
            Q.Dispose();
            K.Dispose();
            V.Dispose();
            Modulation.Dispose();
        }
    }

    internal PreAttention Pre(IBackend backend, Tensor x, Tensor c, ControlFoleyRope? rope)
    {
        using Tensor activated = ControlFoleyOps.Silu(backend, c);
        Tensor modulation = ControlFoleyOps.Linear(backend, activated, _adaW, _adaB);
        using Tensor modulated = ControlFoleyOps.NormModulate(backend, x, ControlFoleyOps.ChunkOf(modulation, 0, PreOnly ? 2 : 6),
            ControlFoleyOps.ChunkOf(modulation, 1, PreOnly ? 2 : 6));

        int b = (int)x.Shape[0], t = (int)x.Shape[1], d = (int)x.Shape[2], hd = d / _heads;
        using Tensor qkv = ControlFoleyOps.Linear(backend, modulated, _qkvW, _qkvB);
        TensorShape head = new(b, _heads, t, hd);
        Tensor q = new(head, DType.F32), k = new(head, DType.F32), v = new(head, DType.F32);
        float* src = (float*)qkv.DataPointer, qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer;
        // The fused projection is laid out (head, dim, {q,k,v}): the official 'b n (h d j) -> b h n d j'.
        for (int bi = 0; bi < b; bi++)
        {
            for (int ti = 0; ti < t; ti++)
            {
                float* row = src + ((long)bi * t + ti) * 3 * d;
                for (int h = 0; h < _heads; h++)
                {
                    long at = (((long)bi * _heads + h) * t + ti) * hd;
                    for (int e = 0; e < hd; e++)
                    {
                        float* j = row + (h * hd + e) * 3;
                        qp[at + e] = j[0];
                        kp[at + e] = j[1];
                        vp[at + e] = j[2];
                    }
                }
            }
        }

        Tensor qn = Normalize(backend, q, _qNorm), kn = Normalize(backend, k, _kNorm);
        rope?.Apply(qn);
        rope?.Apply(kn);
        return new PreAttention(qn, kn, v, modulation);
    }

    private static Tensor Normalize(IBackend backend, Tensor x, Tensor weight)
    {
        Tensor o = new(x.Shape, DType.F32);
        backend.RmsNorm(o, x, weight, ControlFoleyOps.RmsNormEps);
        x.Dispose();
        return o;
    }

    /// <summary>Applies the attention output and the feed-forward to <paramref name="x"/> in place.</summary>
    internal void Post(IBackend backend, Tensor x, Tensor attnOut, Tensor modulation)
    {
        if (PreOnly)
        {
            return;
        }

        using (Tensor projected = ControlFoleyOps.Project(backend, attnOut, _linear1W!, _linear1B))
        {
            ControlFoleyOps.GatedAdd(x, projected, ControlFoleyOps.ChunkOf(modulation, 2, 6));
        }

        using Tensor r = ControlFoleyOps.NormModulate(backend, x, ControlFoleyOps.ChunkOf(modulation, 3, 6),
            ControlFoleyOps.ChunkOf(modulation, 4, 6));
        using Tensor ffn = ControlFoleyOps.SwiGlu(backend, r, _w1!, _w2!, _w3!);
        ControlFoleyOps.GatedAdd(x, ffn, ControlFoleyOps.ChunkOf(modulation, 5, 6));
    }

    /// <summary>Scaled dot-product attention over <c>[B, H, N, headDim]</c> tensors.</summary>
    internal static Tensor Attend(IBackend backend, Tensor q, Tensor k, Tensor v)
    {
        Tensor o = new(q.Shape, DType.F32);
        backend.ScaledDotProductAttention(o, q, k, v, null, 1f / MathF.Sqrt(q.Shape[3]));
        return o;
    }

    /// <summary>Tokens <c>[offset, offset + length)</c> of a head-major attention output as <c>[B, length, H * headDim]</c>.</summary>
    internal static Tensor TokenSlice(Tensor attn, int offset, int length)
    {
        int b = (int)attn.Shape[0], h = (int)attn.Shape[1], n = (int)attn.Shape[2], hd = (int)attn.Shape[3];
        Tensor o = ControlFoleyOps.New(b, length, (long)h * hd);
        float* ap = (float*)attn.DataPointer, op = (float*)o.DataPointer;
        for (int bi = 0; bi < b; bi++)
        {
            for (int ti = 0; ti < length; ti++)
            {
                for (int hi = 0; hi < h; hi++)
                {
                    Buffer.MemoryCopy(ap + (((long)bi * h + hi) * n + offset + ti) * hd,
                        op + ((long)bi * length + ti) * h * hd + hi * hd, hd * 4L, hd * 4L);
                }
            }
        }

        return o;
    }

    /// <summary>Single-stream forward (fused blocks).</summary>
    internal void Forward(IBackend backend, Tensor x, Tensor c, ControlFoleyRope rope)
    {
        using PreAttention pre = Pre(backend, x, c, rope);
        using Tensor attn = Attend(backend, pre.Q, pre.K, pre.V);
        using Tensor merged = TokenSlice(attn, 0, (int)x.Shape[1]);
        Post(backend, x, merged, pre.Modulation);
    }
}
