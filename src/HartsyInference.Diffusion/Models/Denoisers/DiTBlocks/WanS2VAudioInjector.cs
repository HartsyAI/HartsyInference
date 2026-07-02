using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Wan2.2-S2V audio injector — a faithful port of ComfyUI's <c>AudioInjector_WAN</c> with
/// <c>enable_adain=True, adain_mode="attn_norm"</c>. After each injected DiT block, the first <c>seqLen</c> (main
/// video) tokens are regrouped per latent frame; an AdaLayerNorm (no-affine LN at eps 1e-5, shift/scale from
/// <c>linear(silu(globalAudioToken))</c> — shift FIRST in the chunk) modulates them, then each frame's tokens
/// cross-attend (<c>WanT2VCrossAttention</c>: full-dim RMSNorm QK-norm, per-frame KV = that frame's
/// <c>AudioTokens+1</c> local audio tokens) and the result is added back residually. Reference/motion tokens beyond
/// <c>seqLen</c> are untouched. B=1; CPU-glue slicing (GPU-resident rewrite is a later perf pass).</summary>
public sealed unsafe class WanS2VAudioInjector
{
    // torch nn.LayerNorm default eps — AdaLayerNorm is constructed without an explicit norm_eps in the reference.
    private const float AdaLnEps = 1e-5f;

    private readonly int _dim, _heads, _headDim;
    private readonly float _eps;
    private readonly Inj[] _inj;

    private sealed class Inj
    {
        public Tensor? QW, QB, KW, KB, VW, VB, OW, OB, NQ, NK, AdaW, AdaB;
    }

    public WanS2VAudioInjector(int count, int dim, int heads, int headDim, float eps = 1e-6f)
    {
        _dim = dim;
        _heads = heads;
        _headDim = headDim;
        _eps = eps;
        _inj = new Inj[count];
        for (int i = 0; i < count; i++) _inj[i] = new Inj();
    }

    public int Count => _inj.Length;

    /// <summary>Loads <c>audio_injector.injector.{i}.*</c> + <c>audio_injector.injector_adain_layers.{i}.linear.*</c>
    /// (post-converter names — the converter's rule table leaves them untouched; the injector's pre-norm LayerNorms
    /// are no-affine and carry no weights).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        for (int i = 0; i < _inj.Length; i++)
        {
            string p = $"audio_injector.injector.{i}";
            Inj inj = _inj[i];
            inj.QW = w[$"{p}.q.weight"]; w.TryGetValue($"{p}.q.bias", out inj.QB);
            inj.KW = w[$"{p}.k.weight"]; w.TryGetValue($"{p}.k.bias", out inj.KB);
            inj.VW = w[$"{p}.v.weight"]; w.TryGetValue($"{p}.v.bias", out inj.VB);
            inj.OW = w[$"{p}.o.weight"]; w.TryGetValue($"{p}.o.bias", out inj.OB);
            inj.NQ = LoadF32(w, $"{p}.norm_q.weight");
            inj.NK = LoadF32(w, $"{p}.norm_k.weight");
            inj.AdaW = w[$"audio_injector.injector_adain_layers.{i}.linear.weight"];
            w.TryGetValue($"audio_injector.injector_adain_layers.{i}.linear.bias", out inj.AdaB);
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Inj inj in _inj)
            foreach (Tensor? t in new[] { inj.QW, inj.QB, inj.KW, inj.KB, inj.VW, inj.VB, inj.OW, inj.OB, inj.NQ, inj.NK, inj.AdaW, inj.AdaB })
                if (t is not null) yield return t;
    }

    /// <summary>Residually injects audio into the first <paramref name="seqLen"/> rows of <paramref name="hidden"/>
    /// <c>[S, dim]</c> in place. <paramref name="audioLocal"/> is <c>[T, nTok, dim]</c>, <paramref name="audioGlobal"/>
    /// <c>[T, 1, dim]</c>; <c>T</c> must divide <paramref name="seqLen"/>.</summary>
    public void Forward(IBackend backend, Tensor hidden, int injectorIdx, Tensor audioLocal, Tensor audioGlobal, int seqLen)
    {
        Inj inj = _inj[injectorIdx];
        int t = (int)audioLocal.Shape[0], nTok = (int)audioLocal.Shape[1];
        if (seqLen % t != 0)
            throw new ArgumentException($"audio injector needs T|seqLen; T={t} does not divide seqLen={seqLen}.");
        int n = seqLen / t;

        // AdaLayerNorm shift/scale per frame from the global audio token: linear(silu(g)) → chunk(2) = [shift, scale].
        Tensor gAct = new Tensor(new TensorShape(t, _dim), DType.F32);
        backend.Silu(gAct, audioGlobal);   // audioGlobal is contiguous [t, 1, dim] = [t, dim]
        Tensor adaOut = new Tensor(new TensorShape(t, 2 * _dim), DType.F32);
        backend.Linear(adaOut, gAct, inj.AdaW!, inj.AdaB);
        gAct.Dispose();

        // No-affine LN (eps 1e-5) over the main tokens, then the per-frame (1+scale)·x + shift affine.
        Tensor adain = new Tensor(new TensorShape(seqLen, _dim), DType.F32);
        float* hp = (float*)hidden.DataPointer, ap = (float*)adain.DataPointer, mp = (float*)adaOut.DataPointer;
        for (int i = 0; i < seqLen; i++)
        {
            long off = (long)i * _dim;
            long fOff = (long)(i / n) * 2 * _dim;   // frame's [shift(dim), scale(dim)] row
            double mean = 0; for (int d = 0; d < _dim; d++) mean += hp[off + d]; mean /= _dim;
            double var = 0; for (int d = 0; d < _dim; d++) { double dd = hp[off + d] - mean; var += dd * dd; }
            float inv = 1f / MathF.Sqrt((float)(var / _dim) + AdaLnEps);
            for (int d = 0; d < _dim; d++)
            {
                float normed = (float)((hp[off + d] - mean) * inv);
                ap[off + d] = normed * (1f + mp[fOff + _dim + d]) + mp[fOff + d];
            }
        }
        adaOut.Dispose();

        // Q from the modulated video tokens; K/V from the audio tokens; full-dim RMSNorm QK-norm.
        Tensor q = new Tensor(new TensorShape(seqLen, _dim), DType.F32); backend.Linear(q, adain, inj.QW!, inj.QB); adain.Dispose();
        Tensor qn = new Tensor(q.Shape, DType.F32); backend.RmsNorm(qn, q, inj.NQ!, _eps); q.Dispose();
        int kvRows = t * nTok;
        Tensor k = new Tensor(new TensorShape(kvRows, _dim), DType.F32); backend.Linear(k, audioLocal, inj.KW!, inj.KB);
        Tensor kn = new Tensor(k.Shape, DType.F32); backend.RmsNorm(kn, k, inj.NK!, _eps); k.Dispose();
        Tensor v = new Tensor(new TensorShape(kvRows, _dim), DType.F32); backend.Linear(v, audioLocal, inj.VW!, inj.VB);

        // Per-frame attention: frame ti's n video tokens attend only to its nTok audio tokens.
        Tensor attnFlat = new Tensor(new TensorShape(seqLen, _dim), DType.F32);
        float scale = 1f / MathF.Sqrt(_headDim);
        for (int ti = 0; ti < t; ti++)
        {
            Tensor qF = SliceRows(qn, ti * n, n);
            Tensor kF = SliceRows(kn, ti * nTok, nTok);
            Tensor vF = SliceRows(v, ti * nTok, nTok);
            Tensor qMh = ToBhsd(backend, qF, n); qF.Dispose();
            Tensor kMh = ToBhsd(backend, kF, nTok); kF.Dispose();
            Tensor vMh = ToBhsd(backend, vF, nTok); vF.Dispose();
            Tensor attn = new Tensor(new TensorShape(1, _heads, n, _headDim), DType.F32);
            backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, null, scale);
            qMh.Dispose(); kMh.Dispose(); vMh.Dispose();
            Tensor frameOut = new Tensor(new TensorShape(n, _dim), DType.F32);
            backend.Permute0213(frameOut, attn, _heads, n, _headDim);
            attn.Dispose();
            Buffer.MemoryCopy((float*)frameOut.DataPointer, (float*)attnFlat.DataPointer + (long)ti * n * _dim,
                (long)n * _dim * 4, (long)n * _dim * 4);
            frameOut.Dispose();
        }
        qn.Dispose(); kn.Dispose(); v.Dispose();

        Tensor residual = new Tensor(new TensorShape(seqLen, _dim), DType.F32);
        backend.Linear(residual, attnFlat, inj.OW!, inj.OB);
        attnFlat.Dispose();

        float* rp = (float*)residual.DataPointer;
        long total = (long)seqLen * _dim;
        for (long i = 0; i < total; i++) hp[i] += rp[i];
        residual.Dispose();
    }

    private Tensor SliceRows(Tensor x, int start, int len)
    {
        Tensor o = new Tensor(new TensorShape(len, _dim), DType.F32);
        Buffer.MemoryCopy((float*)x.DataPointer + (long)start * _dim, (float*)o.DataPointer,
            (long)len * _dim * 4, (long)len * _dim * 4);
        return o;
    }

    private Tensor ToBhsd(IBackend backend, Tensor x, int s)
    {
        Tensor o = new Tensor(new TensorShape(1, _heads, s, _headDim), DType.F32);
        backend.Permute0213(o, x, s, _heads, _headDim);
        return o;
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key) { Tensor t = w[key]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }
}
