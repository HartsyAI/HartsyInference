using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Wan2.2-S2V audio injector — a faithful port of ComfyUI's <c>AudioInjector_WAN</c> with
/// <c>enable_adain=True, adain_mode="attn_norm"</c>. After each injected DiT block, the first <c>seqLen</c> (main
/// video) tokens are regrouped per latent frame; an AdaLayerNorm (no-affine LN at eps 1e-5, shift/scale from
/// <c>linear(silu(globalAudioToken))</c> — shift FIRST in the chunk) modulates them, then each frame's tokens
/// cross-attend (<c>WanT2VCrossAttention</c>: full-dim RMSNorm QK-norm, per-frame KV = that frame's
/// <c>AudioTokens+1</c> local audio tokens) and the result is added back residually. Reference/motion tokens beyond
/// <c>seqLen</c> are untouched. Fully GPU-resident (every step is an <see cref="IBackend"/> op — no mid-forward
/// <c>DataPointer</c> reads, which would each cost a stream-sync + D2H eviction ×24 per denoise step); the AdaLN
/// linear is split into shift/scale halves at load so the chunk needs no on-device split. B=1.</summary>
public sealed unsafe class WanS2VAudioInjector
{
    // torch nn.LayerNorm default eps — AdaLayerNorm is constructed without an explicit norm_eps in the reference.
    private const float AdaLnEps = 1e-5f;

    private readonly int _dim, _heads, _headDim;
    private readonly float _eps;
    private readonly Inj[] _inj;

    private sealed class Inj
    {
        public Tensor? QW, QB, KW, KB, VW, VB, OW, OB, NQ, NK;
        public Tensor? AdaShiftW, AdaShiftB, AdaScaleW, AdaScaleB;   // linear split into the chunk(2) halves
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
    /// are no-affine and carry no weights). The AdaLN linear (out = 2·dim) is split row-wise into its
    /// <c>chunk(2, dim=1)</c> halves — [shift, scale] — so each half runs as a plain Linear on the GPU.</summary>
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
            Tensor adaW = w[$"audio_injector.injector_adain_layers.{i}.linear.weight"];
            w.TryGetValue($"audio_injector.injector_adain_layers.{i}.linear.bias", out Tensor? adaB);
            (inj.AdaShiftW, inj.AdaScaleW) = SplitRows(adaW, _dim);
            if (adaB is not null) (inj.AdaShiftB, inj.AdaScaleB) = SplitRows(adaB, _dim);
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Inj inj in _inj)
            foreach (Tensor? t in new[] { inj.QW, inj.QB, inj.KW, inj.KB, inj.VW, inj.VB, inj.OW, inj.OB, inj.NQ, inj.NK,
                inj.AdaShiftW, inj.AdaShiftB, inj.AdaScaleW, inj.AdaScaleB })
                if (t is not null) yield return t;
    }

    /// <summary>Residually injects audio into the first <paramref name="seqLen"/> rows of <paramref name="hidden"/>
    /// <c>[S, dim]</c> and returns a FRESH tensor (the input is left untouched — in-place host mutation of an
    /// op output would desync/evict the GPU activation cache). <paramref name="audioLocal"/> is <c>[T, nTok, dim]</c>,
    /// <paramref name="audioGlobal"/> <c>[T, 1, dim]</c>; <c>T</c> must divide <paramref name="seqLen"/>. The caller
    /// disposes the returned tensor (and the consumed <paramref name="hidden"/>).</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, int injectorIdx, Tensor audioLocal, Tensor audioGlobal, int seqLen)
    {
        Inj inj = _inj[injectorIdx];
        int t = (int)audioLocal.Shape[0], nTok = (int)audioLocal.Shape[1];
        if (seqLen % t != 0)
            throw new ArgumentException($"audio injector needs T|seqLen; T={t} does not divide seqLen={seqLen}.");
        int n = seqLen / t;
        int totalS = (int)(hidden.ElementCount / _dim);

        // AdaLN shift/scale per frame from the global audio token (chunk order: shift first).
        Tensor gAct = new Tensor(new TensorShape(t, _dim), DType.F32);
        backend.Silu(gAct, audioGlobal);   // audioGlobal is contiguous [t, 1, dim] = [t, dim]
        Tensor shift = new Tensor(new TensorShape(t, _dim), DType.F32);
        backend.Linear(shift, gAct, inj.AdaShiftW!, inj.AdaShiftB);
        Tensor scale1 = new Tensor(new TensorShape(t, _dim), DType.F32);
        backend.Linear(scale1, gAct, inj.AdaScaleW!, inj.AdaScaleB);
        gAct.Dispose();
        Tensor scale1p = new Tensor(scale1.Shape, DType.F32);
        backend.AddScalar(scale1p, scale1, 1f);   // (1 + scale)
        scale1.Dispose();

        // No-affine LN (eps 1e-5) over the main tokens, then the per-frame affine — shaped [t, n, dim] so the
        // broadcast op maps frame ti's [shift, scale] row over its n tokens.
        Tensor top = new Tensor(new TensorShape(t, n, _dim), DType.F32);
        backend.SliceRows(top, hidden, 0);
        Tensor normed = new Tensor(top.Shape, DType.F32);
        backend.LayerNormNoAffine(normed, top, AdaLnEps);
        Tensor adain = new Tensor(top.Shape, DType.F32);
        backend.AffineBroadcastLastDim(adain, normed, scale1p, shift);
        normed.Dispose(); scale1p.Dispose(); shift.Dispose();

        // Q from the modulated video tokens; K/V from the audio tokens; full-dim RMSNorm QK-norm.
        Tensor q = new Tensor(new TensorShape(seqLen, _dim), DType.F32); backend.Linear(q, adain, inj.QW!, inj.QB); adain.Dispose();
        Tensor qn = new Tensor(q.Shape, DType.F32); backend.RmsNorm(qn, q, inj.NQ!, _eps); q.Dispose();
        int kvRows = t * nTok;
        Tensor k = new Tensor(new TensorShape(kvRows, _dim), DType.F32); backend.Linear(k, audioLocal, inj.KW!, inj.KB);
        Tensor kn = new Tensor(k.Shape, DType.F32); backend.RmsNorm(kn, k, inj.NK!, _eps); k.Dispose();
        Tensor v = new Tensor(new TensorShape(kvRows, _dim), DType.F32); backend.Linear(v, audioLocal, inj.VW!, inj.VB);

        // Per-frame attention: frame ti's n video tokens attend only to its nTok audio tokens. All slicing stays
        // on-device; per-frame outputs are concatenated once at the end.
        float scale = 1f / MathF.Sqrt(_headDim);
        Tensor[] frameOuts = new Tensor[t];
        for (int ti = 0; ti < t; ti++)
        {
            Tensor qF = new Tensor(new TensorShape(n, _dim), DType.F32); backend.SliceRows(qF, qn, ti * n);
            Tensor kF = new Tensor(new TensorShape(nTok, _dim), DType.F32); backend.SliceRows(kF, kn, ti * nTok);
            Tensor vF = new Tensor(new TensorShape(nTok, _dim), DType.F32); backend.SliceRows(vF, v, ti * nTok);
            Tensor qMh = ToBhsd(backend, qF, n); qF.Dispose();
            Tensor kMh = ToBhsd(backend, kF, nTok); kF.Dispose();
            Tensor vMh = ToBhsd(backend, vF, nTok); vF.Dispose();
            Tensor attn = new Tensor(new TensorShape(1, _heads, n, _headDim), DType.F32);
            backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, null, scale);
            qMh.Dispose(); kMh.Dispose(); vMh.Dispose();
            frameOuts[ti] = new Tensor(new TensorShape(n, _dim), DType.F32);
            backend.Permute0213(frameOuts[ti], attn, _heads, n, _headDim);
            attn.Dispose();
        }
        qn.Dispose(); kn.Dispose(); v.Dispose();

        Tensor attnFlat = new Tensor(new TensorShape(seqLen, _dim), DType.F32);
        backend.Concat(attnFlat, frameOuts, 0);
        foreach (Tensor f in frameOuts) f.Dispose();

        Tensor residual = new Tensor(new TensorShape(seqLen, _dim), DType.F32);
        backend.Linear(residual, attnFlat, inj.OW!, inj.OB);
        attnFlat.Dispose();

        // out = hidden with the residual added to the first seqLen rows; trailing (reference) rows pass through.
        Tensor topAdded = new Tensor(new TensorShape(seqLen, _dim), DType.F32);
        backend.Add(topAdded, top, residual);
        top.Dispose(); residual.Dispose();
        if (totalS == seqLen) return topAdded;

        Tensor rest = new Tensor(new TensorShape(totalS - seqLen, _dim), DType.F32);
        backend.SliceRows(rest, hidden, seqLen);
        Tensor outFull = new Tensor(new TensorShape(totalS, _dim), DType.F32);
        backend.Concat(outFull, [topAdded, rest], 0);
        topAdded.Dispose(); rest.Dispose();
        return outFull;
    }

    private Tensor ToBhsd(IBackend backend, Tensor x, int s)
    {
        Tensor o = new Tensor(new TensorShape(1, _heads, s, _headDim), DType.F32);
        backend.Permute0213(o, x, s, _heads, _headDim);
        return o;
    }

    /// <summary>Row-wise split of a <c>[2·rows, …]</c> weight (or <c>[2·rows]</c> bias) into its two halves —
    /// dtype-agnostic byte copy, preserving the fp8 scale factor.</summary>
    private static (Tensor Top, Tensor Bottom) SplitRows(Tensor x, int rows)
    {
        long perRowBytes = x.ElementCount / (2L * rows) * x.DType.SizeInBytes;
        long halfBytes = rows * perRowBytes;
        long[] halfShape = x.Shape.Rank == 1 ? [rows] : [rows, x.Shape[1]];
        Tensor top = new Tensor(new TensorShape(halfShape), x.DType);
        Tensor bottom = new Tensor(new TensorShape(halfShape), x.DType);
        Buffer.MemoryCopy((byte*)x.DataPointer, (byte*)top.DataPointer, halfBytes, halfBytes);
        Buffer.MemoryCopy((byte*)x.DataPointer + halfBytes, (byte*)bottom.DataPointer, halfBytes, halfBytes);
        top.Fp8ScaleFactor = x.Fp8ScaleFactor;
        bottom.Fp8ScaleFactor = x.Fp8ScaleFactor;
        return (top, bottom);
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key) { Tensor t = w[key]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }
}
