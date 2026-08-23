using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Wan I2V CLIP image embedder (<c>MLPProj</c> / diffusers <c>condition_embedder.image_embedder</c>):
/// optional pos-embed add → FP32 LayerNorm → gelu FFN (net.0.proj → net.2) → FP32 LayerNorm, mapping CLIP features
/// <c>[seqImg, imageDim]</c> → <c>[seqImg, dim]</c>. Hoisted from <see cref="WanVideoTransformer"/> so the Animate DiT
/// shares it (both carry the same <c>img_emb</c> module).</summary>
public sealed unsafe class WanImageEmbedder
{
    private readonly float _eps;
    private Tensor? _norm1W, _norm1B, _ff0W, _ff0B, _ff2W, _ff2B, _norm2W, _norm2B, _posEmbed;

    public WanImageEmbedder(float eps = 1e-6f)
    {
        _eps = eps;
    }

    /// <summary>True once the image-embedder weights were found and loaded.</summary>
    public bool IsLoaded => _ff0W is not null;

    /// <summary>Loads the diffusers-named image-embedder weights if present; returns false when the checkpoint has none.</summary>
    public bool TryLoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        if (!w.TryGetValue("condition_embedder.image_embedder.norm1.weight", out Tensor? n1))
            return false;
        _norm1W = TensorCasts.EnsureF32(n1); _norm1B = TensorCasts.LoadF32Opt(w, "condition_embedder.image_embedder.norm1.bias");
        _ff0W = w["condition_embedder.image_embedder.ff.net.0.proj.weight"]; w.TryGetValue("condition_embedder.image_embedder.ff.net.0.proj.bias", out _ff0B);
        _ff2W = w["condition_embedder.image_embedder.ff.net.2.weight"]; w.TryGetValue("condition_embedder.image_embedder.ff.net.2.bias", out _ff2B);
        _norm2W = TensorCasts.LoadF32(w, "condition_embedder.image_embedder.norm2.weight"); _norm2B = TensorCasts.LoadF32Opt(w, "condition_embedder.image_embedder.norm2.bias");
        _posEmbed = TensorCasts.LoadF32Opt(w, "condition_embedder.image_embedder.pos_embed");
        return true;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _norm1W, _norm1B, _ff0W, _ff0B, _ff2W, _ff2B, _norm2W, _norm2B, _posEmbed })
            if (t is not null) yield return t;
    }

    /// <summary>Projects CLIP image embeds <c>[seqImg, imageDim]</c> (rank-2 or rank-3 with batch 1) → <c>[seqImg, dim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor imageEmbeds, int dim)
    {
        // The real CLIP-ViT-H encoder emits rank-3 [1, seq, imageDim] (257×1280); the synthetic test used rank-2.
        // Derive dims from the last axis + element count so both work (same fix class as WanDitOps.TextEmbed).
        int imageDim = (int)imageEmbeds.Shape[imageEmbeds.Shape.Rank - 1];
        int seq = (int)(imageEmbeds.ElementCount / imageDim);

        Tensor x = imageEmbeds;
        bool ownX = false;
        int peRank = _posEmbed?.Shape.Rank ?? 0;
        if (_posEmbed is not null && (int)_posEmbed.Shape[peRank - 2] == seq && (int)_posEmbed.Shape[peRank - 1] == imageDim)
        {
            x = new Tensor(new TensorShape(seq, imageDim), DType.F32);
            ownX = true;
            float* sp = (float*)imageEmbeds.DataPointer, pp = (float*)_posEmbed.DataPointer, xp = (float*)x.DataPointer;
            long n = (long)seq * imageDim;
            for (long i = 0; i < n; i++) xp[i] = sp[i] + pp[i];
        }

        Tensor n1 = LayerNormAffine(x, _norm1W!, _norm1B, seq, imageDim);
        if (ownX) x.Dispose();
        int inner = (int)_ff0W!.Shape[0];
        Tensor h0 = new Tensor(new TensorShape(seq, inner), DType.F32); backend.Linear(h0, n1, _ff0W!, _ff0B); n1.Dispose();
        Tensor act = new Tensor(h0.Shape, DType.F32); backend.Gelu(act, h0); h0.Dispose();
        Tensor h2 = new Tensor(new TensorShape(seq, dim), DType.F32); backend.Linear(h2, act, _ff2W!, _ff2B); act.Dispose();
        Tensor outT = LayerNormAffine(h2, _norm2W!, _norm2B, seq, dim); h2.Dispose();
        return outT;
    }

    /// <summary>FP32 affine LayerNorm over the last dim.</summary>
    private Tensor LayerNormAffine(Tensor x, Tensor weight, Tensor? bias, int s, int d)
    {
        Tensor o = new Tensor(new TensorShape(s, d), DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer, wp = (float*)weight.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;
        for (int i = 0; i < s; i++)
        {
            long off = (long)i * d;
            double mean = 0; for (int k = 0; k < d; k++) mean += xp[off + k]; mean /= d;
            double var = 0; for (int k = 0; k < d; k++) { double dd = xp[off + k] - mean; var += dd * dd; }
            float inv = 1f / MathF.Sqrt((float)(var / d) + _eps);
            for (int k = 0; k < d; k++)
                op[off + k] = (float)((xp[off + k] - mean) * inv) * wp[k] + (bp is null ? 0f : bp[k]);
        }
        return o;
    }

    /// <summary>Clears the weight references (the loader owns the underlying memory).</summary>
    public void Clear()
    {
        _norm1W = _norm1B = _ff0W = _ff0B = _ff2W = _ff2B = _norm2W = _norm2B = _posEmbed = null;
    }

}
