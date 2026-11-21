using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Wan-Animate face-adapter block (<c>FaceBlock</c> in ComfyUI <c>comfy/ldm/wan/model_animate.py</c>): the
/// latent stream <c>[S, dim]</c> cross-attends to per-frame face features <c>[T, N, dim]</c>, temporally aligned —
/// the S tokens split into T contiguous frame groups (T must divide S) and group <c>t</c> attends only to frame
/// <c>t</c>'s N motion tokens. No-affine LayerNorm pre-norms on both streams (<c>pre_norm_feat</c> /
/// <c>pre_norm_motion</c>, no weights); K and V come from the fused <c>linear1_kv</c> (K-major, split into separate
/// K/V weights at load — identical math), Q from <c>linear1_q</c>; per-head-dim affine RMSNorm on Q and K only
/// (<c>q_norm</c>/<c>k_norm</c>, eps 1e-6, V un-normed); output projection <c>linear2</c>. The caller adds the result
/// residually. Also loads the legacy <c>to_q/to_k/to_v/to_out</c> + <c>norm_q/norm_k</c> layout used by the
/// <see cref="WanS2VTransformer"/> audio injector. B=1.</summary>
public sealed unsafe class WanAnimateFaceBlock(int dim, int heads, int headDim, float eps = 1e-6f)
{
    private readonly int _dim = dim, _heads = heads, _headDim = headDim;
    private readonly float _eps = eps;
    private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _nq, _nk;
    private AttentionLayer? _layer;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        if (w.TryGetValue($"{p}.linear1_kv.weight", out Tensor? kvW))
        {
            // ComfyUI FaceBlock layout: fused KV [2·dim, dim] ("(K H D)" is K-major → rows 0..dim-1 = K, dim.. = V).
            (_kW, _vW) = SplitRows(kvW, _dim);
            if (w.TryGetValue($"{p}.linear1_kv.bias", out Tensor? kvB)) (_kB, _vB) = SplitRows1d(kvB, _dim);
            _qW = w[$"{p}.linear1_q.weight"]; w.TryGetValue($"{p}.linear1_q.bias", out _qB);
            _oW = w[$"{p}.linear2.weight"]; w.TryGetValue($"{p}.linear2.bias", out _oB);
            _nq = TensorCasts.LoadF32(w, $"{p}.q_norm.weight");
            _nk = TensorCasts.LoadF32(w, $"{p}.k_norm.weight");
        }
        else
        {
            // Legacy layout (S2V audio injector structural port).
            _qW = w[$"{p}.to_q.weight"]; w.TryGetValue($"{p}.to_q.bias", out _qB);
            _kW = w[$"{p}.to_k.weight"]; w.TryGetValue($"{p}.to_k.bias", out _kB);
            _vW = w[$"{p}.to_v.weight"]; w.TryGetValue($"{p}.to_v.bias", out _vB);
            _oW = w[$"{p}.to_out.weight"]; w.TryGetValue($"{p}.to_out.bias", out _oB);
            _nq = TensorCasts.LoadF32(w, $"{p}.norm_q.weight");
            _nk = TensorCasts.LoadF32(w, $"{p}.norm_k.weight");
        }
        // The projections, the per-head RMS norm on Q and K, the attention and the output projection are the shared layer's.
        _layer = new AttentionLayer(
            new AttentionLayerConfig { Heads = _heads, KvHeads = _heads, HeadDim = _headDim, QkNorm = AttentionQkNorm.PerHead, ComputeDType = DType.F32 },
            new AttentionLayerWeights
            {
                Query = _qW, QueryBias = _qB, Key = _kW, KeyBias = _kB, Value = _vW, ValueBias = _vB, Output = _oW, OutputBias = _oB,
                QueryNorm = _nq, KeyNorm = _nk, QueryNormEps = _eps, KeyNormEps = _eps,
            });
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _nq, _nk }) if (t is not null) yield return t;
    }

    /// <summary>Returns the adapter output <c>[S, dim]</c> to be added to the latent stream. <paramref name="hidden"/>
    /// is <c>[S, dim]</c>; <paramref name="motion"/> is <c>[T, N, dim]</c>; <c>T</c> must divide <c>S</c>. Every step runs on
    /// the backend with one batch item a frame, so frame group <c>t</c> (S/T latent tokens) attends frame <c>t</c>'s N motion
    /// tokens only and nothing is read back to the host.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor motion)
    {
        AttentionLayer layer = _layer ?? throw new InvalidOperationException("Call LoadWeights before Forward.");
        int s = (int)hidden.Shape[0];
        int t = (int)motion.Shape[0], n = (int)motion.Shape[1];
        if (s % t != 0)
            throw new ArgumentException($"face adapter needs T|S; T={t} does not divide S={s}.");

        // pre_norm_feat and pre_norm_motion: no weights. The query rows are declared as T groups of S/T.
        using Tensor qn = new(new TensorShape(t, s / t, _dim), DType.F32);
        backend.LayerNormNoAffine(qn, hidden, _eps);
        using Tensor kn = new(new TensorShape(t, n, _dim), DType.F32);
        backend.LayerNormNoAffine(kn, motion, _eps);

        using Tensor adapted = layer.Forward(backend, qn, kn, default);
        // The layer returns the query's [T, S/T, dim]; the caller adds rows of [S, dim].
        Tensor o = new(new TensorShape(s, _dim), DType.F32);
        backend.CopyTo(o, adapted);
        return o;
    }

    /// <summary>Splits a fused <c>[2·rows, cols]</c> weight into two F32 <c>[rows, cols]</c> halves (row-major).</summary>
    private static (Tensor Top, Tensor Bottom) SplitRows(Tensor fused, int rows)
    {
        Tensor src = fused.DType == DType.F32 ? fused : fused.CastTo(DType.F32);
        int cols = (int)src.Shape[src.Shape.Rank - 1];
        Tensor top = new Tensor(new TensorShape(rows, cols), DType.F32);
        Tensor bottom = new Tensor(new TensorShape(rows, cols), DType.F32);
        long half = (long)rows * cols * 4;
        Buffer.MemoryCopy((float*)src.DataPointer, (float*)top.DataPointer, half, half);
        Buffer.MemoryCopy((float*)src.DataPointer + (long)rows * cols, (float*)bottom.DataPointer, half, half);
        if (!ReferenceEquals(src, fused)) src.Dispose();
        return (top, bottom);
    }

    /// <summary>Splits a fused <c>[2·n]</c> bias into two F32 <c>[n]</c> halves.</summary>
    private static (Tensor Top, Tensor Bottom) SplitRows1d(Tensor fused, int n)
    {
        Tensor src = fused.DType == DType.F32 ? fused : fused.CastTo(DType.F32);
        Tensor top = new Tensor(new TensorShape(n), DType.F32);
        Tensor bottom = new Tensor(new TensorShape(n), DType.F32);
        Buffer.MemoryCopy((float*)src.DataPointer, (float*)top.DataPointer, (long)n * 4, (long)n * 4);
        Buffer.MemoryCopy((float*)src.DataPointer + n, (float*)bottom.DataPointer, (long)n * 4, (long)n * 4);
        if (!ReferenceEquals(src, fused)) src.Dispose();
        return (top, bottom);
    }
}
