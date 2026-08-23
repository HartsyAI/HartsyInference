using HartsyInference.Core.Backends;
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
public sealed unsafe class WanAnimateFaceBlock
{
    private readonly int _dim, _heads, _headDim;
    private readonly float _eps;
    private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _nq, _nk;

    public WanAnimateFaceBlock(int dim, int heads, int headDim, float eps = 1e-6f)
    {
        _dim = dim;
        _heads = heads;
        _headDim = headDim;
        _eps = eps;
    }

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
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _nq, _nk }) if (t is not null) yield return t;
    }

    /// <summary>Returns the adapter output <c>[S, dim]</c> to be added to the latent stream. <paramref name="hidden"/>
    /// is <c>[S, dim]</c>; <paramref name="motion"/> is <c>[T, N, dim]</c>; <c>T</c> must divide <c>S</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor motion)
    {
        int s = (int)hidden.Shape[0];
        int t = (int)motion.Shape[0], n = (int)motion.Shape[1];
        if (s % t != 0)
            throw new ArgumentException($"face adapter needs T|S; T={t} does not divide S={s}.");
        int groupLen = s / t;
        int kvRows = t * n;

        Tensor qn = LayerNormNoAffine(hidden, s, _dim);                 // pre_norm_feat
        Tensor motionFlat = Reshape(motion, kvRows, _dim);
        Tensor kn = LayerNormNoAffine(motionFlat, kvRows, _dim);        // pre_norm_motion
        motionFlat.Dispose();

        Tensor q = new Tensor(new TensorShape(s, _dim), DType.F32); backend.Linear(q, qn, _qW!, _qB); qn.Dispose();
        Tensor k = new Tensor(new TensorShape(kvRows, _dim), DType.F32); backend.Linear(k, kn, _kW!, _kB);
        Tensor v = new Tensor(new TensorShape(kvRows, _dim), DType.F32); backend.Linear(v, kn, _vW!, _vB); kn.Dispose();

        RmsPerHead(q, s, _nq!);
        RmsPerHead(k, kvRows, _nk!);

        Tensor outFlat = new Tensor(new TensorShape(s, _dim), DType.F32);
        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer, op = (float*)outFlat.DataPointer;
        float scale = 1f / MathF.Sqrt(_headDim);
        float[] scores = new float[n];
        // Frame group ti (S/T latent tokens) attends only to frame ti's N motion tokens, per head.
        for (int ti = 0; ti < t; ti++)
        {
            for (int gi = 0; gi < groupLen; gi++)
            {
                int qRow = ti * groupLen + gi;
                for (int h = 0; h < _heads; h++)
                {
                    long qOff = (long)qRow * _dim + (long)h * _headDim;
                    float maxS = float.NegativeInfinity;
                    for (int j = 0; j < n; j++)
                    {
                        long kOff = ((long)ti * n + j) * _dim + (long)h * _headDim;
                        float dot = 0;
                        for (int d = 0; d < _headDim; d++) dot += qp[qOff + d] * kp[kOff + d];
                        scores[j] = dot * scale;
                        if (scores[j] > maxS) maxS = scores[j];
                    }
                    float sum = 0;
                    for (int j = 0; j < n; j++) { scores[j] = MathF.Exp(scores[j] - maxS); sum += scores[j]; }
                    float invSum = 1f / sum;
                    long oOff = (long)qRow * _dim + (long)h * _headDim;
                    for (int d = 0; d < _headDim; d++)
                    {
                        float acc = 0;
                        for (int j = 0; j < n; j++)
                        {
                            long vOff = ((long)ti * n + j) * _dim + (long)h * _headDim;
                            acc += scores[j] * vp[vOff + d];
                        }
                        op[oOff + d] = acc * invSum;
                    }
                }
            }
        }
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor o = new Tensor(new TensorShape(s, _dim), DType.F32);
        backend.Linear(o, outFlat, _oW!, _oB);
        outFlat.Dispose();
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

    private Tensor LayerNormNoAffine(Tensor x, int rows, int dim)
    {
        Tensor o = new Tensor(new TensorShape(rows, dim), DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int i = 0; i < rows; i++)
        {
            long off = (long)i * dim;
            double mean = 0; for (int d = 0; d < dim; d++) mean += xp[off + d]; mean /= dim;
            double var = 0; for (int d = 0; d < dim; d++) { double dd = xp[off + d] - mean; var += dd * dd; }
            float inv = 1f / MathF.Sqrt((float)(var / dim) + _eps);
            for (int d = 0; d < dim; d++) op[off + d] = (float)((xp[off + d] - mean) * inv);
        }
        return o;
    }

    /// <summary>Per-head-dim RMSNorm (over <c>headDim</c>) with affine weight, in place on <c>[rows, heads·headDim]</c>.</summary>
    private void RmsPerHead(Tensor x, int rows, Tensor weight)
    {
        float* xp = (float*)x.DataPointer, wp = (float*)weight.DataPointer;
        for (int i = 0; i < rows; i++)
            for (int h = 0; h < _heads; h++)
            {
                long off = (long)i * _dim + (long)h * _headDim;
                double sumSq = 0; for (int d = 0; d < _headDim; d++) sumSq += (double)xp[off + d] * xp[off + d];
                float inv = 1f / MathF.Sqrt((float)(sumSq / _headDim) + _eps);
                for (int d = 0; d < _headDim; d++) xp[off + d] = xp[off + d] * inv * wp[d];
            }
    }

    private static Tensor Reshape(Tensor x, int rows, int dim)
    {
        Tensor o = new Tensor(new TensorShape(rows, dim), DType.F32);
        Buffer.MemoryCopy((float*)x.DataPointer, (float*)o.DataPointer, (long)rows * dim * 4, (long)rows * dim * 4);
        return o;
    }
}
