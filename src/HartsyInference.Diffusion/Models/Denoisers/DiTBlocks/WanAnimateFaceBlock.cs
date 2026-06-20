using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Wan-Animate face-adapter cross-attention (<c>WanAnimateFaceBlockCrossAttention</c>): the latent stream
/// <c>[S, dim]</c> cross-attends to the per-frame face features <c>[T, N+1, dim]</c>. It is <b>temporally aligned</b> —
/// the S tokens are split into T contiguous groups (so T must divide S) and group <c>t</c> attends only to face frame
/// <c>t</c>'s <c>N+1</c> tokens. No-affine LayerNorm pre-norms on both sides; per-head-dim RMSNorm QK-norm. B=1.</summary>
public sealed unsafe class WanAnimateFaceBlock
{
    private readonly int _dim, _heads, _headDim, _inner;
    private readonly float _eps;
    private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _nq, _nk;

    public WanAnimateFaceBlock(int dim, int heads, int headDim, float eps = 1e-6f)
    {
        _dim = dim;
        _heads = heads;
        _headDim = headDim;
        _inner = heads * headDim;
        _eps = eps;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _qW = w[$"{p}.to_q.weight"]; w.TryGetValue($"{p}.to_q.bias", out _qB);
        _kW = w[$"{p}.to_k.weight"]; w.TryGetValue($"{p}.to_k.bias", out _kB);
        _vW = w[$"{p}.to_v.weight"]; w.TryGetValue($"{p}.to_v.bias", out _vB);
        _oW = w[$"{p}.to_out.weight"]; w.TryGetValue($"{p}.to_out.bias", out _oB);
        _nq = LoadF32(w, $"{p}.norm_q.weight");
        _nk = LoadF32(w, $"{p}.norm_k.weight");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _nq, _nk }) if (t is not null) yield return t;
    }

    /// <summary>Returns the adapter output <c>[S, dim]</c> to be added to the latent stream. <paramref name="hidden"/>
    /// is <c>[S, dim]</c>; <paramref name="motion"/> is <c>[T, N+1, dim]</c>; <c>T</c> must divide <c>S</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor motion)
    {
        int s = (int)hidden.Shape[0];
        int t = (int)motion.Shape[0], n = (int)motion.Shape[1];
        if (s % t != 0)
            throw new ArgumentException($"face adapter needs T|S; T={t} does not divide S={s}.");
        int groupLen = s / t;
        int kvRows = t * n;

        Tensor qn = LayerNormNoAffine(hidden, s, _dim);
        Tensor motionFlat = Reshape(motion, kvRows, _dim);
        Tensor kn = LayerNormNoAffine(motionFlat, kvRows, _dim);
        motionFlat.Dispose();

        Tensor q = new Tensor(new TensorShape(s, _inner), DType.F32); backend.Linear(q, qn, _qW!, _qB); qn.Dispose();
        Tensor k = new Tensor(new TensorShape(kvRows, _inner), DType.F32); backend.Linear(k, kn, _kW!, _kB);
        Tensor v = new Tensor(new TensorShape(kvRows, _inner), DType.F32); backend.Linear(v, kn, _vW!, _vB); kn.Dispose();

        RmsNormHeads(q, s);   // per-head-dim RMSNorm (norm_q)
        RmsNormHeadsK(k, kvRows);

        Tensor outFlat = new Tensor(new TensorShape(s, _inner), DType.F32);
        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer, op = (float*)outFlat.DataPointer;
        float scale = 1f / MathF.Sqrt(_headDim);
        float[] scores = new float[n];
        // For each motion frame t, the group of S/T latent tokens attends to that frame's N tokens, per head.
        for (int ti = 0; ti < t; ti++)
        {
            for (int gi = 0; gi < groupLen; gi++)
            {
                int qRow = ti * groupLen + gi;
                for (int h = 0; h < _heads; h++)
                {
                    long qOff = (long)qRow * _inner + (long)h * _headDim;
                    float maxS = float.NegativeInfinity;
                    for (int j = 0; j < n; j++)
                    {
                        long kOff = ((long)ti * n + j) * _inner + (long)h * _headDim;
                        float dot = 0;
                        for (int d = 0; d < _headDim; d++) dot += qp[qOff + d] * kp[kOff + d];
                        scores[j] = dot * scale;
                        if (scores[j] > maxS) maxS = scores[j];
                    }
                    float sum = 0;
                    for (int j = 0; j < n; j++) { scores[j] = MathF.Exp(scores[j] - maxS); sum += scores[j]; }
                    float invSum = 1f / sum;
                    long oOff = (long)qRow * _inner + (long)h * _headDim;
                    for (int d = 0; d < _headDim; d++)
                    {
                        float acc = 0;
                        for (int j = 0; j < n; j++)
                        {
                            long vOff = ((long)ti * n + j) * _inner + (long)h * _headDim;
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

    private void RmsNormHeads(Tensor x, int rows) => RmsPerHead(x, rows, _nq!);
    private void RmsNormHeadsK(Tensor x, int rows) => RmsPerHead(x, rows, _nk!);

    /// <summary>Per-head-dim RMSNorm (over <c>headDim</c>) with affine weight, applied to <c>[rows, heads·headDim]</c>.</summary>
    private void RmsPerHead(Tensor x, int rows, Tensor weight)
    {
        float* xp = (float*)x.DataPointer, wp = (float*)weight.DataPointer;
        for (int i = 0; i < rows; i++)
            for (int h = 0; h < _heads; h++)
            {
                long off = (long)i * _inner + (long)h * _headDim;
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

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key) { Tensor t = w[key]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }
}
