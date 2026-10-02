using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Auk;

/// <summary>One stream's attention projections: fused biased qkv, per-head RMSNorm on q and k (before RoPE) and the output projection.</summary>
public sealed class AukAttnProj
{
    private Tensor? _qkvW, _qkvB, _qNorm, _kNorm, _outW, _outB;
    private int _dim;
    private int _heads;
    private int _headDim;

    public void Load(IReadOnlyDictionary<string, Tensor> weights, string prefix, string qkvName, string qNormName, string kNormName,
        string outName, int dim, int heads)
    {
        int headDim = dim / heads;
        _qkvW = AukOps.Take(weights, $"{prefix}.{qkvName}.weight", 3 * dim, dim);
        _qkvB = AukOps.Take(weights, $"{prefix}.{qkvName}.bias", 3 * dim);
        _qNorm = AukOps.Take(weights, $"{prefix}.{qNormName}.weight", headDim);
        _kNorm = AukOps.Take(weights, $"{prefix}.{kNormName}.weight", headDim);
        _outW = AukOps.Take(weights, $"{prefix}.{outName}.weight", dim, dim);
        _outB = AukOps.Take(weights, $"{prefix}.{outName}.bias", dim);
        _dim = dim;
        _heads = heads;
        _headDim = headDim;
    }

    /// <summary>Returns q, k, v as <c>[1, n, heads, headDim]</c> with q/k RMS-normed and rotated at positions 0..n-1 of the tables; the caller owns all three.</summary>
    public (Tensor Q, Tensor K, Tensor V) Project(IBackend backend, Tensor normed, int n, Tensor cos, Tensor sin)
    {
        Tensor qkv = WhisperOps.ProjectLinear(backend, normed, _qkvW!, _qkvB, 1, n, _dim, 3 * _dim);
        TensorShape packed = new(1, n, _heads, _headDim);
        Tensor q = Normed(backend, qkv, 0, packed, _qNorm!);
        Tensor k = Normed(backend, qkv, _dim, packed, _kNorm!);
        Tensor v = new(packed, DType.F32);
        backend.SliceLastDim(v, qkv, 2 * _dim);
        qkv.Dispose();
        backend.WanRopeInterleaved(q, cos, sin, n, _heads, _headDim);
        backend.WanRopeInterleaved(k, cos, sin, n, _heads, _headDim);
        return (q, k, v);
    }

    /// <summary>Output projection of merged heads <c>[1, n, dim]</c>.</summary>
    public Tensor Output(IBackend backend, Tensor merged, int n) =>
        WhisperOps.ProjectLinear(backend, merged, _outW!, _outB, 1, n, _dim, _dim);

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_qkvW, _qkvB, _qNorm, _kNorm, _outW, _outB];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    private static Tensor Normed(IBackend backend, Tensor qkv, int offset, TensorShape packed, Tensor norm)
    {
        Tensor raw = new(packed, DType.F32);
        backend.SliceLastDim(raw, qkv, offset);
        Tensor normed = new(packed, DType.F32);
        backend.RmsNorm(normed, raw, norm, AukOps.RmsEps);
        raw.Dispose();
        return normed;
    }
}
