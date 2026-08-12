using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>3D neighborhood attention with per-axis absolute RoPE, the attention primitive of both the deterministic
/// and the diffusion blocks of the LTX-2.5 video decoder. Fused <c>qkv</c> → per-head RMSNorm on q/k → RoPE on q/k →
/// <see cref="IBackend.Na3d"/> → output projection, over a channels-last <c>[T·H·W, dim]</c> token grid.</summary>
internal sealed class LtxVideo25NeighborhoodAttention3d
{
    private readonly int _dim;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly (int T, int H, int W) _kernel;
    private readonly (int T, int H, int W) _ropeSplit;
    private readonly float[] _invFreqT;
    private readonly float[] _invFreqH;
    private readonly float[] _invFreqW;
    private readonly float _eps;

    private Tensor? _qkvWeight, _qkvBias, _projWeight, _projBias, _qNormWeight, _kNormWeight;

    public LtxVideo25NeighborhoodAttention3d(int dim, (int T, int H, int W) kernel, LtxVideo25DiffusionDecoderConfig config)
    {
        if (dim % config.HeadDim != 0)
            throw new ArgumentException($"dim {dim} is not divisible by head_dim {config.HeadDim}.", nameof(dim));
        _dim = dim;
        _headDim = config.HeadDim;
        _heads = dim / config.HeadDim;
        _kernel = kernel;
        _eps = config.NormEps;
        _ropeSplit = LtxVideo25DiffusionDecoderConfig.RopeDimSplit(_headDim);
        _invFreqT = LtxVideo25DiffusionDecoderConfig.RopeInverseFrequencies(_ropeSplit.T, config.RopeBase);
        _invFreqH = LtxVideo25DiffusionDecoderConfig.RopeInverseFrequencies(_ropeSplit.H, config.RopeBase);
        _invFreqW = LtxVideo25DiffusionDecoderConfig.RopeInverseFrequencies(_ropeSplit.W, config.RopeBase);
    }

    /// <summary>Loads <c>{prefix}.qkv|proj|q_norm|k_norm</c>. The <c>1/√head_dim</c> attention scale is folded into the
    /// q-norm weight — it commutes with the RoPE rotation that follows, so <see cref="IBackend.Na3d"/> runs unscaled.</summary>
    public void LoadWeights(LtxVideo25WeightScope scope, string prefix)
    {
        _qkvWeight = scope.Raw($"{prefix}.qkv.weight");
        _qkvBias = scope.OptionalF32($"{prefix}.qkv.bias");
        _projWeight = scope.Raw($"{prefix}.proj.weight");
        _projBias = scope.OptionalF32($"{prefix}.proj.bias");
        _qNormWeight = scope.ScaledF32($"{prefix}.q_norm.weight", 1f / MathF.Sqrt(_headDim));
        _kNormWeight = scope.F32($"{prefix}.k_norm.weight");
        if (_qkvWeight.Shape[0] != 3L * _dim || _qkvWeight.Shape[1] != _dim)
            throw new InvalidOperationException($"'{prefix}.qkv.weight' is {_qkvWeight.Shape}, expected [{3 * _dim}, {_dim}].");
        if (_qNormWeight.ElementCount != _headDim)
            throw new InvalidOperationException($"'{prefix}.q_norm.weight' has {_qNormWeight.ElementCount} entries, expected {_headDim}.");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _qkvWeight, _qkvBias, _projWeight, _projBias, _qNormWeight, _kNormWeight })
        {
            if (t is not null) yield return t;
        }
    }

    /// <summary>Attends <paramref name="x"/> <c>[t·h·w, dim]</c> and returns a freshly allocated result of the same shape.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int t, int h, int w)
    {
        long tokens = (long)t * h * w;
        TensorShape headShape = new TensorShape([1, t, h, w, _heads, _headDim]);

        using Tensor qkv = new Tensor(new TensorShape(tokens, 3L * _dim), DType.F32);
        backend.Linear(qkv, x, _qkvWeight!, _qkvBias);

        using Tensor q = new Tensor(headShape, DType.F32);
        using Tensor k = new Tensor(headShape, DType.F32);
        using Tensor v = new Tensor(headShape, DType.F32);
        backend.SliceLastDim(q, qkv, 0);
        backend.SliceLastDim(k, qkv, _dim);
        backend.SliceLastDim(v, qkv, 2 * _dim);

        TensorShape perHead = new TensorShape(tokens * _heads, _headDim);
        using (Tensor qRows = q.Reshape(perHead))
        using (Tensor kRows = k.Reshape(perHead))
        {
            backend.RmsNorm(qRows, qRows, _qNormWeight!, _eps);
            backend.RmsNorm(kRows, kRows, _kNormWeight!, _eps);
        }
        ApplyRope(q, t, h, w);
        ApplyRope(k, t, h, w);

        using Tensor attended = new Tensor(headShape, DType.F32);
        backend.Na3d(attended, q, k, v, _kernel.T, _kernel.H, _kernel.W, scale: 1f);

        Tensor result = new Tensor(new TensorShape(tokens, _dim), DType.F32);
        using Tensor attendedRows = attended.Reshape(new TensorShape(tokens, _dim));
        backend.Linear(result, attendedRows, _projWeight!, _projBias);
        return result;
    }

    /// <summary>Rotates <c>[1, t, h, w, heads, head_dim]</c> in place: each head's channels split into three
    /// contiguous chunks rotated by the token's global t / h / w coordinate, interleaved (even, odd) pairs.</summary>
    private unsafe void ApplyRope(Tensor x, int t, int h, int w)
    {
        (float[] cosT, float[] sinT) = BuildTable(t, _invFreqT);
        (float[] cosH, float[] sinH) = BuildTable(h, _invFreqH);
        (float[] cosW, float[] sinW) = BuildTable(w, _invFreqW);
        int pairsT = _invFreqT.Length, pairsH = _invFreqH.Length, pairsW = _invFreqW.Length;
        int offsetH = _ropeSplit.T, offsetW = _ropeSplit.T + _ropeSplit.H;
        float* p = (float*)x.DataPointer;

        for (int ti = 0; ti < t; ti++)
        for (int hi = 0; hi < h; hi++)
        for (int wi = 0; wi < w; wi++)
        {
            long tokenBase = (((long)ti * h + hi) * w + wi) * _heads * _headDim;
            for (int head = 0; head < _heads; head++)
            {
                float* row = p + tokenBase + (long)head * _headDim;
                Rotate(row, pairsT, cosT, sinT, ti * pairsT);
                Rotate(row + offsetH, pairsH, cosH, sinH, hi * pairsH);
                Rotate(row + offsetW, pairsW, cosW, sinW, wi * pairsW);
            }
        }
    }

    private static unsafe void Rotate(float* row, int pairs, float[] cos, float[] sin, int tableOffset)
    {
        for (int i = 0; i < pairs; i++)
        {
            float even = row[2 * i], odd = row[2 * i + 1];
            float c = cos[tableOffset + i], s = sin[tableOffset + i];
            row[2 * i] = even * c - odd * s;
            row[2 * i + 1] = even * s + odd * c;
        }
    }

    /// <summary>Angles are formed and rounded in F32 exactly like the reference's <c>pos[:, None] * inv[None, :]</c>.</summary>
    private static (float[] Cos, float[] Sin) BuildTable(int length, float[] inverseFrequencies)
    {
        int pairs = inverseFrequencies.Length;
        float[] cos = new float[(long)length * pairs];
        float[] sin = new float[(long)length * pairs];
        for (int position = 0; position < length; position++)
        {
            for (int i = 0; i < pairs; i++)
            {
                float angle = position * inverseFrequencies[i];
                cos[position * pairs + i] = MathF.Cos(angle);
                sin[position * pairs + i] = MathF.Sin(angle);
            }
        }
        return (cos, sin);
    }
}
