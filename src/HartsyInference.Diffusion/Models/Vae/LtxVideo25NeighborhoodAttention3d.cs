using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>3D neighborhood attention with per-axis absolute RoPE, the attention primitive of both the deterministic
/// and the diffusion blocks of the LTX-2.5 video decoder. Fused <c>qkv</c> → per-head RMSNorm on q/k → RoPE on q/k →
/// <see cref="IBackend.Na3d"/> → output projection, over a channels-last <c>[T·H·W, dim]</c> token grid.</summary>
internal sealed class LtxVideo25NeighborhoodAttention3d
{
    /// <summary>Set by the ComfyUI layer-diff harness to tap this module's internals; null in production.</summary>
    internal static Action<string, Tensor>? Tap;

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

        // Normalize q/k on the tensors themselves, NOT through a [tokens·heads, head_dim] Reshape view: RmsNorm
        // already rows by the last dim, and an in-place op on a view leaves its result in the view's own device
        // cache, which Dispose frees without writing back — the ApplyKeyframesAbsPos defect, which passes on
        // CpuBackend and silently feeds un-normalized q/k to attention on CUDA.
        Tap?.Invoke("attn_qkv", qkv);
        backend.RmsNorm(q, q, _qNormWeight!, _eps);
        backend.RmsNorm(k, k, _kNormWeight!, _eps);
        Tap?.Invoke("attn_qnorm", q);
        using (Tensor cosT = BuildTable(t, _invFreqT, sine: false))
        using (Tensor sinT = BuildTable(t, _invFreqT, sine: true))
        using (Tensor cosH = BuildTable(h, _invFreqH, sine: false))
        using (Tensor sinH = BuildTable(h, _invFreqH, sine: true))
        using (Tensor cosW = BuildTable(w, _invFreqW, sine: false))
        using (Tensor sinW = BuildTable(w, _invFreqW, sine: true))
        {
            backend.Ltx25NaRope3d(q, cosT, sinT, cosH, sinH, cosW, sinW, _ropeSplit.T, _ropeSplit.H);
            backend.Ltx25NaRope3d(k, cosT, sinT, cosH, sinH, cosW, sinW, _ropeSplit.T, _ropeSplit.H);
        }

        Tap?.Invoke("attn_qrope", q);
        Tap?.Invoke("attn_krope", k);
        Tap?.Invoke("attn_v", v);
        using Tensor attended = new Tensor(headShape, DType.F32);
        backend.Na3d(attended, q, k, v, _kernel.T, _kernel.H, _kernel.W, scale: 1f);

        Tap?.Invoke("attn_na3d", attended);
        Tensor result = new Tensor(new TensorShape(tokens, _dim), DType.F32);
        using Tensor attendedRows = attended.Reshape(new TensorShape(tokens, _dim));
        backend.Linear(result, attendedRows, _projWeight!, _projBias);
        Tap?.Invoke("attn_out", result);
        return result;
    }

    /// <summary>One <c>[length, pairs]</c> rope table. Angles are formed and rounded in F32 exactly like the
    /// reference's <c>pos[:, None] * inv[None, :]</c>, and on the host in both backends, so the device path cannot
    /// drift from the reference through a different <c>cos</c>/<c>sin</c> implementation.</summary>
    private static unsafe Tensor BuildTable(int length, float[] inverseFrequencies, bool sine)
    {
        int pairs = inverseFrequencies.Length;
        Tensor table = new Tensor(new TensorShape(length, pairs), DType.F32);
        float* p = (float*)table.DataPointer;
        for (int position = 0; position < length; position++)
        {
            for (int i = 0; i < pairs; i++)
            {
                float angle = position * inverseFrequencies[i];
                p[position * pairs + i] = sine ? MathF.Sin(angle) : MathF.Cos(angle);
            }
        }
        return table;
    }
}
