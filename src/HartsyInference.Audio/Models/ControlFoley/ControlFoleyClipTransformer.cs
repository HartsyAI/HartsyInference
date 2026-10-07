using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>open_clip <c>Transformer</c>: pre-norm residual blocks with packed-QKV multi-head attention and a QuickGELU MLP.</summary>
internal sealed unsafe class ControlFoleyClipTransformer
{
    private readonly int _width;
    private readonly int _heads;
    private readonly int _hidden;
    private readonly float _eps;
    private readonly Block[] _blocks;

    private sealed record Block(Tensor Ln1W, Tensor Ln1B, Tensor InW, Tensor InB, Tensor OutW, Tensor OutB,
        Tensor Ln2W, Tensor Ln2B, Tensor FcW, Tensor FcB, Tensor ProjW, Tensor ProjB);

    public ControlFoleyClipTransformer(IReadOnlyDictionary<string, Tensor> weights, string prefix, int layers, int width,
        int heads, int mlpRatio, float eps)
    {
        if (width % heads != 0)
        {
            throw new ArgumentException($"Width {width} is not divisible by {heads} heads.");
        }

        _width = width;
        _heads = heads;
        _hidden = width * mlpRatio;
        _eps = eps;
        _blocks = new Block[layers];
        for (int i = 0; i < layers; i++)
        {
            string p = $"{prefix}.resblocks.{i}";
            _blocks[i] = new Block(
                Load(weights, $"{p}.ln_1.weight", width), Load(weights, $"{p}.ln_1.bias", width),
                Load(weights, $"{p}.attn.in_proj_weight", 3 * width, width), Load(weights, $"{p}.attn.in_proj_bias", 3 * width),
                Load(weights, $"{p}.attn.out_proj.weight", width, width), Load(weights, $"{p}.attn.out_proj.bias", width),
                Load(weights, $"{p}.ln_2.weight", width), Load(weights, $"{p}.ln_2.bias", width),
                Load(weights, $"{p}.mlp.c_fc.weight", _hidden, width), Load(weights, $"{p}.mlp.c_fc.bias", _hidden),
                Load(weights, $"{p}.mlp.c_proj.weight", width, _hidden), Load(weights, $"{p}.mlp.c_proj.bias", width));
        }
    }

    /// <summary>Fetches <paramref name="key"/> as F32 after checking its shape.</summary>
    internal static Tensor Load(IReadOnlyDictionary<string, Tensor> weights, string key, params long[] shape)
    {
        if (!weights.TryGetValue(key, out Tensor? tensor))
        {
            throw new KeyNotFoundException($"ControlFoley CLIP checkpoint is missing '{key}'.");
        }

        if (tensor.Shape.Rank != shape.Length)
        {
            throw new InvalidDataException($"'{key}' has rank {tensor.Shape.Rank}, expected {shape.Length}.");
        }

        for (int d = 0; d < shape.Length; d++)
        {
            if (tensor.Shape[d] != shape[d])
            {
                throw new InvalidDataException($"'{key}' dimension {d} is {tensor.Shape[d]}, expected {shape[d]}.");
            }
        }

        return WhisperOps.EnsureF32(tensor);
    }

    /// <summary>Runs all blocks over <paramref name="x"/> (<c>[n, width]</c> row-major) and returns the new activations.</summary>
    public float[] Forward(IBackend backend, float[] x, int n, bool causal)
    {
        using Tensor? mask = causal ? WhisperOps.BuildCausalMask(n) : null;
        Tensor? mask4 = null;
        if (mask is not null)
        {
            mask4 = new Tensor(new TensorShape(1, 1, n, n), DType.F32);
            new ReadOnlySpan<float>((void*)mask.DataPointer, n * n).CopyTo(new Span<float>((void*)mask4.DataPointer, n * n));
        }

        try
        {
            foreach (Block b in _blocks)
            {
                float[] attn = Attention(backend, LayerNorm(backend, x, b.Ln1W, b.Ln1B, n), b, n, mask4);
                for (int i = 0; i < x.Length; i++)
                {
                    x[i] += attn[i];
                }

                float[] h = DacOps.Linear(backend, LayerNorm(backend, x, b.Ln2W, b.Ln2B, n), b.FcW, n, _width, _hidden, b.FcB);
                for (int i = 0; i < h.Length; i++)
                {
                    h[i] *= 1f / (1f + MathF.Exp(-1.702f * h[i]));
                }

                float[] ff = DacOps.Linear(backend, h, b.ProjW, n, _hidden, _width, b.ProjB);
                for (int i = 0; i < x.Length; i++)
                {
                    x[i] += ff[i];
                }
            }
        }
        finally
        {
            mask4?.Dispose();
        }

        return x;
    }

    /// <summary>LayerNorm over the last dimension of <c>[rows, width]</c> data.</summary>
    internal float[] LayerNorm(IBackend backend, float[] x, Tensor weight, Tensor bias, int rows) =>
        LayerNorm(backend, x, weight, bias, rows, _width, _eps);

    internal static float[] LayerNorm(IBackend backend, float[] x, Tensor weight, Tensor bias, int rows, int width, float eps)
    {
        using Tensor input = DacOps.FromHost(x, 1, rows, width);
        using Tensor output = new(new TensorShape(1, rows, width), DType.F32);
        backend.LayerNorm(output, input, weight, bias, eps);
        return DacOps.ToHost(output);
    }

    private float[] Attention(IBackend backend, float[] normed, Block b, int n, Tensor? mask)
    {
        int headDim = _width / _heads;
        float[] qkv = DacOps.Linear(backend, normed, b.InW, n, _width, 3 * _width, b.InB);
        TensorShape headShape = new(1, _heads, n, headDim);
        using Tensor q = new(headShape, DType.F32);
        using Tensor k = new(headShape, DType.F32);
        using Tensor v = new(headShape, DType.F32);
        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer;
        for (int t = 0; t < n; t++)
        {
            for (int h = 0; h < _heads; h++)
            {
                int src = t * 3 * _width + h * headDim;
                int dst = (h * n + t) * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    qp[dst + d] = qkv[src + d];
                    kp[dst + d] = qkv[src + _width + d];
                    vp[dst + d] = qkv[src + 2 * _width + d];
                }
            }
        }

        using Tensor ctx = new(headShape, DType.F32);
        backend.ScaledDotProductAttention(ctx, q, k, v, mask, 1f / MathF.Sqrt(headDim));
        float[] merged = new float[n * _width];
        float* cp = (float*)ctx.DataPointer;
        for (int t = 0; t < n; t++)
        {
            for (int h = 0; h < _heads; h++)
            {
                for (int d = 0; d < headDim; d++)
                {
                    merged[t * _width + h * headDim + d] = cp[(h * n + t) * headDim + d];
                }
            }
        }

        return DacOps.Linear(backend, merged, b.OutW, n, _width, _width, b.OutB);
    }
}
