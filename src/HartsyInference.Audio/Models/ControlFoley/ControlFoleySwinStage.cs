using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>One HTS-AT Swin stage (<c>BasicLayer</c>): alternating plain and shifted window-attention blocks with a relative
/// position bias, then an optional patch-merging downsample. Activations are host <c>[H * W, C]</c> row-major arrays; the
/// matrix products run on the backend.</summary>
internal sealed class ControlFoleySwinStage
{
    private const float LayerNormEps = 1e-5f;
    private const float ShiftMaskValue = -100f;

    private readonly int _dim;
    private readonly int _heads;
    private readonly int _resolution;
    private readonly int _window;
    private readonly Block[] _blocks;
    private readonly Tensor? _mergeNormW;
    private readonly Tensor? _mergeNormB;
    private readonly Tensor? _mergeW;
    private readonly int[] _relativeIndex;

    private sealed record Block(int Shift, float[]? Mask, Tensor Norm1W, Tensor Norm1B, Tensor Bias, Tensor QkvW, Tensor QkvB,
        Tensor ProjW, Tensor ProjB, Tensor Norm2W, Tensor Norm2B, Tensor Fc1W, Tensor Fc1B, Tensor Fc2W, Tensor Fc2B);

    internal ControlFoleySwinStage(IReadOnlyDictionary<string, Tensor> weights, string prefix, int dim, int heads, int depth,
        int resolution, int window, bool downsample)
    {
        if (dim % heads != 0)
        {
            throw new ArgumentException($"Stage width {dim} is not divisible by {heads} heads.");
        }

        _dim = dim;
        _heads = heads;
        _resolution = resolution;
        _window = resolution <= window ? resolution : window;
        _relativeIndex = BuildRelativeIndex(_window);
        int tableRows = (2 * _window - 1) * (2 * _window - 1);
        _blocks = new Block[depth];
        for (int i = 0; i < depth; i++)
        {
            string p = $"{prefix}.blocks.{i}";
            int shift = resolution <= window || i % 2 == 0 ? 0 : window / 2;
            _blocks[i] = new Block(shift, shift > 0 ? BuildShiftMask(resolution, _window, shift) : null,
                ControlFoleyClap.Require(weights, $"{p}.norm1.weight", dim), ControlFoleyClap.Require(weights, $"{p}.norm1.bias", dim),
                ControlFoleyClap.Require(weights, $"{p}.attn.relative_position_bias_table", tableRows, heads),
                ControlFoleyClap.Require(weights, $"{p}.attn.qkv.weight", 3 * dim, dim),
                ControlFoleyClap.Require(weights, $"{p}.attn.qkv.bias", 3 * dim),
                ControlFoleyClap.Require(weights, $"{p}.attn.proj.weight", dim, dim),
                ControlFoleyClap.Require(weights, $"{p}.attn.proj.bias", dim),
                ControlFoleyClap.Require(weights, $"{p}.norm2.weight", dim), ControlFoleyClap.Require(weights, $"{p}.norm2.bias", dim),
                ControlFoleyClap.Require(weights, $"{p}.mlp.fc1.weight", 4 * dim, dim),
                ControlFoleyClap.Require(weights, $"{p}.mlp.fc1.bias", 4 * dim),
                ControlFoleyClap.Require(weights, $"{p}.mlp.fc2.weight", dim, 4 * dim),
                ControlFoleyClap.Require(weights, $"{p}.mlp.fc2.bias", dim));
        }

        if (downsample)
        {
            _mergeNormW = ControlFoleyClap.Require(weights, $"{prefix}.downsample.norm.weight", 4 * dim);
            _mergeNormB = ControlFoleyClap.Require(weights, $"{prefix}.downsample.norm.bias", 4 * dim);
            _mergeW = ControlFoleyClap.Require(weights, $"{prefix}.downsample.reduction.weight", 2 * dim, 4 * dim);
        }
    }

    /// <summary>Width of the activations this stage returns.</summary>
    internal int OutputDim => _mergeW is null ? _dim : 2 * _dim;

    /// <summary>Runs the blocks and the downsample over <c>[resolution^2, dim]</c> activations.</summary>
    internal float[] Forward(IBackend backend, float[] x)
    {
        int n = _resolution * _resolution;
        if (x.Length != n * _dim)
        {
            throw new ArgumentException($"Stage expects {n * _dim} activations, got {x.Length}.", nameof(x));
        }

        foreach (Block b in _blocks)
        {
            float[] normed = ControlFoleyClipTransformer.LayerNorm(backend, x, b.Norm1W, b.Norm1B, n, _dim, LayerNormEps);
            float[] attn = WindowAttention(backend, normed, b);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] += attn[i];
            }

            float[] hidden = DacOps.Linear(backend, ControlFoleyClipTransformer.LayerNorm(backend, x, b.Norm2W, b.Norm2B, n, _dim, LayerNormEps),
                b.Fc1W, n, _dim, 4 * _dim, b.Fc1B);
            for (int i = 0; i < hidden.Length; i++)
            {
                hidden[i] = Activations.ErfGelu(hidden[i]);
            }

            float[] mlp = DacOps.Linear(backend, hidden, b.Fc2W, n, 4 * _dim, _dim, b.Fc2B);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] += mlp[i];
            }
        }

        return _mergeW is null ? x : Merge(backend, x);
    }

    private float[] WindowAttention(IBackend backend, float[] normed, Block b)
    {
        int res = _resolution, ws = _window, c = _dim, tokensPerWindow = ws * ws, windowsPerSide = res / ws;
        int n = res * res;
        int[] order = new int[n];
        int at = 0;
        for (int wy = 0; wy < windowsPerSide; wy++)
        {
            for (int wx = 0; wx < windowsPerSide; wx++)
            {
                for (int iy = 0; iy < ws; iy++)
                {
                    for (int ix = 0; ix < ws; ix++)
                    {
                        int y = (wy * ws + iy + b.Shift) % res, x = (wx * ws + ix + b.Shift) % res;
                        order[at++] = y * res + x;
                    }
                }
            }
        }

        float[] gathered = new float[n * c];
        for (int i = 0; i < n; i++)
        {
            Array.Copy(normed, order[i] * c, gathered, i * c, c);
        }

        float[] qkv = DacOps.Linear(backend, gathered, b.QkvW, n, c, 3 * c, b.QkvB);
        float[] merged = new float[n * c];
        int headDim = c / _heads;
        float scale = MathF.Pow(headDim, -0.5f);
        float[] scores = new float[tokensPerWindow];
        float[] bias = new float[_heads * tokensPerWindow * tokensPerWindow];
        unsafe
        {
            float* table = (float*)b.Bias.DataPointer;
            for (int h = 0; h < _heads; h++)
            {
                for (int i = 0; i < tokensPerWindow * tokensPerWindow; i++)
                {
                    bias[h * tokensPerWindow * tokensPerWindow + i] = table[_relativeIndex[i] * _heads + h];
                }
            }
        }

        for (int w = 0; w < windowsPerSide * windowsPerSide; w++)
        {
            int baseRow = w * tokensPerWindow;
            for (int h = 0; h < _heads; h++)
            {
                for (int i = 0; i < tokensPerWindow; i++)
                {
                    int qOff = (baseRow + i) * 3 * c + h * headDim;
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < tokensPerWindow; j++)
                    {
                        int kOff = (baseRow + j) * 3 * c + c + h * headDim;
                        float dot = 0f;
                        for (int d = 0; d < headDim; d++)
                        {
                            dot += qkv[qOff + d] * scale * qkv[kOff + d];
                        }

                        dot += bias[(h * tokensPerWindow + i) * tokensPerWindow + j];
                        if (b.Mask is not null)
                        {
                            dot += b.Mask[(w * tokensPerWindow + i) * tokensPerWindow + j];
                        }

                        scores[j] = dot;
                        max = MathF.Max(max, dot);
                    }

                    float sum = 0f;
                    for (int j = 0; j < tokensPerWindow; j++)
                    {
                        scores[j] = MathF.Exp(scores[j] - max);
                        sum += scores[j];
                    }

                    int outOff = (baseRow + i) * c + h * headDim;
                    for (int d = 0; d < headDim; d++)
                    {
                        float acc = 0f;
                        for (int j = 0; j < tokensPerWindow; j++)
                        {
                            acc += scores[j] / sum * qkv[(baseRow + j) * 3 * c + 2 * c + h * headDim + d];
                        }

                        merged[outOff + d] = acc;
                    }
                }
            }
        }

        float[] projected = DacOps.Linear(backend, merged, b.ProjW, n, c, c, b.ProjB);
        float[] result = new float[n * c];
        for (int i = 0; i < n; i++)
        {
            Array.Copy(projected, i * c, result, order[i] * c, c);
        }

        return result;
    }

    private float[] Merge(IBackend backend, float[] x)
    {
        int res = _resolution, half = res / 2, c = _dim;
        float[] merged = new float[half * half * 4 * c];
        for (int y = 0; y < half; y++)
        {
            for (int xx = 0; xx < half; xx++)
            {
                int dst = (y * half + xx) * 4 * c;
                Array.Copy(x, ((2 * y) * res + 2 * xx) * c, merged, dst, c);
                Array.Copy(x, ((2 * y + 1) * res + 2 * xx) * c, merged, dst + c, c);
                Array.Copy(x, ((2 * y) * res + 2 * xx + 1) * c, merged, dst + 2 * c, c);
                Array.Copy(x, ((2 * y + 1) * res + 2 * xx + 1) * c, merged, dst + 3 * c, c);
            }
        }

        float[] normed = ControlFoleyClipTransformer.LayerNorm(backend, merged, _mergeNormW!, _mergeNormB!, half * half, 4 * c, LayerNormEps);
        return DacOps.Linear(backend, normed, _mergeW!, half * half, 4 * c, 2 * c);
    }

    private static int[] BuildRelativeIndex(int window)
    {
        int n = window * window;
        int[] index = new int[n * n];
        for (int a = 0; a < n; a++)
        {
            for (int b = 0; b < n; b++)
            {
                int dy = a / window - b / window + window - 1, dx = a % window - b % window + window - 1;
                index[a * n + b] = dy * (2 * window - 1) + dx;
            }
        }

        return index;
    }

    /// <summary>The 0 / -100 additive mask of shifted-window attention, <c>[windows, N, N]</c>.</summary>
    private static float[] BuildShiftMask(int resolution, int window, int shift)
    {
        int[] region = new int[resolution * resolution];
        int count = 0;
        int[] edges = [0, resolution - window, resolution - shift, resolution];
        for (int hy = 0; hy < 3; hy++)
        {
            for (int hx = 0; hx < 3; hx++)
            {
                for (int y = edges[hy]; y < edges[hy + 1]; y++)
                {
                    for (int x = edges[hx]; x < edges[hx + 1]; x++)
                    {
                        region[y * resolution + x] = count;
                    }
                }

                count++;
            }
        }

        int perSide = resolution / window, n = window * window;
        float[] mask = new float[perSide * perSide * n * n];
        for (int wy = 0; wy < perSide; wy++)
        {
            for (int wx = 0; wx < perSide; wx++)
            {
                int w = wy * perSide + wx;
                for (int i = 0; i < n; i++)
                {
                    int ri = region[(wy * window + i / window) * resolution + wx * window + i % window];
                    for (int j = 0; j < n; j++)
                    {
                        int rj = region[(wy * window + j / window) * resolution + wx * window + j % window];
                        mask[(w * n + i) * n + j] = ri == rj ? 0f : ShiftMaskValue;
                    }
                }
            }
        }

        return mask;
    }
}
