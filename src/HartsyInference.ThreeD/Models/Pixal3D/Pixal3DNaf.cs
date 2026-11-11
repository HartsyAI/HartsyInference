using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Models.Pixal3D;

/// <summary>
/// NAF (<c>valeoai/NAF</c>, dim 256, 4 attention heads, 4 RoPE heads, kernel 9), the feature upsampler Pixal3D's shape and texture stages use.
/// A conv encoder turns the RGB guide image into a 256-channel query map at the output size; the keys are that map averaged down to the
/// feature map's size, the values the feature map itself, and each output pixel attends to the <c>9 × 9</c> block of feature cells around its own
/// cell (a dilated neighbourhood attention with dilation = output / feature size). Because keys and values are constant inside each cell, the
/// dilated window over the upsampled maps is exactly that cell window, so no <c>natten</c> and no upsampled key or value map is needed, and
/// the output is evaluated only where Pixal3D's projection samples it. The checkpoint holds only the encoder: the attention has no weights.
/// </summary>
public sealed class Pixal3DNaf : IDisposable
{
    /// <summary>Channels of the query and key maps (<c>dim</c>).</summary>
    public const int Dim = 256;

    /// <summary>Attention and RoPE heads (<c>heads_attn</c>, <c>heads_rope</c>).</summary>
    public const int Heads = 4;

    /// <summary>Feature cells per window edge (<c>kernel_size</c>).</summary>
    public const int Kernel = 9;

    private const int HeadDim = Dim / Heads;
    private const int BranchChannels = Dim / 2;
    private const int Blocks = 2;
    private const int GroupCount = 8;
    private const int PeriodCount = HeadDim / 4;
    private const float GroupNormEps = 1e-5f;
    private const string Prefix = "image_encoder.";

    private readonly Dictionary<string, Tensor> _weights;
    private readonly IDisposable? _owner;
    private int _disposed;

    private Pixal3DNaf(Dictionary<string, Tensor> weights, IDisposable? owner)
    {
        _weights = weights;
        _owner = owner;
    }

    /// <summary>Reads <c>naf_release.pth</c> with the safe-subset pickle reader (nothing in the file is executed) and checks every tensor.</summary>
    public static Pixal3DNaf Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"NAF checkpoint not found: '{path}'.", path);
        PytorchPickleLoader loader = new();
        try
        {
            loader.Load(path);
            return FromTensors(loader.GetAllTensors(), loader);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    /// <summary>Checks <paramref name="tensors"/> (F32, the checkpoint's names) against the architecture. <paramref name="owner"/> holds their memory
    /// and is disposed with the result; with null the caller keeps the tensors alive.</summary>
    public static Pixal3DNaf FromTensors(IReadOnlyDictionary<string, Tensor> tensors, IDisposable? owner = null)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        Dictionary<string, long[]> expected = Spec();
        foreach (string key in tensors.Keys)
        {
            if (!expected.ContainsKey(key))
                throw new HartsyInferenceException($"NAF checkpoint holds '{key}', which the architecture does not use.");
        }

        foreach ((string key, long[] dims) in expected)
        {
            if (!tensors.TryGetValue(key, out Tensor? t))
                throw new HartsyInferenceException($"NAF checkpoint is missing '{key}'.");
            if (t.DType != DType.F32 || !t.Shape.Equals(new TensorShape(dims)))
                throw new HartsyInferenceException($"NAF tensor '{key}' is {t.DType} {t.Shape}; expected F32 [{string.Join(", ", dims)}].");
        }

        return new Pixal3DNaf(new Dictionary<string, Tensor>(tensors, StringComparer.Ordinal), owner);
    }

    /// <summary>The tensors the encoder reads, for <c>IBackend.PreloadWeights</c> and <c>FreeWeights</c>.</summary>
    public IEnumerable<Tensor> EnumerateWeights() => _weights.Values;

    /// <summary>The query map <c>[1, 256, outputSize, outputSize]</c> for the guide <paramref name="image"/> <c>[1, 3, S, S]</c> in <c>[0, 1]</c>:
    /// both conv branches at the image's size, averaged down to <paramref name="outputSize"/>, then the axial RoPE. The caller owns the result.</summary>
    public Tensor QueryMap(IBackend backend, Tensor image, int outputSize)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        if (image.DType != DType.F32 || image.Shape.Rank != 4 || image.Shape[0] != 1 || image.Shape[1] != 3 || image.Shape[2] != image.Shape[3])
            throw new ArgumentException($"The guide image must be F32 [1, 3, S, S]; got {image.DType} {image.Shape}.", nameof(image));
        int size = (int)image.Shape[2];
        if (outputSize <= 0 || size % outputSize != 0 || size > 4 * outputSize)
            throw new ArgumentOutOfRangeException(nameof(outputSize), outputSize, $"The output size must divide the {size} px image and be at least a quarter of it.");
        using Tensor pointwise = Branch(backend, image, "encoder", 1);
        using Tensor spatial = Branch(backend, image, "sem_encoder", 3);
        Tensor joined = new(new TensorShape(1, Dim, size, size), DType.F32);
        try
        {
            backend.Concat(joined, [pointwise, spatial], 1);
            backend.Sync();
            Tensor map = size == outputSize ? joined : AveragePool(joined, size / outputSize);
            if (!ReferenceEquals(map, joined))
                joined.Dispose();
            ApplyRope(map, outputSize, outputSize);
            return map;
        }
        catch
        {
            joined.Dispose();
            throw;
        }
    }

    /// <summary>
    /// The NAF output at the sample positions of a bilinear, border-clamped <c>grid_sample</c>: for each <c>(u, v)</c> pair in
    /// <paramref name="uv"/> the four neighbouring output pixels are attended and blended. <paramref name="queryMap"/> is
    /// <see cref="QueryMap"/>'s <c>[1, 256, H, W]</c>, <paramref name="featureRows"/> the feature map as <c>[fh · fw, channels]</c> rows (the DINOv3
    /// patch tokens), with <c>H = d · fh</c> and <c>W = d · fw</c>. Returns <c>[1, cells, channels]</c>.
    /// </summary>
    public static unsafe Tensor SampleUpsampled(
        ReadOnlySpan<float> queryMap, int height, int width, ReadOnlySpan<float> featureRows, int featureHeight, int featureWidth, int channels,
        ReadOnlySpan<float> uv)
    {
        if (queryMap.Length != (long)Dim * height * width)
            throw new ArgumentException($"{queryMap.Length} values do not form a [{Dim}, {height}, {width}] query map.", nameof(queryMap));
        if (featureHeight < Kernel || featureWidth < Kernel || height % featureHeight != 0 || width % featureWidth != 0)
            throw new ArgumentException(
                $"A {featureHeight}x{featureWidth} feature map must hold at least {Kernel} cells per axis and divide the {height}x{width} output.");
        if (channels <= 0 || channels % Heads != 0 || featureRows.Length != (long)featureHeight * featureWidth * channels)
            throw new ArgumentException($"{featureRows.Length} values do not form {featureHeight}x{featureWidth} feature rows of {channels} channels.", nameof(featureRows));
        if (uv.Length % 2 != 0)
            throw new ArgumentException("Sampling coordinates must be (u, v) pairs.", nameof(uv));

        int cells = uv.Length / 2;
        Tensor result = new(new TensorShape(1, cells, channels), DType.F32);
        float[] keys = PoolKeys(queryMap, height, width, featureHeight, featureWidth);
        float[] uvValues = uv.ToArray();
        // Spans cannot be captured by the parallel body, so it works from the pinned addresses of the inputs and of the result.
        fixed (float* queryPointer = queryMap, rowsPointer = featureRows, outputPointer = result.AsSpan<float>())
        {
            nint query = (nint)queryPointer, rows = (nint)rowsPointer, output = (nint)outputPointer;
            Parallel.For(0, cells, () => new Scratch(channels), (cell, _, scratch) =>
            {
                BorderTapsFor(uvValues[cell * 2], uvValues[cell * 2 + 1], height, width, scratch);
                Span<float> row = new((float*)output + (long)cell * channels, channels);
                row.Clear();
                for (int tap = 0; tap < 4; tap++)
                {
                    float weight = scratch.Weights[tap];
                    if (weight == 0f)
                        continue;
                    Attend((float*)query, height, width, keys, (float*)rows, featureHeight, featureWidth, channels, scratch.Y[tap], scratch.X[tap], scratch);
                    for (int c = 0; c < channels; c++)
                        row[c] += weight * scratch.Out[c];
                }

                return scratch;
            }, _ => { });
        }

        return result;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _owner?.Dispose();
    }

    private Tensor Branch(IBackend backend, Tensor image, string name, int kernel)
    {
        string p = $"{Prefix}{name}.";
        Tensor h = Conv(backend, image, _weights[p + "0.weight"], _weights[p + "0.bias"], kernel);
        for (int block = 1; block <= Blocks; block++)
        {
            string b = $"{p}{block}.";
            using Tensor current = h;
            using Tensor a = new(current.Shape, DType.F32);
            backend.GroupNormSilu(a, current, _weights[b + "norm1.weight"], _weights[b + "norm1.bias"], GroupCount, GroupNormEps);
            using Tensor c1 = Conv(backend, a, _weights[b + "conv1.weight"], _weights[b + "conv1.bias"], kernel);
            using Tensor d = new(c1.Shape, DType.F32);
            backend.GroupNormSilu(d, c1, _weights[b + "norm2.weight"], _weights[b + "norm2.bias"], GroupCount, GroupNormEps);
            h = Conv(backend, d, _weights[b + "conv2.weight"], _weights[b + "conv2.bias"], kernel);
        }

        return h;
    }

    // padding_mode="reflect": mirror the border explicitly, then convolve without padding.
    private static Tensor Conv(IBackend backend, Tensor x, Tensor weight, Tensor bias, int kernel)
    {
        int pad = kernel / 2;
        long h = x.Shape[2], w = x.Shape[3];
        Tensor? padded = null;
        try
        {
            Tensor source = x;
            if (pad > 0)
            {
                padded = new Tensor(new TensorShape(1, x.Shape[1], h + 2 * pad, w + 2 * pad), DType.F32);
                backend.Pad(padded, x, new PadSpec(PadMode.Reflect, [0, 0, pad, pad], [0, 0, pad, pad]));
                source = padded;
            }

            Tensor output = new(new TensorShape(1, weight.Shape[0], h, w), DType.F32);
            backend.Conv2D(output, source, weight, bias, 1, 1, 0, 0);
            return output;
        }
        finally
        {
            padded?.Dispose();
        }
    }

    private static Tensor AveragePool(Tensor map, int factor)
    {
        int size = (int)map.Shape[2], target = size / factor;
        Tensor pooled = new(new TensorShape(1, Dim, target, target), DType.F32);
        ReadOnlySpan<float> source = map.AsReadOnlySpan<float>();
        Span<float> output = pooled.AsSpan<float>();
        float inverse = 1f / (factor * factor);
        for (int c = 0; c < Dim; c++)
        {
            for (int y = 0; y < target; y++)
            {
                for (int x = 0; x < target; x++)
                {
                    float sum = 0f;
                    for (int dy = 0; dy < factor; dy++)
                    {
                        for (int dx = 0; dx < factor; dx++)
                            sum += source[(c * size + y * factor + dy) * size + x * factor + dx];
                    }

                    output[(c * target + y) * target + x] = sum * inverse;
                }
            }
        }

        return pooled;
    }

    // Upstream RoPE: per head of 64 channels, angle index m = c % 32; m < 16 uses the row coordinate with period m, else the column coordinate
    // with period m - 16; coordinates are (i + 0.5) / n · 2 - 1. Pairs (c, c + 32) rotate together.
    private unsafe void ApplyRope(Tensor map, int height, int width)
    {
        float[] periods = _weights[Prefix + "rope.periods"].AsReadOnlySpan<float>().ToArray();
        float[] rowAngle = new float[height * PeriodCount], columnAngle = new float[width * PeriodCount];
        for (int i = 0; i < height; i++)
        {
            float coord = ((i + 0.5f) / height) * 2f - 1f;
            for (int k = 0; k < PeriodCount; k++)
                rowAngle[i * PeriodCount + k] = 2f * MathF.PI * coord / periods[k];
        }

        for (int j = 0; j < width; j++)
        {
            float coord = ((j + 0.5f) / width) * 2f - 1f;
            for (int k = 0; k < PeriodCount; k++)
                columnAngle[j * PeriodCount + k] = 2f * MathF.PI * coord / periods[k];
        }

        int plane = height * width, half = HeadDim / 2;
        float* data = (float*)map.DataPointer;
        Parallel.For(0, Heads * half, index =>
        {
            int head = index / half, m = index % half;
            int first = (head * HeadDim + m) * plane, second = (head * HeadDim + m + half) * plane;
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    float angle = m < PeriodCount ? rowAngle[i * PeriodCount + m] : columnAngle[j * PeriodCount + m - PeriodCount];
                    float cos = MathF.Cos(angle), sin = MathF.Sin(angle);
                    int p = i * width + j;
                    float x1 = data[first + p], x2 = data[second + p];
                    data[first + p] = x1 * cos - x2 * sin;
                    data[second + p] = x2 * cos + x1 * sin;
                }
            }
        });
    }

    // adaptive_avg_pool2d of the query map to the feature size: the mean of each d x d block (sizes divide).
    private static unsafe float[] PoolKeys(ReadOnlySpan<float> queryMap, int height, int width, int featureHeight, int featureWidth)
    {
        int dh = height / featureHeight, dw = width / featureWidth, plane = height * width;
        float[] keys = new float[featureHeight * featureWidth * Dim];
        float inverse = 1f / (dh * dw);
        fixed (float* pinned = queryMap)
        {
            nint address = (nint)pinned;
            Parallel.For(0, Dim, c =>
            {
                float* map = (float*)address;
                for (int y = 0; y < featureHeight; y++)
                {
                    for (int x = 0; x < featureWidth; x++)
                    {
                        float sum = 0f;
                        for (int dy = 0; dy < dh; dy++)
                        {
                            int row = c * plane + (y * dh + dy) * width + x * dw;
                            for (int dx = 0; dx < dw; dx++)
                                sum += map[row + dx];
                        }

                        keys[(y * featureWidth + x) * Dim + c] = sum * inverse;
                    }
                }
            });
        }

        return keys;
    }

    private static void BorderTapsFor(float u, float v, int height, int width, Scratch scratch)
    {
        GridSampler.BorderTaps(u, v, height, width, out int x0, out int y0, out int x1, out int y1, out float tx, out float ty);
        scratch.X[0] = x0; scratch.Y[0] = y0; scratch.Weights[0] = (1f - ty) * (1f - tx);
        scratch.X[1] = x1; scratch.Y[1] = y0; scratch.Weights[1] = (1f - ty) * tx;
        scratch.X[2] = x0; scratch.Y[2] = y1; scratch.Weights[2] = ty * (1f - tx);
        scratch.X[3] = x1; scratch.Y[3] = y1; scratch.Weights[3] = ty * tx;
    }

    // One output pixel: per head, softmax over the 9 x 9 feature cells around the pixel's own cell (window slid inward at the borders, as NATTEN does),
    // scores q · k / sqrt(64), output the weighted sum of the cells' value channels for that head.
    private static unsafe void Attend(
        float* queryMap, int height, int width, float[] keys, float* rows, int featureHeight, int featureWidth, int channels, int y, int x, Scratch scratch)
    {
        int cellY = y / (height / featureHeight), cellX = x / (width / featureWidth);
        int startY = Math.Clamp(cellY - Kernel / 2, 0, featureHeight - Kernel), startX = Math.Clamp(cellX - Kernel / 2, 0, featureWidth - Kernel);
        int plane = height * width, valueWidth = channels / Heads;
        float scale = 1f / MathF.Sqrt(HeadDim);
        float[] q = scratch.Query, logits = scratch.Logits, output = scratch.Out;
        for (int head = 0; head < Heads; head++)
        {
            for (int d = 0; d < HeadDim; d++)
                q[d] = queryMap[(head * HeadDim + d) * plane + y * width + x];
            float max = float.NegativeInfinity;
            for (int a = 0; a < Kernel; a++)
            {
                for (int b = 0; b < Kernel; b++)
                {
                    int key = ((startY + a) * featureWidth + startX + b) * Dim + head * HeadDim;
                    float dot = 0f;
                    for (int d = 0; d < HeadDim; d++)
                        dot += q[d] * keys[key + d];
                    dot *= scale;
                    logits[a * Kernel + b] = dot;
                    if (dot > max)
                        max = dot;
                }
            }

            float sum = 0f;
            for (int i = 0; i < Kernel * Kernel; i++)
            {
                logits[i] = MathF.Exp(logits[i] - max);
                sum += logits[i];
            }

            Span<float> target = output.AsSpan(head * valueWidth, valueWidth);
            target.Clear();
            for (int a = 0; a < Kernel; a++)
            {
                for (int b = 0; b < Kernel; b++)
                {
                    float p = logits[a * Kernel + b] / sum;
                    int value = ((startY + a) * featureWidth + startX + b) * channels + head * valueWidth;
                    for (int c = 0; c < valueWidth; c++)
                        target[c] += p * rows[value + c];
                }
            }
        }
    }

    private static Dictionary<string, long[]> Spec()
    {
        Dictionary<string, long[]> spec = new(StringComparer.Ordinal) { [Prefix + "rope.periods"] = [PeriodCount] };
        foreach ((string name, int kernel) in new[] { ("encoder", 1), ("sem_encoder", 3) })
        {
            string p = $"{Prefix}{name}.";
            spec[p + "0.weight"] = [BranchChannels, 3, kernel, kernel];
            spec[p + "0.bias"] = [BranchChannels];
            for (int block = 1; block <= Blocks; block++)
            {
                string b = $"{p}{block}.";
                foreach (string conv in new[] { "conv1", "conv2" })
                {
                    spec[$"{b}{conv}.weight"] = [BranchChannels, BranchChannels, kernel, kernel];
                    spec[$"{b}{conv}.bias"] = [BranchChannels];
                }

                foreach (string norm in new[] { "norm1", "norm2" })
                {
                    spec[$"{b}{norm}.weight"] = [BranchChannels];
                    spec[$"{b}{norm}.bias"] = [BranchChannels];
                }
            }
        }

        return spec;
    }

    private sealed class Scratch
    {
        internal readonly int[] X = new int[4], Y = new int[4];
        internal readonly float[] Weights = new float[4];
        internal readonly float[] Query = new float[HeadDim];
        internal readonly float[] Logits = new float[Kernel * Kernel];
        internal readonly float[] Out;

        internal Scratch(int channels) => Out = new float[channels];
    }
}
