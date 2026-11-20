using System.Numerics;
using HartsyInference.Audio.Dsp.Alignment;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Whisper;

/// <summary>Collects the decoder's cross-attention for a set of alignment heads during one decoder pass and keeps their mean.
/// The decoder's fused attention never surfaces its weights, so the probe recomputes them from the query and the cached
/// encoder keys for the heads it watches. Each head is folded into one running sum as it is captured, so the memory held
/// is one matrix however many heads are watched. A probe records one pass and is not thread-safe.</summary>
public sealed class WhisperAttentionProbe
{
    /// <summary>The median window along time, in frames.</summary>
    private const int SmoothingWidth = 7;

    private readonly IReadOnlyList<WhisperAlignmentHead> _heads;
    private float[]? _sum;
    private int _captured;

    /// <summary>Watches <paramref name="heads"/> over the first <paramref name="frames"/> encoder positions.</summary>
    public WhisperAttentionProbe(IReadOnlyList<WhisperAlignmentHead> heads, int frames)
    {
        ArgumentNullException.ThrowIfNull(heads);
        if (heads.Count == 0)
        {
            throw new ArgumentException("At least one alignment head is needed.", nameof(heads));
        }
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(frames);
        _heads = heads;
        Frames = frames;
    }

    /// <summary>The heads watched.</summary>
    public IReadOnlyList<WhisperAlignmentHead> Heads => _heads;

    /// <summary>Audio frames each row spans: encoder positions at 50 per second, cut to the audio's own length before the softmax.</summary>
    public int Frames { get; }

    /// <summary>Decoder positions captured, the rows of <see cref="MeanAttention"/>.</summary>
    public int Rows { get; private set; }

    /// <summary>The mean over the watched heads of each head's attention, a row-major <c>[Rows, Frames]</c> matrix. Every
    /// head's softmax is normalized to zero mean and unit variance across the rows of each frame, so a frame every token
    /// attends to alike carries no signal, then median-filtered along time, as OpenAI's Whisper word timestamps do.</summary>
    public float[] MeanAttention()
    {
        if (_sum is null || _captured != _heads.Count)
        {
            throw new InvalidOperationException($"The decoder has captured {_captured} of {_heads.Count} watched heads.");
        }
        float[] mean = new float[_sum.Length];
        float inverse = 1f / _heads.Count;
        for (int i = 0; i < mean.Length; i++)
        {
            mean[i] = _sum[i] * inverse;
        }
        return mean;
    }

    /// <summary>Whether any watched head lives in decoder layer <paramref name="layer"/>.</summary>
    internal bool WantsLayer(int layer)
    {
        foreach (WhisperAlignmentHead head in _heads)
        {
            if (head.Layer == layer)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Records the watched heads of <paramref name="layer"/> for a decoder pass of <paramref name="rows"/>
    /// positions from this layer's head-major query <c>[1, H, rows, D]</c> and encoder keys <c>[1, H, E,
    /// D]</c>.</summary>
    internal void Capture(IBackend backend, int layer, Tensor queryHeadMajor, Tensor encoderKeys, int rows, float scale)
    {
        int heads = (int)queryHeadMajor.Shape[1];
        int headDim = (int)queryHeadMajor.Shape[3];
        int encoderFrames = (int)encoderKeys.Shape[2];
        if (Frames > encoderFrames)
        {
            throw new InvalidOperationException($"The probe spans {Frames} frames but the encoder produced {encoderFrames}.");
        }
        if (Rows != 0 && Rows != rows)
        {
            throw new InvalidOperationException($"A probe records one decoder pass: it saw {Rows} rows, then {rows}.");
        }
        Rows = rows;
        _sum ??= new float[rows * Frames];

        float[] keys = new float[heads * encoderFrames * headDim];
        backend.ReadResidentInto(encoderKeys, keys);
        float[] queries = new float[heads * rows * headDim];
        backend.ReadResidentInto(queryHeadMajor, queries);

        float[] weights = new float[rows * Frames];
        int frames = Frames;
        foreach (WhisperAlignmentHead watched in _heads)
        {
            if (watched.Layer != layer)
            {
                continue;
            }
            int headKeysOffset = watched.Head * encoderFrames * headDim;
            int headQueriesOffset = watched.Head * rows * headDim;
            CpuParallel.For(rows, (long)rows * frames * headDim, r =>
            {
                ReadOnlySpan<float> query = queries.AsSpan(headQueriesOffset + r * headDim, headDim);
                Span<float> row = weights.AsSpan(r * frames, frames);
                for (int f = 0; f < frames; f++)
                {
                    row[f] = Dot(query, keys.AsSpan(headKeysOffset + f * headDim, headDim)) * scale;
                }
                Softmax(row);
            });
            NormalizeColumns(weights, rows, frames);
            MedianFilter.Rows(weights, rows, frames, SmoothingWidth);
            for (int i = 0; i < weights.Length; i++)
            {
                _sum[i] += weights[i];
            }
            _captured++;
        }
    }

    /// <summary>Gives each column of the row-major <c>[rows, cols]</c> matrix zero mean and unit variance across its
    /// rows; a flat column becomes zero.</summary>
    private static void NormalizeColumns(Span<float> data, int rows, int cols)
    {
        for (int c = 0; c < cols; c++)
        {
            double mean = 0d;
            for (int r = 0; r < rows; r++)
            {
                mean += data[r * cols + c];
            }
            mean /= rows;
            double variance = 0d;
            for (int r = 0; r < rows; r++)
            {
                double delta = data[r * cols + c] - mean;
                variance += delta * delta;
            }
            double deviation = Math.Sqrt(variance / rows);
            for (int r = 0; r < rows; r++)
            {
                data[r * cols + c] = deviation > 1e-9 ? (float)((data[r * cols + c] - mean) / deviation) : 0f;
            }
        }
    }

    /// <summary>The dot product of two equally long vectors.</summary>
    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int i = 0;
        float sum = 0f;
        if (Vector.IsHardwareAccelerated && a.Length >= Vector<float>.Count)
        {
            Vector<float> acc = Vector<float>.Zero;
            for (; i <= a.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                acc += new Vector<float>(a.Slice(i)) * new Vector<float>(b.Slice(i));
            }
            sum = Vector.Sum(acc);
        }
        for (; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /// <summary>Normalizes <paramref name="row"/> in place to <c>exp(x - max) / sum</c>.</summary>
    private static void Softmax(Span<float> row)
    {
        float max = float.NegativeInfinity;
        foreach (float value in row)
        {
            max = Math.Max(max, value);
        }
        float sum = 0f;
        for (int i = 0; i < row.Length; i++)
        {
            row[i] = MathF.Exp(row[i] - max);
            sum += row[i];
        }
        float inverse = 1f / sum;
        for (int i = 0; i < row.Length; i++)
        {
            row[i] *= inverse;
        }
    }
}
