using System.Numerics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Whisper;

/// <summary>Collects the decoder's cross-attention weights for a set of alignment heads during one decoder pass. The decoder runs fused attention that never surfaces its weights, so the probe recomputes them from the query and the cached encoder keys for the heads it was asked to watch.</summary>
public sealed class WhisperAttentionProbe
{
    private readonly IReadOnlyList<WhisperAlignmentHead> _heads;
    private readonly float[]?[] _weights;

    /// <summary>The heads watched, in the order <see cref="HeadWeights"/> indexes them.</summary>
    public IReadOnlyList<WhisperAlignmentHead> Heads => _heads;

    /// <summary>Audio frames each row spans: encoder positions at 50 per second, cut to the audio's own length before the softmax.</summary>
    public int Frames { get; }

    /// <summary>Decoder positions captured so far, the rows of every head's weights.</summary>
    public int Rows { get; private set; }

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
        _weights = new float[heads.Count][];
    }

    /// <summary>The softmaxed attention of watched head <paramref name="index"/> as a row-major <c>[Rows, Frames]</c> matrix.</summary>
    public ReadOnlySpan<float> HeadWeights(int index)
        => _weights[index] ?? throw new InvalidOperationException("The decoder has not run with this probe yet.");

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

    /// <summary>Records the watched heads of <paramref name="layer"/> for a decoder pass of <paramref name="rows"/> positions from this layer's head-major query <c>[1, H, rows, D]</c> and encoder keys <c>[1, H, E, D]</c>.</summary>
    internal unsafe void Capture(IBackend backend, int layer, Tensor queryHeadMajor, Tensor encoderKeys, int rows, float scale)
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

        float[] keys = new float[heads * encoderFrames * headDim];
        backend.ReadResidentInto(encoderKeys, keys);
        float[] queries = new float[heads * rows * headDim];
        new ReadOnlySpan<float>((void*)queryHeadMajor.DataPointer, queries.Length).CopyTo(queries);

        for (int w = 0; w < _heads.Count; w++)
        {
            if (_heads[w].Layer != layer)
            {
                continue;
            }
            int head = _heads[w].Head;
            float[] weights = new float[rows * Frames];
            int headKeysOffset = head * encoderFrames * headDim;
            int headQueriesOffset = head * rows * headDim;
            int frames = Frames;
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
            _weights[w] = weights;
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
