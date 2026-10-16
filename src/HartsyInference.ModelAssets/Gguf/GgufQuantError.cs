using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf.Codecs;

namespace HartsyInference.ModelAssets.Gguf;

/// <summary>Accumulates the error a re-quantization adds over the tensors it converts: the largest and mean absolute error and
/// the error energy relative to the reference's, so a load can report what a precision change cost.</summary>
public sealed class GgufQuantError
{
    private readonly object _gate = new();
    private double _sumAbs, _sumSq, _referenceSq;
    private long _count;

    /// <summary>Largest absolute difference seen.</summary>
    public float MaxAbs { get; private set; }

    /// <summary>Mean absolute difference over every element compared.</summary>
    public double MeanAbs => _count == 0 ? 0 : _sumAbs / _count;

    /// <summary>Root of the error energy over the reference energy.</summary>
    public double RelativeRms => _referenceSq == 0 ? 0 : Math.Sqrt(_sumSq / _referenceSq);

    /// <summary>Elements compared.</summary>
    public long Count => _count;

    /// <summary>Folds in the difference between <paramref name="reference"/> and <paramref name="decoded"/>.</summary>
    public void Accumulate(ReadOnlySpan<float> reference, ReadOnlySpan<float> decoded)
    {
        if (reference.Length != decoded.Length) throw new ArgumentException("The reference and the decoded values differ in length.", nameof(decoded));
        double sumAbs = 0, sumSq = 0, refSq = 0;
        float max = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            float diff = MathF.Abs(reference[i] - decoded[i]);
            if (diff > max) max = diff;
            sumAbs += diff;
            sumSq += (double)diff * diff;
            refSq += (double)reference[i] * reference[i];
        }
        lock (_gate)
        {
            MaxAbs = MathF.Max(MaxAbs, max);
            _sumAbs += sumAbs;
            _sumSq += sumSq;
            _referenceSq += refSq;
            _count += reference.Length;
        }
    }

    /// <summary>Quantizes <paramref name="source"/> (F32, not consumed) to <paramref name="target"/> and folds the round-trip error in.</summary>
    public unsafe Tensor Quantize(Tensor source, DType target)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source.DType != DType.F32) throw new ArgumentException($"The reference must be F32, not {source.DType.Name}.", nameof(source));
        Tensor quantized = GgufQuantizer.Quantize(source, target);
        try
        {
            long count = source.ElementCount;
            float[] decoded = new float[Math.Min(count, 1L << 22)];
            IGgufCodec codec = GgufCodecRegistry.Get(target);
            long blockElements = target.BlockElementCount;
            for (long done = 0; done < count;)
            {
                long chunk = Math.Min(count - done, decoded.Length / blockElements * blockElements);
                fixed (float* dst = decoded)
                {
                    codec.DequantizeToF32((byte*)quantized.DataPointer + target.ComputeByteCount(done), dst, chunk);
                }
                Accumulate(new ReadOnlySpan<float>((float*)source.DataPointer + done, (int)chunk), decoded.AsSpan(0, (int)chunk));
                done += chunk;
            }
            return quantized;
        }
        catch
        {
            quantized.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public override string ToString() =>
        $"max |Δ| {MaxAbs:G3}, mean |Δ| {MeanAbs:G3}, relative RMS {RelativeRms:G3} over {Count:N0} values";
}
