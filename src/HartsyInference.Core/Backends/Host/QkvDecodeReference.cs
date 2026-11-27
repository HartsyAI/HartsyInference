using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Decode;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>The executable form of the decode epilogue contract (<c>docs/BACKEND_OPS.md</c>) for a known position: each q and k
/// head is normalized as the norm kind says (<c>x · rsqrt(mean(x²) + eps) · w</c>), rotated with the rotary formula over row
/// <c>position</c> of the tables, then q goes to its output and k and v to slot <c>position</c> of their caches; an F16 cache stores
/// the value rounded to nearest with saturation at ±65504. The host has no device position buffer, so the position is explicit;
/// a slot at or past the cache length writes nothing.</summary>
public static unsafe class QkvDecodeReference
{
    public static void Apply(in QkvDecodeCall call, int position)
    {
        QkvDecodeContract.Validate(call);
        if (position < 0 || position >= call.CacheLength) return;
        int d = call.HeadDim;
        float* cos = (float*)call.Cos.DataPointer + (long)position * d;
        float* sin = (float*)call.Sin.DataPointer + (long)position * d;
        float[] q = Normalized(call.Q, call.Heads, d, call.Norm, call.QNormWeight, call.Eps);
        float[] k = Normalized(call.K, call.KvHeads, d, call.Norm, call.KNormWeight, call.Eps);
        fixed (float* pq = q)
        fixed (float* pk = k)
        {
            for (int h = 0; h < call.Heads; h++) Rotate(pq + (long)h * d, cos, sin, d, call.RotaryDim, call.Pairing);
            for (int h = 0; h < call.KvHeads; h++) Rotate(pk + (long)h * d, cos, sin, d, call.RotaryDim, call.Pairing);
        }
        q.AsSpan().CopyTo(new Span<float>((float*)call.QOut.DataPointer, (int)call.QWidth));
        float* v = (float*)call.V.Tensor.DataPointer + call.V.Offset;
        for (int h = 0; h < call.KvHeads; h++)
        {
            long slot = ((long)h * call.CacheLength + position) * d;
            Store(call.KCache, slot, k.AsSpan(h * d, d));
            Store(call.VCache, slot, new ReadOnlySpan<float>(v + (long)h * d, d));
        }
    }

    /// <summary>The section's heads after the norm kind's RMS normalization.</summary>
    private static float[] Normalized(in QkvSection section, int heads, int d, QkvNormKind norm, Tensor? weight, float eps)
    {
        float[] values = new float[heads * d];
        new ReadOnlySpan<float>((float*)section.Tensor.DataPointer + section.Offset, values.Length).CopyTo(values);
        if (norm == QkvNormKind.None) return values;
        float* w = (float*)weight!.DataPointer;
        int rowWidth = norm == QkvNormKind.PerHead ? d : values.Length;
        for (int row = 0; row < values.Length / rowWidth; row++)
        {
            Span<float> x = values.AsSpan(row * rowWidth, rowWidth);
            float sumSquares = 0f;
            for (int i = 0; i < rowWidth; i++) sumSquares += x[i] * x[i];
            float invRms = 1f / MathF.Sqrt(sumSquares / rowWidth + eps);
            for (int i = 0; i < rowWidth; i++) x[i] = x[i] * invRms * w[i];
        }
        return values;
    }

    /// <summary>The rotary formula of <see cref="RotaryReference"/> on one head: <c>x0' = x0·c0 − x1·s0</c>, <c>x1' = x1·c1 + x0·s1</c>.</summary>
    private static void Rotate(float* x, float* cos, float* sin, int d, int rotaryDim, RotaryPairing pairing)
    {
        if (pairing == RotaryPairing.Interleaved)
        {
            for (int p = 0; p < d / 2; p++)
            {
                float x0 = x[2 * p], x1 = x[2 * p + 1];
                float c = cos[p], s = sin[p];
                x[2 * p] = x0 * c - x1 * s;
                x[2 * p + 1] = x1 * c + x0 * s;
            }
            return;
        }
        int rotated = rotaryDim == 0 ? d : rotaryDim;
        int half = rotated / 2;
        for (int i = 0; i < half; i++)
        {
            float x0 = x[i], x1 = x[i + half];
            x[i] = x0 * cos[i] - x1 * sin[i];
            x[i + half] = x1 * cos[i + half] + x0 * sin[i + half];
        }
    }

    private static void Store(Tensor cache, long slot, ReadOnlySpan<float> values)
    {
        if (cache.DType == DType.F32)
        {
            values.CopyTo(new Span<float>((float*)cache.DataPointer + slot, values.Length));
            return;
        }
        Half* dst = (Half*)cache.DataPointer + slot;
        for (int i = 0; i < values.Length; i++) dst[i] = (Half)Math.Clamp(values[i], -65504f, 65504f);
    }
}
