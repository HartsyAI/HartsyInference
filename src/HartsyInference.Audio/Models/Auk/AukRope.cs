using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Auk;

/// <summary>Builds the duplicated-pair cos/sin tables that <c>IBackend.WanRopeInterleaved</c> consumes from an explicit inv_freq vector.</summary>
public static unsafe class AukRope
{
    /// <summary>The closed-form x_transformers inv_freq, <c>theta^(-2i/headDim)</c>; AuK checkpoints ship their own bf16-rounded buffer instead.</summary>
    public static float[] TheoryInvFreq(int headDim, float theta = 10_000f)
    {
        float[] invFreq = new float[headDim / 2];
        for (int i = 0; i < invFreq.Length; i++) invFreq[i] = (float)(1.0 / Math.Pow(theta, 2.0 * i / headDim));
        return invFreq;
    }

    /// <summary>Returns <c>[maxPos, 2*invFreq.Length]</c> F32 tables with the angle for pair i at columns 2i and 2i+1; the caller owns both tensors.</summary>
    public static (Tensor Cos, Tensor Sin) BuildTables(ReadOnlySpan<float> invFreq, int maxPos)
    {
        if (invFreq.IsEmpty) throw new ArgumentException("inv_freq must not be empty.", nameof(invFreq));
        if (maxPos < 1) throw new ArgumentOutOfRangeException(nameof(maxPos));
        int headDim = 2 * invFreq.Length;
        Tensor cos = new(new TensorShape(maxPos, headDim), DType.F32);
        Tensor sin = new(new TensorShape(maxPos, headDim), DType.F32);
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;
        for (int p = 0; p < maxPos; p++)
        {
            for (int i = 0; i < invFreq.Length; i++)
            {
                double angle = p * (double)invFreq[i];
                int idx = p * headDim + 2 * i;
                cp[idx] = cp[idx + 1] = (float)Math.Cos(angle);
                sp[idx] = sp[idx + 1] = (float)Math.Sin(angle);
            }
        }
        return (cos, sin);
    }
}
