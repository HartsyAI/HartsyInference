using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>CPU reference for rotating a slice of each vector as adjacent (interleaved) pairs, <c>view_as_complex</c> style.</summary>
public static class RopeInterleavedOffsetReference
{
    /// <summary>Checks the operands of <see cref="Apply"/>; shared by every backend.</summary>
    public static void Validate(Tensor x, Tensor cos, Tensor sin, int rotaryDim, int dimOffset)
    {
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("ApplyRopeInterleaved with a dimension offset supports F32 only.");
        if (x.Shape.Rank != 3 && x.Shape.Rank != 4)
            throw new ArgumentException("x must be [batch, len, dim] or [batch, len, heads, dim].", nameof(x));
        long dim = x.Shape[x.Shape.Rank - 1];
        if (rotaryDim < 2 || (rotaryDim & 1) != 0) throw new ArgumentOutOfRangeException(nameof(rotaryDim), "rotaryDim must be even and positive.");
        if (dimOffset < 0 || (dimOffset & 1) != 0 || dimOffset + rotaryDim > dim)
            throw new ArgumentOutOfRangeException(nameof(dimOffset),
                $"dimOffset must be even and keep [{dimOffset},{dimOffset + rotaryDim}) inside {dim}.");
        long positions = x.Shape[0] * x.Shape[1];
        if (cos.ElementCount != positions * (rotaryDim / 2) || sin.ElementCount != cos.ElementCount)
            throw new ArgumentException($"cos and sin must hold [{x.Shape[0]},{x.Shape[1]},{rotaryDim / 2}] entries.");
    }

    /// <summary>Rotates <c>x[.., dimOffset + 2i]</c> and <c>x[.., dimOffset + 2i + 1]</c> by angle <c>i</c> of the position.</summary>
    /// <remarks>cos and sin are <c>[batch, len, rotaryDim/2]</c>. Pass the negated sin for the inverse rotation.</remarks>
    public static unsafe void Apply(Tensor x, Tensor cos, Tensor sin, int rotaryDim, int dimOffset)
    {
        Validate(x, cos, sin, rotaryDim, dimOffset);
        int rank = x.Shape.Rank, dim = (int)x.Shape[rank - 1], half = rotaryDim / 2;
        int heads = rank == 4 ? (int)x.Shape[2] : 1;
        long positions = x.Shape[0] * x.Shape[1];
        float* xp = (float*)x.DataPointer, cp = (float*)cos.DataPointer, sp = (float*)sin.DataPointer;
        for (long pos = 0; pos < positions; pos++)
            for (int h = 0; h < heads; h++)
            {
                float* v = xp + (pos * heads + h) * dim + dimOffset;
                for (int i = 0; i < half; i++)
                {
                    float c = cp[pos * half + i], s = sp[pos * half + i];
                    float even = v[2 * i], odd = v[2 * i + 1];
                    v[2 * i] = even * c - odd * s;
                    v[2 * i + 1] = even * s + odd * c;
                }
            }
    }
}
