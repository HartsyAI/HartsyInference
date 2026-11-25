using System.Numerics;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Small F32 host kernels and tensor marshalling shared by the reference transformer modules.</summary>
internal static class HostMath
{
    /// <summary><c>y[t,o] = sum_i x[t,i] * w[o,i]</c> for <paramref name="rows"/> rows.</summary>
    /// <remarks>Each output element is one <see cref="Dot"/>, whose summation order is fixed, so running them across cores changes the speed and nothing else.</remarks>
    public static float[] Linear(ReadOnlySpan<float> x, ReadOnlySpan<float> w, int rows, int inDim, int outDim)
    {
        if (x.Length != (long)rows * inDim || w.Length != (long)outDim * inDim) throw new ArgumentException("Linear operands do not match the stated shape.");
        float[] y = new float[checked(rows * outDim)];
        LinearInto(x, w, rows, inDim, outDim, y, outDim, 0);
        return y;
    }

    /// <summary>The same product for <paramref name="chunkRows"/> weight rows only, written to columns <c>[firstRow, firstRow + chunkRows)</c> of each <paramref name="y"/> row of width <paramref name="outDim"/>.</summary>
    /// <remarks>Lets a weight be streamed through a small window; every element is the same <see cref="Dot"/>, so the result equals the whole-matrix product bit for bit.</remarks>
    public static unsafe void LinearInto(ReadOnlySpan<float> x, ReadOnlySpan<float> w, int rows, int inDim, int chunkRows, float[] y, int outDim, int firstRow)
    {
        if (x.Length != (long)rows * inDim || w.Length != (long)chunkRows * inDim) throw new ArgumentException("Linear operands do not match the stated shape.");
        if (firstRow < 0 || firstRow + chunkRows > outDim || y.Length != (long)rows * outDim) throw new ArgumentException("The weight window does not fit the output.");
        const int Block = 32;
        int total = checked(rows * chunkRows), blocks = (total + Block - 1) / Block;
        fixed (float* xp = x)
        fixed (float* wp = w)
        fixed (float* yp = y)
        {
            nint xa = (nint)xp, wa = (nint)wp, ya = (nint)yp;
            CpuParallel.For(blocks, (long)total * inDim, block =>
            {
                float* xs = (float*)xa, ws = (float*)wa, ys = (float*)ya;
                int end = Math.Min(total, (block + 1) * Block);
                for (int index = block * Block; index < end; index++)
                {
                    int t = index / chunkRows, o = index % chunkRows;
                    float* xr = xs + (long)t * inDim, wr = ws + (long)o * inDim;
                    ys[(long)t * outDim + firstRow + o] = Dot(xr, wr, inDim);
                }
            });
        }
    }

    /// <summary>The inner product of two equal-length spans, in the fixed summation order <see cref="Linear"/> uses.</summary>
    public static unsafe float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length) throw new ArgumentException("Dot operands must have the same length.");
        fixed (float* ap = a)
        fixed (float* bp = b)
            return Dot(ap, bp, a.Length);
    }

    /// <summary>One dot product in a fixed order: four accumulators of <see cref="Vector{T}"/> lanes, then the scalar tail.</summary>
    /// <remarks>The order depends only on <paramref name="n"/> and the machine's vector width, so one machine gives one value for the same
    /// inputs however the caller splits its outputs.</remarks>
    private static unsafe float Dot(float* a, float* b, int n)
    {
        int i = 0, width = Vector<float>.Count, step = width * 4;
        float sum = 0f;
        if (Vector.IsHardwareAccelerated && n >= step)
        {
            Vector<float> s0 = Vector<float>.Zero, s1 = s0, s2 = s0, s3 = s0;
            for (; i <= n - step; i += step)
            {
                s0 += Vector.Load(a + i) * Vector.Load(b + i);
                s1 += Vector.Load(a + i + width) * Vector.Load(b + i + width);
                s2 += Vector.Load(a + i + 2 * width) * Vector.Load(b + i + 2 * width);
                s3 += Vector.Load(a + i + 3 * width) * Vector.Load(b + i + 3 * width);
            }
            sum = Vector.Sum((s0 + s1) + (s2 + s3));
        }
        for (; i < n; i++) sum += a[i] * b[i];
        return sum;
    }

    /// <summary>In-place RMS norm per row of width <paramref name="dim"/>: <c>w * x * rsqrt(mean(x^2) + eps)</c>.</summary>
    public static void RmsNormRows(Span<float> x, ReadOnlySpan<float> weight, int dim, float eps)
    {
        if (weight.Length != dim || x.Length % dim != 0) throw new ArgumentException("RmsNorm operands do not match the stated width.");
        for (int off = 0; off < x.Length; off += dim)
        {
            Span<float> row = x.Slice(off, dim);
            float sq = 0f;
            for (int i = 0; i < dim; i++) sq += row[i] * row[i];
            float scale = 1f / MathF.Sqrt(sq / dim + eps);
            for (int i = 0; i < dim; i++) row[i] = weight[i] * (row[i] * scale);
        }
    }

    /// <summary>An F32 tensor holding a copy of <paramref name="values"/> with the given shape.</summary>
    public static Tensor Tensor(ReadOnlySpan<float> values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    /// <summary>An I32 tensor holding a copy of <paramref name="values"/> with the given shape.</summary>
    public static Tensor Tensor(ReadOnlySpan<int> values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.I32);
        values.CopyTo(t.AsSpan<int>());
        return t;
    }
}
