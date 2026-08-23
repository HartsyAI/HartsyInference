using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Ssm;

/// <summary>Host-side primitives shared by the RWKV recurrences (<see cref="RwkvModel"/> v6, <see cref="Rwkv7Model"/> v7), whose block math diverges but whose projection/norm/token-shift glue is identical.</summary>
internal static unsafe class RwkvOps
{
    /// <summary>x[t,in] @ <paramref name="weight"/>[out,in]ᵀ → [t,out] via <see cref="IBackend.Linear"/> (outDim read from the weight shape).</summary>
    public static float[] Lin(IBackend backend, float[] x, int t, int inDim, Tensor weight)
    {
        int outDim = (int)weight.Shape[0];
        using Tensor xt = new(new TensorShape(1, t, inDim), DType.F32);
        fixed (float* s = x) Buffer.MemoryCopy(s, (void*)xt.DataPointer, (long)t * inDim * 4, (long)t * inDim * 4);
        using Tensor o = new(new TensorShape(1, t, outDim), DType.F32);
        backend.Linear(o, xt, weight, null);
        backend.Sync();
        float[] r = new float[(long)t * outDim];
        fixed (float* d = r) Buffer.MemoryCopy((void*)o.DataPointer, d, r.Length * 4L, r.Length * 4L);
        return r;
    }

    /// <summary>In-place affine LayerNorm over the last dim of a row-major <c>[t, d]</c> host buffer, accumulating mean/variance in double.</summary>
    public static void LayerNorm(float[] x, int t, int d, float eps, float* w, float* b)
    {
        fixed (float* xp = x)
            for (int s = 0; s < t; s++)
            {
                float* row = xp + (long)s * d;
                double mean = 0; for (int c = 0; c < d; c++) mean += row[c]; mean /= d;
                double var = 0; for (int c = 0; c < d; c++) { double dd = row[c] - mean; var += dd * dd; } var /= d;
                float inv = (float)(1.0 / Math.Sqrt(var + eps));
                for (int c = 0; c < d; c++) row[c] = (float)((row[c] - mean) * inv) * w[c] + b[c];
            }
    }

    /// <summary>Token-shift difference <c>sx[s] = xx[s-1] - xx[s]</c>, where <c>xx[-1]</c> comes from <paramref name="prevRow"/> (the carried last row of the previous call); overwrites <paramref name="prevRow"/> with this call's last row.</summary>
    public static float[] ShiftDiff(float[] xx, int t, int d, float[] prevRow)
    {
        float[] sx = new float[(long)t * d];
        for (int s = 0; s < t; s++)
            for (int c = 0; c < d; c++) sx[s * d + c] = (s == 0 ? prevRow[c] : xx[(s - 1) * d + c]) - xx[s * d + c];
        Array.Copy(xx, ((long)t - 1) * d, prevRow, 0, d);
        return sx;
    }
}
