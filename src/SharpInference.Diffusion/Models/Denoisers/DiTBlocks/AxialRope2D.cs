using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>2-D axial rotary embedding in lucidrains' "pixel" mode (vendored by Oasis): per axis,
/// <c>dimPerAxis/2</c> frequencies <c>linspace(1, maxFreq/2, n)·π</c> over normalized coordinates
/// <c>linspace(−1, 1, axisLen)</c>, each duplicated across its interleaved pair; the H-axis rotates the first
/// <c>dimPerAxis</c> head dims, the W-axis the next, and any remaining head dims pass through unrotated (Oasis' VAE
/// rotates 32 of 64 dims, its DiT spatial attention all 64).</summary>
public sealed unsafe class AxialRope2D
{
    private readonly int _dimPerAxis;
    private readonly double[] _freqs;   // dimPerAxis/2 angular frequencies (shared by both axes)

    public AxialRope2D(int dimPerAxis, double maxFreq)
    {
        if (dimPerAxis % 2 != 0) throw new ArgumentException("dimPerAxis must be even.", nameof(dimPerAxis));
        _dimPerAxis = dimPerAxis;
        int n = dimPerAxis / 2;
        _freqs = new double[n];
        for (int i = 0; i < n; i++)
            _freqs[i] = (n == 1 ? 1.0 : 1.0 + (maxFreq / 2.0 - 1.0) * i / (n - 1)) * Math.PI;
    }

    /// <summary>Head dims rotated (both axes).</summary>
    public int RotDim => 2 * _dimPerAxis;

    /// <summary>Builds cos/sin <c>[h·w, RotDim]</c> for the token grid in (h, w) row-major order.</summary>
    public (Tensor Cos, Tensor Sin) Build(int h, int w)
    {
        Tensor cos = new Tensor(new TensorShape(h * w, RotDim), DType.F32);
        Tensor sin = new Tensor(new TensorShape(h * w, RotDim), DType.F32);
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;
        for (int y = 0; y < h; y++)
        {
            double cy = h == 1 ? -1.0 : -1.0 + 2.0 * y / (h - 1);
            for (int x = 0; x < w; x++)
            {
                double cx = w == 1 ? -1.0 : -1.0 + 2.0 * x / (w - 1);
                long off = ((long)y * w + x) * RotDim;
                int d = 0;
                FillAxis(cp, sp, off, ref d, cy);
                FillAxis(cp, sp, off, ref d, cx);
            }
        }
        return (cos, sin);
    }

    private void FillAxis(float* cp, float* sp, long off, ref int d, double coord)
    {
        foreach (double f in _freqs)
        {
            double angle = coord * f;
            float c = (float)Math.Cos(angle), s = (float)Math.Sin(angle);
            cp[off + d] = c; sp[off + d] = s;
            cp[off + d + 1] = c; sp[off + d + 1] = s;
            d += 2;
        }
    }

    /// <summary>Rotates the first <see cref="RotDim"/> dims of each head in place. <paramref name="x"/> is
    /// <c>[S, heads, headDim]</c> contiguous, with row s using grid row s of <paramref name="cos"/>/<paramref name="sin"/>.</summary>
    public void ApplyRotary(Tensor x, Tensor cos, Tensor sin, int heads, int headDim)
    {
        int seq = (int)x.Shape[0];
        float* xp = (float*)x.DataPointer;
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;
        int pairs = RotDim / 2;
        for (int s = 0; s < seq; s++)
        {
            long gOff = (long)s * RotDim;
            for (int h = 0; h < heads; h++)
            {
                long xOff = ((long)s * heads + h) * headDim;
                for (int i = 0; i < pairs; i++)
                {
                    int i0 = 2 * i;
                    float re = xp[xOff + i0], im = xp[xOff + i0 + 1];
                    float c = cp[gOff + i0], sn = sp[gOff + i0];
                    xp[xOff + i0] = re * c - im * sn;
                    xp[xOff + i0 + 1] = re * sn + im * c;
                }
            }
        }
    }
}
