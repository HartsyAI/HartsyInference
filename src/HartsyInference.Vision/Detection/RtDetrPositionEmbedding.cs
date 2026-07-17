using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>Builds the RT-DETR 2D sinusoidal position embedding used by the AIFI encoder block. Each
/// grid position <c>(h, w)</c> (row-major, h-outer) gets a <c>[sin_h | cos_h | sin_w | cos_w]</c>
/// vector; frequency arithmetic is done in double precision to match the reference exactly.</summary>
public static class RtDetrPositionEmbedding
{
    /// <summary>Returns a <c>[1, H·W, embedDim]</c> position embedding tensor. <paramref name="embedDim"/>
    /// must be divisible by 4.</summary>
    public static unsafe Tensor Build2dSincos(int height, int width, int embedDim, float temperature)
    {
        if (embedDim % 4 != 0)
            throw new ArgumentException($"embedDim must be divisible by 4, got {embedDim}.", nameof(embedDim));
        int posDim = embedDim / 4;
        int s = height * width;
        double[] omega = new double[posDim];
        for (int i = 0; i < posDim; i++)
            omega[i] = 1.0 / Math.Pow(temperature, (double)i / posDim);

        Tensor result = new Tensor(new TensorShape(1, s, embedDim), DType.F32);
        float* dst = (float*)result.DataPointer;
        for (int gy = 0; gy < height; gy++)
            for (int gx = 0; gx < width; gx++)
            {
                long token = (long)gy * width + gx;
                float* row = dst + token * embedDim;
                for (int i = 0; i < posDim; i++)
                {
                    double eh = gy * omega[i];
                    double ew = gx * omega[i];
                    row[i] = (float)Math.Sin(eh);
                    row[posDim + i] = (float)Math.Cos(eh);
                    row[2 * posDim + i] = (float)Math.Sin(ew);
                    row[3 * posDim + i] = (float)Math.Cos(ew);
                }
            }
        return result;
    }
}
