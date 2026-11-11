namespace HartsyInference.Engine.Tests.Worlds;

/// <summary>Frame comparison metrics for the world end-to-end tests.</summary>
internal static class FrameMetrics
{
    /// <summary>PSNR in dB of two RGB24 frames; infinity when they are identical.</summary>
    public static double Psnr(byte[] expected, byte[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        double sum = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = expected[i] - actual[i];
            sum += d * d;
        }
        double mse = sum / expected.Length;
        return mse == 0 ? double.PositiveInfinity : 10.0 * Math.Log10(255.0 * 255.0 / mse);
    }

    /// <summary>Mean SSIM on BT.601 luma with an 11×11 Gaussian window (σ 1.5) over the valid region, as Wang et al.'s reference.</summary>
    public static double Ssim(byte[] expected, byte[] actual, int width, int height)
    {
        const double C1 = 0.01 * 255 * 0.01 * 255, C2 = 0.03 * 255 * 0.03 * 255;
        double[] x = Luma(expected, width, height), y = Luma(actual, width, height);
        double[] xx = new double[x.Length], yy = new double[x.Length], xy = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            xx[i] = x[i] * x[i];
            yy[i] = y[i] * y[i];
            xy[i] = x[i] * y[i];
        }
        double[] kernel = GaussianKernel();
        (double[] mx, int ow, int oh) = Filter(x, width, height, kernel);
        double[] my = Filter(y, width, height, kernel).Values;
        double[] sxx = Filter(xx, width, height, kernel).Values;
        double[] syy = Filter(yy, width, height, kernel).Values;
        double[] sxy = Filter(xy, width, height, kernel).Values;
        double total = 0;
        for (int i = 0; i < mx.Length; i++)
        {
            double vx = sxx[i] - mx[i] * mx[i], vy = syy[i] - my[i] * my[i], cov = sxy[i] - mx[i] * my[i];
            total += (2 * mx[i] * my[i] + C1) * (2 * cov + C2) / ((mx[i] * mx[i] + my[i] * my[i] + C1) * (vx + vy + C2));
        }
        return total / (ow * oh);
    }

    private static double[] Luma(byte[] rgb, int width, int height)
    {
        double[] luma = new double[width * height];
        for (int i = 0; i < luma.Length; i++)
            luma[i] = 0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2];
        return luma;
    }

    private static double[] GaussianKernel()
    {
        double[] kernel = new double[11];
        double sum = 0;
        for (int i = 0; i < kernel.Length; i++)
        {
            double d = i - 5;
            kernel[i] = Math.Exp(-d * d / (2 * 1.5 * 1.5));
            sum += kernel[i];
        }
        for (int i = 0; i < kernel.Length; i++)
            kernel[i] /= sum;
        return kernel;
    }

    // Separable 'valid' filtering: the output shrinks by the kernel size minus one in each axis.
    private static (double[] Values, int Width, int Height) Filter(double[] source, int width, int height, double[] kernel)
    {
        int k = kernel.Length, ow = width - k + 1, oh = height - k + 1;
        double[] rows = new double[ow * height];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < ow; x++)
            {
                double acc = 0;
                for (int t = 0; t < k; t++)
                    acc += kernel[t] * source[y * width + x + t];
                rows[y * ow + x] = acc;
            }
        }
        double[] result = new double[ow * oh];
        for (int y = 0; y < oh; y++)
        {
            for (int x = 0; x < ow; x++)
            {
                double acc = 0;
                for (int t = 0; t < k; t++)
                    acc += kernel[t] * rows[(y + t) * ow + x];
                result[y * ow + x] = acc;
            }
        }
        return (result, ow, oh);
    }
}
