namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Kandinsky 5 coordinate rules and host angle math. The 3D scales divide positions in double before the angle is
/// formed, so the division stays here and is never routed through the float positions of <c>RopeTableCache</c>.</summary>
internal static class Kandinsky5PosEmbed
{
    /// <summary>Per-token pair angles for 1D positions over the whole head: <c>pos × (1 / theta^(2k/headDim))</c>.</summary>
    public static void Fill1D(float theta, int headDim, ReadOnlySpan<int> positions, Span<float> cos, Span<float> sin)
    {
        int seqLen = positions.Length;
        int halfDim = headDim / 2;
        Span<double> freqs = stackalloc double[halfDim];
        for (int k = 0; k < halfDim; k++)
            freqs[k] = 1.0 / Math.Pow(theta, (double)(2 * k) / headDim);

        for (int s = 0; s < seqLen; s++)
        {
            double pos = positions[s];
            int rowOffset = s * halfDim;
            for (int k = 0; k < halfDim; k++)
            {
                double angle = pos * freqs[k];
                cos[rowOffset + k] = (float)Math.Cos(angle);
                sin[rowOffset + k] = (float)Math.Sin(angle);
            }
        }
    }

    /// <summary>Per-token pair angles for visual tokens in <c>(t, h, w)</c> order. Each axis divides its position by its scale in
    /// double, then multiplies by its own frequencies; the axis pairs are concatenated as <c>[t, h, w]</c>.</summary>
    public static void Fill3D(
        float theta,
        int[] axesDims,
        int headDim,
        int duration,
        int height,
        int width,
        int tStart,
        float scaleT,
        float scaleH,
        float scaleW,
        Span<float> cos,
        Span<float> sin)
    {
        int halfDim = headDim / 2;
        int dt = axesDims[0], dh = axesDims[1], dw = axesDims[2];
        int pt = dt / 2, ph = dh / 2, pw = dw / 2;
        Span<double> ft = stackalloc double[pt];
        Span<double> fh = stackalloc double[ph];
        Span<double> fw = stackalloc double[pw];
        for (int k = 0; k < pt; k++) ft[k] = 1.0 / Math.Pow(theta, (double)(2 * k) / dt);
        for (int k = 0; k < ph; k++) fh[k] = 1.0 / Math.Pow(theta, (double)(2 * k) / dh);
        for (int k = 0; k < pw; k++) fw[k] = 1.0 / Math.Pow(theta, (double)(2 * k) / dw);

        for (int t = 0; t < duration; t++)
        {
            double posT = (t + tStart) / (double)scaleT;
            for (int h = 0; h < height; h++)
            {
                double posH = h / (double)scaleH;
                for (int w = 0; w < width; w++)
                {
                    double posW = w / (double)scaleW;
                    int s = (t * height + h) * width + w;
                    int rowOffset = s * halfDim;

                    int k = 0;
                    for (int i = 0; i < pt; i++)
                    {
                        double angle = posT * ft[i];
                        cos[rowOffset + k] = (float)Math.Cos(angle);
                        sin[rowOffset + k] = (float)Math.Sin(angle);
                        k++;
                    }
                    for (int i = 0; i < ph; i++)
                    {
                        double angle = posH * fh[i];
                        cos[rowOffset + k] = (float)Math.Cos(angle);
                        sin[rowOffset + k] = (float)Math.Sin(angle);
                        k++;
                    }
                    for (int i = 0; i < pw; i++)
                    {
                        double angle = posW * fw[i];
                        cos[rowOffset + k] = (float)Math.Cos(angle);
                        sin[rowOffset + k] = (float)Math.Sin(angle);
                        k++;
                    }
                }
            }
        }
    }

    /// <summary>Cache signature for 1D positions: one value per token, exact for token indices.</summary>
    public static float[] Signature1D(ReadOnlySpan<int> positions)
    {
        float[] signature = new float[positions.Length];
        for (int s = 0; s < positions.Length; s++)
            signature[s] = positions[s];
        return signature;
    }

    /// <summary>Cache signature for the 3D grid: the raw <c>(t + tStart, h, w)</c> coordinates per token, before scaling.</summary>
    public static float[] Signature3D(int duration, int height, int width, int tStart)
    {
        float[] signature = new float[duration * height * width * 3];
        int at = 0;
        for (int t = 0; t < duration; t++)
            for (int h = 0; h < height; h++)
                for (int w = 0; w < width; w++)
                {
                    signature[at++] = t + tStart;
                    signature[at++] = h;
                    signature[at++] = w;
                }
        return signature;
    }
}
