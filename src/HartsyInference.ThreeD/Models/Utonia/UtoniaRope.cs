namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>
/// Utonia's 3D rotary embedding. Each head's channels split into x, y and z chunks; each chunk rotates pairs
/// (i, i + chunk/2) by its own coordinate's frequencies. Positions are the raw point coordinates.
/// </summary>
internal static class UtoniaRope
{
    /// <summary>Rotates the query and key blocks of a row-major <c>[rows, 3·channels]</c> qkv buffer in place.</summary>
    /// <param name="qkv">Token rows laid out as q | k | v, each <c>channels</c> wide.</param>
    /// <param name="slotCoord">Position of each row, three floats per row.</param>
    public static unsafe void Apply(float* qkv, ReadOnlySpan<float> slotCoord, int channels, int heads, float ropeBase)
    {
        int headDim = channels / heads;
        int chunk = headDim / 3;
        int half = chunk / 2;
        float[] invFreq = new float[half];
        for (int j = 0; j < half; j++)
        {
            invFreq[j] = 1f / MathF.Pow(ropeBase, 2f * j / chunk);
        }

        int rows = slotCoord.Length / 3;
        long rowStride = 3L * channels;
        for (int r = 0; r < rows; r++)
        {
            for (int part = 0; part < 2; part++)
            {
                for (int h = 0; h < heads; h++)
                {
                    for (int a = 0; a < 3; a++)
                    {
                        float pos = slotCoord[r * 3 + a];
                        float* block = qkv + r * rowStride + part * channels + h * headDim + a * chunk;
                        for (int i = 0; i < half; i++)
                        {
                            float theta = pos * invFreq[i];
                            float c = MathF.Cos(theta), s = MathF.Sin(theta);
                            float x1 = block[i], x2 = block[i + half];
                            block[i] = x1 * c - x2 * s;
                            block[i + half] = x2 * c + x1 * s;
                        }
                    }
                }
            }
        }
    }
}
