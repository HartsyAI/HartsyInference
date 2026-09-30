using HartsyInference.ModelAssets.Mxfp4;

namespace HartsyInference.ModelAssets.BlockScale;

/// <summary>Decodes F8E8M0 scale bytes: value <c>2^(byte - 127)</c>, with 0xFF as NaN.</summary>
internal static class E8M0Scale
{
    /// <summary>All 256 decoded scales; byte 0 is the subnormal 2^-127, not zero.</summary>
    internal static readonly float[] Table = Build();

    private static float[] Build()
    {
        float[] table = new float[256];
        for (int e = 0; e < 255; e++)
            table[e] = (float)Math.ScaleB(1.0, e - Mxfp4Codec.E8M0Bias);
        table[255] = float.NaN;
        return table;
    }
}
