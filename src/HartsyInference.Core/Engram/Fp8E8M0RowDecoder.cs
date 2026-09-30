using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Engram;

/// <summary>Decodes an fp8 e4m3 row with one e8m0 scale per 32 columns, as the official checkpoint and the GGUF conversion store it.</summary>
internal static class Fp8E8M0RowDecoder
{
    public const int Columns = 256;
    public const int BlockSize = 32;
    public const int Blocks = Columns / BlockSize;

    private static readonly float[] E4M3 = BuildE4M3();
    private static readonly float[] E8M0 = BuildE8M0();

    /// <summary>Value of a scale byte, 2^(b-127); 255 is NaN. Byte 0 is the float32 subnormal 2^-127, which its bit pattern cannot express as a normal.</summary>
    public static float ScaleOf(byte code) => E8M0[code];

    public static void Decode(ReadOnlySpan<byte> fp8, ReadOnlySpan<byte> scales, Span<ushort> dest)
    {
        if (fp8.Length < Columns || scales.Length < Blocks || dest.Length < Columns)
            throw new ArgumentException("An fp8 e8m0 row needs 256 codes, 8 scales and room for 256 outputs.");
        float[] e4m3 = E4M3;
        for (int block = 0; block < Blocks; block++)
        {
            float scale = E8M0[scales[block]];
            int start = block * BlockSize;
            for (int i = start; i < start + BlockSize; i++)
                dest[i] = Bf16Rounding.FromSingle(e4m3[fp8[i]] * scale);
        }
    }

    private static float[] BuildE4M3()
    {
        float[] table = new float[256];
        for (int code = 0; code < 256; code++)
            table[code] = LatentCodec.DecodeE4M3((byte)code);
        return table;
    }

    private static float[] BuildE8M0()
    {
        float[] table = new float[256];
        table[0] = BitConverter.Int32BitsToSingle(0x00400000);
        for (int code = 1; code < 255; code++)
            table[code] = BitConverter.Int32BitsToSingle(code << 23);
        table[255] = float.NaN;
        return table;
    }
}
