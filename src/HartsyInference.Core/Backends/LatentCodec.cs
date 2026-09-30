using System.Runtime.CompilerServices;

namespace HartsyInference.Core.Backends;

/// <summary>Scalar codecs for the latent formats, reproducing the reference <c>act_quant</c> / <c>fp4_act_quant</c> exactly.</summary>
public static class LatentCodec
{
    private const float Fp8Max = 448f;
    private const float Fp4Max = 6f;
    private const float Fp8MaxInv = 1f / 448f;
    private const float Fp4MaxInv = 1f / 6f;
    private static readonly float MinNormal = BitConverter.Int32BitsToSingle(0x00800000);
    private static readonly float[] E2M1 = { 0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f };

    /// <summary>Decodes an e4m3fn byte (bias 7, no infinities; 0x7F and 0xFF are NaN).</summary>
    public static float DecodeE4M3(byte code)
    {
        int exp = (code >> 3) & 0xF, man = code & 7;
        float sign = (code & 0x80) != 0 ? -1f : 1f;
        if (exp == 0xF && man == 7) return float.NaN;
        if (exp == 0) return sign * man * 0.001953125f;
        return sign * BitConverter.Int32BitsToSingle(((exp + 120) << 23) | (man << 20));
    }

    /// <summary>Round-to-nearest-even e4m3fn encode with the sign kept for zeros; past 464 it is NaN (0x7F), as the cast is.</summary>
    public static byte EncodeE4M3(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        uint sign = (bits >> 24) & 0x80u;
        float mag = MathF.Abs(value);
        uint code;
        if (mag < 0.015625f) code = (uint)MathF.Round(mag * 512f, MidpointRounding.ToEven);
        else
        {
            uint m = bits & 0x7FFFFFFFu;
            uint rounded = (m + 0x7FFFFu + ((m >> 20) & 1u)) >> 20;
            code = rounded - 960u > 0x7Eu ? 0x7Fu : rounded - 960u;
        }
        return (byte)(sign | code);
    }

    /// <summary>Decodes an e8m0 scale byte to <c>2^(byte-127)</c>; byte 0 is the denormal 2^-127.</summary>
    public static float DecodeUe8m0(byte code) =>
        BitConverter.Int32BitsToSingle(code == 0 ? 0x00400000 : code << 23);

    /// <summary>Decodes an e2m1 code 0..15 (sign in bit 3).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DecodeE2M1(int code) => (code & 8) != 0 ? -E2M1[code & 7] : E2M1[code & 7];

    /// <summary>E2M1 code of a value already clamped to +-6: round-to-nearest with ties to the even code; -0 keeps its sign bit.</summary>
    public static int EncodeE2M1(float value)
    {
        float mag = MathF.Abs(value);
        int code = 0;
        for (int i = 0; i < 7; i++)
        {
            float mid = (E2M1[i] + E2M1[i + 1]) * 0.5f;
            if (mag > mid) code++;
            else if (mag == mid && (i & 1) == 1) code++;
        }
        // NaN (an e4m3 scale that overflowed) has no sign and rounds to +0, as in the torch port.
        bool negative = value < 0f || (value == 0f && BitConverter.SingleToUInt32Bits(value) >> 31 != 0);
        return code | (negative ? 8 : 0);
    }

    /// <summary>Power-of-two ceiling of <c>amax * maxInv</c> read from the fp32 exponent; returns the e8m0 byte.</summary>
    public static byte Ue8m0Byte(float amax, float maxInv)
    {
        int bits = BitConverter.SingleToInt32Bits(amax * maxInv);
        int exp = (bits >> 23) & 0xFF;
        int log2Ceil = exp - 127 + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
        return (byte)(log2Ceil + 127);
    }

    /// <summary>Quantizes one scale group: writes a code per element (fp8 byte or fp4 nibble) and the scale byte,
    /// and returns the decoded scale.</summary>
    public static float QuantizeGroup(LatentEncoding encoding, ReadOnlySpan<float> group, Span<byte> codes,
        out byte scaleByte)
    {
        float amax = 0f;
        for (int i = 0; i < group.Length; i++) amax = MathF.Max(amax, MathF.Abs(group[i]));
        float scale;
        switch (encoding)
        {
            case LatentEncoding.Fp8E4M3Ue8m0x32:
                scaleByte = Ue8m0Byte(MathF.Max(amax, 1e-4f), Fp8MaxInv);
                scale = DecodeUe8m0(scaleByte);
                for (int i = 0; i < group.Length; i++)
                    codes[i] = EncodeE4M3(Math.Clamp(group[i] / scale, -Fp8Max, Fp8Max));
                return scale;
            case LatentEncoding.Fp4E2M1E4M3x16:
                scaleByte = EncodeE4M3(MathF.Max(amax, 6f * 0.001953125f) / Fp4Max);
                scale = DecodeE4M3(scaleByte);
                break;
            case LatentEncoding.Fp4E2M1E8M0x32:
                scaleByte = Ue8m0Byte(MathF.Max(amax, 6f * MinNormal), Fp4MaxInv);
                scale = DecodeUe8m0(scaleByte);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(encoding), encoding, "Only quantized encodings have groups.");
        }
        for (int i = 0; i < group.Length; i++) codes[i] = (byte)EncodeE2M1(Math.Clamp(group[i] / scale, -Fp4Max, Fp4Max));
        return scale;
    }

    /// <summary>Decodes a code produced by <see cref="QuantizeGroup"/> (before applying the scale).</summary>
    public static float DecodeCode(LatentEncoding encoding, byte code) =>
        encoding == LatentEncoding.Fp8E4M3Ue8m0x32 ? DecodeE4M3(code) : DecodeE2M1(code);

    /// <summary>Decodes a scale byte for <paramref name="encoding"/>.</summary>
    public static float DecodeScale(LatentEncoding encoding, byte scaleByte) =>
        encoding == LatentEncoding.Fp4E2M1E4M3x16 ? DecodeE4M3(scaleByte) : DecodeUe8m0(scaleByte);

    /// <summary>Dequantizes element <paramref name="col"/> of a row given that row's code and scale bytes.</summary>
    public static float DecodeElement(LatentEncoding encoding, ReadOnlySpan<byte> rowCodes, ReadOnlySpan<byte> rowScales,
        int col)
    {
        int group = LatentEncodings.GroupSize(encoding);
        float scale = DecodeScale(encoding, rowScales[col / group]);
        if (encoding == LatentEncoding.Fp8E4M3Ue8m0x32) return DecodeE4M3(rowCodes[col]) * scale;
        int packed = rowCodes[col >> 1];
        return DecodeE2M1((col & 1) == 0 ? packed & 0xF : packed >> 4) * scale;
    }
}
