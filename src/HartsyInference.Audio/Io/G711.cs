namespace HartsyInference.Audio.Io;

/// <summary>G.711 μ-law and A-law companding between 16-bit PCM and the 8-bit codes a telephone leg carries.</summary>
/// <remarks>Both directions are table lookups built once from the ITU-T reference arithmetic (μ-law: bias 0x84,
/// clip 32635, complemented output; A-law: even-bit inversion with 0x55), so encoding a frame is one indexed read
/// per sample with no branches. One deliberate departure from the NAudio-derived encoders: <c>short.MinValue</c>
/// is clipped like every other over-range value and lands on the most negative code, where those encoders let
/// its negation wrap and emit a near-zero code. Every other input agrees with them code for code.</remarks>
public static class G711
{
    /// <summary>The μ-law code for digital silence (linear 0).</summary>
    public const byte MuLawSilence = 0xFF;

    /// <summary>The A-law code for digital silence (linear +8, the smallest positive step).</summary>
    public const byte ALawSilence = 0xD5;

    private const int Bias = 0x84;
    private const int Clip = 32635;

    private static readonly byte[] _muLawEncode = BuildEncodeTable(G711Law.MuLaw);
    private static readonly byte[] _aLawEncode = BuildEncodeTable(G711Law.ALaw);
    private static readonly short[] _muLawDecode = BuildDecodeTable(G711Law.MuLaw);
    private static readonly short[] _aLawDecode = BuildDecodeTable(G711Law.ALaw);

    /// <summary>The code a silent frame is filled with under <paramref name="law"/>.</summary>
    public static byte Silence(G711Law law) => law == G711Law.MuLaw ? MuLawSilence : ALawSilence;

    /// <summary>Encodes <paramref name="pcm"/> into <paramref name="encoded"/>, one byte per sample.</summary>
    public static void Encode(ReadOnlySpan<short> pcm, Span<byte> encoded, G711Law law)
    {
        if (encoded.Length < pcm.Length)
        {
            throw new ArgumentException($"encoded needs {pcm.Length} bytes, got {encoded.Length}.", nameof(encoded));
        }
        byte[] table = law == G711Law.MuLaw ? _muLawEncode : _aLawEncode;
        for (int i = 0; i < pcm.Length; i++)
        {
            encoded[i] = table[(ushort)pcm[i]];
        }
    }

    /// <summary>Decodes <paramref name="encoded"/> into <paramref name="pcm"/>, one sample per byte.</summary>
    public static void Decode(ReadOnlySpan<byte> encoded, Span<short> pcm, G711Law law)
    {
        if (pcm.Length < encoded.Length)
        {
            throw new ArgumentException($"pcm needs {encoded.Length} samples, got {pcm.Length}.", nameof(pcm));
        }
        short[] table = law == G711Law.MuLaw ? _muLawDecode : _aLawDecode;
        for (int i = 0; i < encoded.Length; i++)
        {
            pcm[i] = table[encoded[i]];
        }
    }

    /// <summary>Encodes one sample.</summary>
    public static byte EncodeSample(short sample, G711Law law) =>
        (law == G711Law.MuLaw ? _muLawEncode : _aLawEncode)[(ushort)sample];

    /// <summary>Decodes one code.</summary>
    public static short DecodeSample(byte code, G711Law law) =>
        (law == G711Law.MuLaw ? _muLawDecode : _aLawDecode)[code];

    private static byte[] BuildEncodeTable(G711Law law)
    {
        byte[] table = new byte[65536];
        for (int i = 0; i < table.Length; i++)
        {
            short sample = (short)i;
            table[(ushort)sample] = law == G711Law.MuLaw ? LinearToMuLaw(sample) : LinearToALaw(sample);
        }
        return table;
    }

    private static short[] BuildDecodeTable(G711Law law)
    {
        short[] table = new short[256];
        for (int code = 0; code < 256; code++)
        {
            table[code] = law == G711Law.MuLaw ? MuLawToLinear((byte)code) : ALawToLinear((byte)code);
        }
        return table;
    }

    /// <summary>Reference μ-law encoder in int arithmetic, so negating <c>short.MinValue</c> clips instead of wrapping.</summary>
    private static byte LinearToMuLaw(short sample)
    {
        int pcm = sample;
        int sign = (pcm >> 8) & 0x80;
        if (sign != 0)
        {
            pcm = -pcm;
        }
        if (pcm > Clip)
        {
            pcm = Clip;
        }
        pcm += Bias;
        int exponent = MuLawExponent((pcm >> 7) & 0xFF);
        int mantissa = (pcm >> (exponent + 3)) & 0x0F;
        return (byte)~(sign | (exponent << 4) | mantissa);
    }

    private static short MuLawToLinear(byte code)
    {
        int u = ~code & 0xFF;
        int sign = u & 0x80;
        int exponent = (u >> 4) & 0x07;
        int mantissa = u & 0x0F;
        int magnitude = ((mantissa << 3) + Bias) << exponent;
        return (short)(sign != 0 ? Bias - magnitude : magnitude - Bias);
    }

    /// <summary>Reference A-law encoder in int arithmetic; see <see cref="LinearToMuLaw"/> for the clip note.</summary>
    private static byte LinearToALaw(short sample)
    {
        int pcm = sample;
        int sign = (~pcm >> 8) & 0x80;
        if (sign == 0)
        {
            pcm = -pcm;
        }
        if (pcm > Clip)
        {
            pcm = Clip;
        }
        int code;
        if (pcm >= 256)
        {
            int exponent = ALawExponent((pcm >> 8) & 0x7F);
            int mantissa = (pcm >> (exponent + 3)) & 0x0F;
            code = (exponent << 4) | mantissa;
        }
        else
        {
            code = pcm >> 4;
        }
        return (byte)(code ^ (sign ^ 0x55));
    }

    private static short ALawToLinear(byte code)
    {
        int a = code ^ 0x55;
        int sign = a & 0x80;
        int exponent = (a >> 4) & 0x07;
        int magnitude = (a & 0x0F) << 4;
        if (exponent == 0)
        {
            magnitude += 8;
        }
        else
        {
            magnitude = (magnitude + 0x108) << (exponent - 1);
        }
        return (short)(sign != 0 ? magnitude : -magnitude);
    }

    /// <summary>The μ-law segment table: 0,0,1,1,2,2,2,2,3×8,4×16,5×32,6×64,7×128 over the biased sample's top byte.</summary>
    private static int MuLawExponent(int topByte) => topByte == 0 ? 0 : BitOperations.Log2((uint)topByte);

    /// <summary>The A-law segment table: 1,1,2,2,3,3,3,3,4×8,5×16,6×32,7×64 over the sample's top seven bits.</summary>
    private static int ALawExponent(int topBits) => topBits == 0 ? 1 : BitOperations.Log2((uint)topBits) + 1;
}
