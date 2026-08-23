namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>One phoneme-table entry, a 16-byte record in espeak-ng's compiled <c>phontab</c> file (mirrors <c>PHONEME_TAB</c> in <c>phoneme.h</c>); holds the data the text path needs: the up-to-4-char mnemonic, the phoneme type/flags used for stress placement, and the internal code byte that dictionary phoneme strings reference.</summary>
internal readonly struct EspeakPhoneme
{
    /// <summary>Up to four characters packed little-endian (first char in the least-significant byte).</summary>
    public uint Mnemonic { get; init; }

    /// <summary>Phoneme property bits (bits 16-19 are place of articulation); see the <c>Ph*</c> flag masks.</summary>
    public uint PhFlags { get; init; }

    /// <summary>Index into <c>phondata</c> (unused by the text-only phoneme path; kept for completeness).</summary>
    public ushort Program { get; init; }

    /// <summary>The phoneme number; dictionary phoneme strings are sequences of these code bytes.</summary>
    public byte Code { get; init; }

    /// <summary>Phoneme class: <see cref="TypePause"/>, <see cref="TypeStress"/>, <see cref="TypeVowel"/>, etc.</summary>
    public byte Type { get; init; }

    public byte StartType { get; init; }
    public byte EndType { get; init; }

    /// <summary>For vowels, length in mS/2; for stress phonemes, the stress/tone level.</summary>
    public byte StdLength { get; init; }

    public byte LengthMod { get; init; }

    // Phoneme types (phoneme.h ph* constants).
    public const byte TypePause = 0;
    public const byte TypeStress = 1;
    public const byte TypeVowel = 2;
    public const byte TypeLiquid = 3;
    public const byte TypeStop = 4;
    public const byte TypeVStop = 5;
    public const byte TypeFricative = 6;
    public const byte TypeVFricative = 7;
    public const byte TypeNasal = 8;
    public const byte TypeInvalid = 15;

    // Phoneme flag bits (phoneme.h phFLAGBIT_*).
    public const uint FlagUnstressed = 1U << 1;
    public const uint FlagVoiced = 1U << 4;
    public const uint FlagNonSyllabic = 1U << 20;

    /// <summary>Reads a 16-byte phoneme record from <paramref name="data"/> at <paramref name="offset"/>.</summary>
    public static EspeakPhoneme Read(byte[] data, int offset) => new()
    {
        Mnemonic = (uint)(data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16) | (data[offset + 3] << 24)),
        PhFlags = (uint)(data[offset + 4] | (data[offset + 5] << 8) | (data[offset + 6] << 16) | (data[offset + 7] << 24)),
        Program = (ushort)(data[offset + 8] | (data[offset + 9] << 8)),
        Code = data[offset + 10],
        Type = data[offset + 11],
        StartType = data[offset + 12],
        EndType = data[offset + 13],
        StdLength = data[offset + 14],
        LengthMod = data[offset + 15],
    };

    /// <summary>Decodes a packed 4-char mnemonic to text (e.g. the <c>aI</c> diphthong or <c>'</c> stress mark).</summary>
    public static string DecodeMnemonic(uint mnemonic)
    {
        Span<char> chars = stackalloc char[4];
        int n = 0;
        for (int i = 0; i < 4; i++)
        {
            byte b = (byte)((mnemonic >> (i * 8)) & 0xff);
            if (b == 0) break;
            chars[n++] = (char)b;
        }
        return new string(chars[..n]);
    }

    /// <summary>The mnemonic as text.</summary>
    public string MnemonicText => DecodeMnemonic(Mnemonic);
}
