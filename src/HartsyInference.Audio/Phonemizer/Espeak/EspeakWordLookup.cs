namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>Looks a word up in the compiled dictionary word list, ported from <c>TransposeAlphabet</c> + <c>HashDictionary</c> + <c>LookupDict2</c> in espeak-ng dictionary.c: for Latin languages the key is first transposed (a-&gt;1, b-&gt;2, ...) and packed 6 bits per letter, exactly as the dictionary compiler stored it, then hashed into the word-list buckets. A match returns the stored phoneme code bytes plus the dictionary flag sets that drive stress placement and special-attribute handling.</summary>
internal sealed class EspeakWordLookup
{
    private const int TransposeMin = 0x60;
    private const int TransposeMax = 0x17f;
    private const int TransposeOffset = TransposeMin - 1;

    // transpose_map_latin (tr_languages.c): codepoint - 0x60 -> single-byte code (1..57), 0 = no mapping.
    private static readonly byte[] TransposeMap =
    [
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
        24,25,26,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,27,28,29,0,0,30,31,32,33,34,35,36,0,37,38,0,
        0,0,0,39,0,0,40,0,41,0,42,0,43,0,0,0,0,0,0,44,0,45,0,46,
        0,0,0,0,0,47,0,0,0,48,0,0,0,0,0,0,0,49,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,50,0,51,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,52,0,0,0,0,0,53,0,54,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,0,56,0,57,0,
    ];

    private readonly EspeakDictFile _dict;
    private readonly byte[] _data;
    private readonly int _dictCondition;

    public EspeakWordLookup(EspeakDictFile dict, int dictCondition = 0)
    {
        _dict = dict;
        _data = dict.Data;
        _dictCondition = dictCondition;
    }

    /// <summary>Looks up <paramref name="word"/> (already lowercased, no surrounding spaces) as a word spoken on its
    /// own; on a hit, returns true and fills <paramref name="result"/> with the stored phoneme codes and flag sets,
    /// otherwise false.</summary>
    public bool Lookup(string word, out EspeakLookupResult result) => Lookup(word, EspeakLookupContext.SingleWord, out result);

    /// <summary>Looks up <paramref name="word"/> where it stands in its clause: entries whose conditions (<c>$atend</c>,
    /// <c>$atstart</c>, <c>$noun</c>, <c>$verb</c>, <c>$past</c>, <c>$capital</c>, <c>$allcaps</c>, <c>$only</c>,
    /// <c>$onlys</c>, <c>$sentence</c>) do not hold are skipped, as <c>LookupDict2</c> does.</summary>
    public bool Lookup(string word, in EspeakLookupContext ctx, out EspeakLookupResult result)
    {
        result = default;
        byte[] key = TransposeAlphabet(word, out int wlen);
        int hash = HashDictionary(key);

        int p = _dict.HashStart[hash];
        int matchLen = wlen & 0x3f;

        while (_data[p] != 0)
        {
            int next = p + _data[p];
            if ((_data[p + 1] & 0x7f) != wlen || !MemEqual(key, _data, p + 2, matchLen))
            {
                p = next;
                continue;
            }

            bool noPhonemes = (_data[p + 1] & 0x80) != 0;
            int q = p + (_data[p + 1] & 0x3f) + 2;

            List<byte> phonemes = new(32);
            if (!noPhonemes)
            {
                while (_data[q] != 0)
                    phonemes.Add(_data[q++]);
                q++; // skip terminator
            }

            uint flags = 0;
            uint flags2 = 0;
            int skipWords = 0;
            bool conditionFailed = false;

            while (q < next)
            {
                byte flag = _data[q++];
                if (flag >= 100)
                {
                    if (flag >= 132)
                    {
                        if ((_dictCondition & (1 << (flag - 132))) != 0) conditionFailed = true;
                    }
                    else if ((_dictCondition & (1 << (flag - 100))) == 0) conditionFailed = true;
                }
                else if (flag > 80)
                {
                    // a multi-word entry: the following words must be the stored text (e.g. French "en tous ")
                    int nChars = next - q;
                    byte[] following = ctx.NextWords is null ? [] : System.Text.Encoding.UTF8.GetBytes(ctx.NextWords);
                    if (following.Length < nChars || !MemEqual(following, _data, q, nChars))
                        conditionFailed = true;
                    else
                        skipWords = flag - 80;
                    q = next;
                    break;
                }
                else if (flag > 64)
                {
                    flags = (flags & ~0xfu) | (uint)(flag & 0xf);
                    if ((flag & 0xc) == 0xc) flags |= FlagStressEnd;
                }
                else if (flag >= 32)
                    flags2 |= 1u << (flag - 32);
                else
                    flags |= 1u << flag;
            }

            if (conditionFailed)
            {
                p = next;
                continue;
            }

            if (!ConditionsHold(flags, flags2, ctx))
            {
                p = next;
                continue;
            }

            // FLAG_FOUND marks a spelled pronunciation; an entry of flags alone is FLAG_FOUND_ATTRIBUTES
            result = new EspeakLookupResult(phonemes, flags | (phonemes.Count > 0 ? FlagFound : FlagFoundAttributes), flags2) { SkipWords = skipWords };
            return true;
        }

        return false;
    }

    // Port of TransposeAlphabet for Latin languages (no frequent-pair compression): transpose then pack 6 bits/char.
    // Returns the lookup buffer and wlen (compressed length | 0x40 when packed, else plain strlen).
    //
    // IMPORTANT (espeak fidelity): espeak compresses the word IN PLACE and copies the compressed bytes back WITHOUT a
    // null terminator, so the buffer keeps the tail of the original (longer) word. HashDictionary then hashes
    // "compressed prefix + original tail" (up to the original null), while the entry match uses only the compressed
    // bytes. We replicate that exactly: the returned buffer is [compressed prefix][original tail][0]; the caller hashes
    // the whole thing but matches only the first `wlen & 0x3f` bytes.
    private static byte[] TransposeAlphabet(string word, out int wlen)
    {
        Span<byte> buf = stackalloc byte[EspeakRuleCodes.WordBytes];
        int bufix = 0;
        bool allAlpha = true;
        foreach (char ch in word)
        {
            int c = ch;
            if (c >= TransposeMin && c <= TransposeMax && TransposeMap[c - TransposeMin] > 0)
                buf[bufix++] = TransposeMap[c - TransposeMin];
            else { allAlpha = false; break; }
            if (bufix >= EspeakRuleCodes.WordBytes - 1) break;
        }

        if (allAlpha)
        {
            Span<byte> outBuf = stackalloc byte[EspeakRuleCodes.WordBytes];
            int acc = 0, bits = 0, o = 0;
            for (int i = 0; i < bufix; i++)
            {
                acc = (acc << 6) + (buf[i] & 0x3f);
                bits += 6;
                if (bits >= 8)
                {
                    bits -= 8;
                    outBuf[o++] = (byte)(acc >> bits);
                }
            }
            if (bits > 0)
                outBuf[o++] = (byte)(acc << (8 - bits));

            // [compressed o bytes][leftover original UTF-8 word bytes from o..len][0]
            byte[] original = System.Text.Encoding.UTF8.GetBytes(word);
            byte[] key = new byte[Math.Max(original.Length, o) + 1];
            outBuf[..o].CopyTo(key);
            for (int i = o; i < original.Length; i++)
                key[i] = original[i];
            wlen = o | 0x40;
            return key;
        }

        // Not pure alpha: the raw UTF-8 word is the key (the compiler's non-transposed path).
        byte[] utf8 = System.Text.Encoding.UTF8.GetBytes(word);
        byte[] plain = new byte[utf8.Length + 1];
        utf8.CopyTo(plain, 0);
        wlen = utf8.Length;
        return plain;
    }

    // Port of HashDictionary: 10-bit hash over the null-terminated key bytes.
    private static int HashDictionary(byte[] key)
    {
        int hash = 0, chars = 0, i = 0;
        while (key[i] != 0)
        {
            int c = key[i++] & 0xff;
            hash = hash * 8 + c;
            hash = (hash & 0x3ff) ^ (hash >> 8);
            chars++;
        }
        return (hash + chars) & 0x3ff;
    }

    private static bool MemEqual(byte[] a, byte[] b, int bOffset, int len)
    {
        for (int i = 0; i < len; i++)
            if (a[i] != b[bOffset + i]) return false;
        return true;
    }

    // LookupDict2's per-entry conditions, in its order.
    private static bool ConditionsHold(uint flags, uint flags2, in EspeakLookupContext ctx)
    {
        int endFlags = ctx.EndFlags;
        if ((endFlags & FlagSufx) == 0 && (flags2 & FlagStem) != 0) return false; // must have a suffix
        if ((endFlags & EspeakRuleCodes.SufxP) != 0 && (flags2 & (FlagOnly | FlagOnlyS)) != 0) return false;
        if ((endFlags & FlagSufx) != 0)
        {
            if ((flags2 & FlagOnly) != 0) return false;
            if ((flags2 & FlagOnlyS) != 0 && (endFlags & FlagSufxS) == 0) return false;
        }
        if ((flags2 & FlagCapital) != 0 && !ctx.FirstUpper) return false;
        if ((flags2 & FlagAllCaps) != 0 && !ctx.AllUpper) return false;
        if ((flags & FlagNeedsDot) != 0) return false; // no dot follows a word inside a clause
        if ((flags2 & FlagAtEnd) != 0 && !ctx.AtEnd) return false;
        if ((flags2 & FlagAtStart) != 0 && !ctx.AtStart) return false;
        if ((flags2 & FlagSentence) != 0 && !ctx.Sentence) return false;
        if ((flags2 & FlagVerb) != 0 && !(ctx.ExpectVerb || (ctx.ExpectVerbS && (endFlags & FlagSufxS) != 0))) return false;
        if ((flags2 & FlagPast) != 0 && !ctx.ExpectPast) return false;
        if ((flags2 & FlagNoun) != 0 && (!ctx.ExpectNoun || (endFlags & SufxV) != 0)) return false;
        return true;
    }

    // dictionary_flags2 bits and end flags (translate.h).
    private const uint FlagVerb = 0x10;
    private const uint FlagNoun = 0x20;
    private const uint FlagPast = 0x40;
    private const uint FlagCapital = 0x200;
    private const uint FlagAllCaps = 0x400;
    private const uint FlagSentence = 0x2000;
    private const uint FlagOnly = 0x4000;
    private const uint FlagOnlyS = 0x8000;
    private const uint FlagAtEnd = 0x20000;
    private const uint FlagAtStart = 0x40000;
    private const uint FlagNeedsDot = 0x02000000;
    private const int FlagSufx = 0x04;
    private const int FlagSufxS = 0x08;
    private const int SufxV = 0x0800;

    private const uint FlagFound = 0x80000000;
    private const uint FlagFoundAttributes = 0x40000000;
    private const uint FlagStressEnd = 0x200;
    private const uint FlagStem = 0x10000; // flags2 bit 16
}
