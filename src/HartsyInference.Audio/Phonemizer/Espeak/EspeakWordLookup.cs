namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>Looks a word up in the compiled dictionary word list, ported from <c>TransposeAlphabet</c> +
/// <c>HashDictionary</c> + <c>LookupDict2</c> in espeak-ng dictionary.c. For Latin languages the key is first
/// transposed (a-&gt;1, b-&gt;2, ...) and packed 6 bits per letter, exactly as the dictionary compiler stored it, then
/// hashed into the word-list buckets. A match returns the stored phoneme code bytes plus the dictionary flag sets that
/// drive stress placement and special-attribute handling.</summary>
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
    private int _dictCondition;

    public EspeakWordLookup(EspeakDictFile dict, int dictCondition = 0)
    {
        _dict = dict;
        _data = dict.Data;
        _dictCondition = dictCondition;
    }

    /// <summary>Looks up <paramref name="word"/> (already lowercased, no surrounding spaces). On a hit, returns true
    /// and fills <paramref name="result"/> with the stored phoneme codes and flag sets; otherwise false.</summary>
    public bool Lookup(string word, out EspeakLookupResult result)
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
                    // multi-word contraction match; not resolved in single-word lookup, treat as non-matching here.
                    conditionFailed = true;
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

            // No suffix removed in single-word lookup: a word that requires a suffix cannot match.
            if ((flags2 & FlagStem) != 0)
            {
                p = next;
                continue;
            }

            result = new EspeakLookupResult(phonemes, flags | FlagFound, flags2);
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

            // [compressed o bytes][leftover original word bytes from o..len][0]
            byte[] key = new byte[word.Length + 1];
            outBuf[..o].CopyTo(key);
            for (int i = o; i < word.Length; i++)
                key[i] = (byte)word[i];
            wlen = o | 0x40;
            return key;
        }

        // Not pure alpha: use the raw word bytes as the key (matches the compiler's non-transposed path).
        byte[] plain = new byte[word.Length + 1];
        for (int i = 0; i < word.Length; i++)
            plain[i] = (byte)word[i];
        wlen = word.Length;
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

    private const uint FlagFound = 0x80000000;
    private const uint FlagStressEnd = 0x200;
    private const uint FlagStem = 0x10000; // flags2 bit 16
}
