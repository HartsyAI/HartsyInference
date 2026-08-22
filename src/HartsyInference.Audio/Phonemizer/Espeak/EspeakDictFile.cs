using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;

namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>Parses a compiled espeak-ng <c>&lt;lang&gt;_dict</c> file into the in-memory indices espeak builds at load time (ported from <c>LoadDictionary</c> + <c>InitGroups</c> in espeak-ng v1.50 <c>dictionary.c</c>); on-disk layout is bytes 0-3 = hash-table size (must equal 1024), bytes 4-7 = offset to the rules section, then the word-list hash buckets, then the letter-to-sound rule groups, all little-endian.</summary>
internal sealed class EspeakDictFile
{
    /// <summary>The whole dictionary file. Rule and word-entry offsets index directly into this array.</summary>
    public byte[] Data { get; }

    /// <summary>Absolute offset into <see cref="Data"/> where the rules section (<c>data_dictrules</c>) begins.</summary>
    public int RulesOffset { get; }

    /// <summary>Per-bucket start offset into <see cref="Data"/> for the word-list hash table (length 1024).</summary>
    public int[] HashStart { get; } = new int[EspeakRuleCodes.HashDictSize];

    /// <summary>Rule-chain offset indexed by single initial letter (byte), or -1 when unset.</summary>
    public int[] Groups1 { get; } = NewFilled(256, -1);

    /// <summary>Rule-chain offset indexed by offset-from-letter-base (groups whose name starts with byte 1), or -1.</summary>
    public int[] Groups3 { get; } = NewFilled(128, -1);

    /// <summary>Rule-chain offsets for two-letter groups, in file order.</summary>
    public int[] Groups2 { get; }

    /// <summary>The two-letter pair (c + (c2 &lt;&lt; 8)) for each <see cref="Groups2"/> entry.</summary>
    public int[] Groups2Name { get; }

    /// <summary>Number of two-letter groups whose first letter is the index byte.</summary>
    public byte[] Groups2Count { get; } = new byte[256];

    /// <summary>Index into <see cref="Groups2"/> of the first group for the index byte, or 255 when unset.</summary>
    public byte[] Groups2Start { get; } = NewFilledBytes(256, 255);

    /// <summary>Number of populated <see cref="Groups2"/> entries.</summary>
    public int NumGroups2 { get; }

    /// <summary>Letter-group (<c>.Lnn</c>) rule offsets, or -1 when unset (length 95).</summary>
    public int[] LetterGroups { get; } = NewFilled(EspeakRuleCodes.LetterGroups, -1);

    /// <summary>Offset of the <c>.replace</c> character-replacement table, or -1 when absent.</summary>
    public int ReplaceChars { get; } = -1;

    private EspeakDictFile(byte[] data)
    {
        Data = data;

        if (data.Length <= EspeakRuleCodes.HashDictSize + sizeof(int) * 2)
            throw new HartsyInferenceException($"espeak dictionary is too small ({data.Length} bytes) to be valid.");

        int hashSize = ReadInt32(data, 0);
        RulesOffset = ReadInt32(data, 4);
        if (hashSize != EspeakRuleCodes.HashDictSize)
            throw new HartsyInferenceException($"espeak dictionary header hash size {hashSize} != {EspeakRuleCodes.HashDictSize}.");
        if (RulesOffset <= 0 || RulesOffset > 0x8000000 || RulesOffset >= data.Length)
            throw new HartsyInferenceException($"espeak dictionary rules offset {RulesOffset} is out of range.");

        BuildHashTable();

        List<int> groups2 = new();
        List<int> groups2Name = new();
        ReplaceChars = InitGroups(groups2, groups2Name);
        Groups2 = groups2.ToArray();
        Groups2Name = groups2Name.ToArray();
        NumGroups2 = groups2.Count;
    }

    /// <summary>Reads and parses a compiled dictionary file from <paramref name="path"/>.</summary>
    public static EspeakDictFile Load(string path)
    {
        try
        {
            return new EspeakDictFile(File.ReadAllBytes(path));
        }
        catch (Exception ex) when (ex is not HartsyInferenceException)
        {
            Logs.Error($"Failed to load espeak dictionary '{path}': {ex.Message}");
            throw new HartsyInferenceException($"Failed to load espeak dictionary '{path}'.", ex);
        }
    }

    /// <summary>Parses already-loaded dictionary bytes (used when the bytes come from a cache or embedded asset).</summary>
    public static EspeakDictFile Parse(byte[] data) => new(data);

    // Port of the hash-table walk in LoadDictionary: each bucket is a run of length-prefixed entries
    // terminated by a zero length byte.
    private void BuildHashTable()
    {
        int p = 8;
        for (int hash = 0; hash < EspeakRuleCodes.HashDictSize; hash++)
        {
            HashStart[hash] = p;
            int length;
            while ((length = Data[p]) != 0)
                p += length;
            p++; // skip the terminating zero
        }
    }

    // Faithful port of InitGroups (dictionary.c). Returns the replace-table offset, or -1 when none.
    private int InitGroups(List<int> groups2, List<int> groups2Name)
    {
        int replaceChars = -1;
        int p = RulesOffset;

        if (Data[p] == EspeakRuleCodes.RuleGroupEnd)
            return replaceChars;

        while (Data[p] != 0)
        {
            if (Data[p] != EspeakRuleCodes.RuleGroupStart)
                throw new HartsyInferenceException($"Bad espeak rules data at offset 0x{p - RulesOffset:x} (byte {Data[p]}).");
            p++;

            if (Data[p] == EspeakRuleCodes.RuleReplacements)
            {
                p = (p + 4) & ~3; // advance to next 4-byte boundary (aligned within the buffer)
                replaceChars = p;
                while (!IsFourNull(p))
                    p++;
                while (Data[p] != EspeakRuleCodes.RuleGroupEnd)
                    p++;
                p++;
                continue;
            }

            if (Data[p] == EspeakRuleCodes.RuleLetterGp2)
            {
                int ix = Data[p + 1] - 'A';
                if (ix < 0)
                    ix += 256;
                p += 2;
                if (ix >= 0 && ix < EspeakRuleCodes.LetterGroups)
                    LetterGroups[ix] = p;
            }
            else
            {
                int len = StrLen(p);
                byte c = Data[p];
                byte c2 = Data[p + 1];
                p += len + 1;
                if (len == 1)
                    Groups1[c] = p;
                else if (len == 0)
                    Groups1[0] = p;
                else if (c == 1)
                    Groups3[c2 - 1] = p;
                else
                {
                    if (Groups2Start[c] == 255)
                        Groups2Start[c] = (byte)groups2.Count;
                    Groups2Count[c]++;
                    groups2.Add(p);
                    groups2Name.Add(c + (c2 << 8));
                }
            }

            while (Data[p] != EspeakRuleCodes.RuleGroupEnd)
                p += StrLen(p) + 1;
            p++;
        }

        return replaceChars;
    }

    private bool IsFourNull(int p) => Data[p] == 0 && Data[p + 1] == 0 && Data[p + 2] == 0 && Data[p + 3] == 0;

    private int StrLen(int p)
    {
        int start = p;
        while (Data[p] != 0)
            p++;
        return p - start;
    }

    private static int ReadInt32(byte[] data, int offset)
        => data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16) | (data[offset + 3] << 24);

    private static int[] NewFilled(int count, int value)
    {
        int[] arr = new int[count];
        Array.Fill(arr, value);
        return arr;
    }

    private static byte[] NewFilledBytes(int count, byte value)
    {
        byte[] arr = new byte[count];
        Array.Fill(arr, value);
        return arr;
    }
}
