using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;

namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>Parses espeak-ng's compiled <c>phontab</c> file and resolves the active phoneme table for a named language (ported from <c>LoadPhData</c> + <c>SetUpPhonemeTable</c> in <c>synthdata.c</c>); a language table such as <c>en</c> overlays a chain of base tables via its <c>includes</c> field, and this class follows that chain to build the <c>phoneme_tab</c> indexed by code byte.</summary>
internal sealed class EspeakPhonemeTable
{
    private const int PhonemeRecordSize = 16;
    private const int TableNameLength = 32;

    private readonly EspeakPhoneme[] _byCode = new EspeakPhoneme[256];
    private readonly bool[] _present = new bool[256];

    /// <summary>Highest populated phoneme code in the active table (espeak's <c>n_phoneme_tab - 1</c>).</summary>
    public int MaxCode { get; private set; }

    private EspeakPhonemeTable(byte[] phontab, string tableName)
    {
        List<TableInfo> tables = ParseTables(phontab);
        int index = tables.FindIndex(t => string.Equals(t.Name, tableName, StringComparison.Ordinal));
        if (index < 0)
            throw new HartsyInferenceException($"Phoneme table '{tableName}' not found in phontab ({tables.Count} tables).");
        BuildActive(phontab, tables, index);
    }

    /// <summary>Loads <c>phontab</c> from <paramref name="path"/> and resolves the active table for <paramref name="tableName"/> (e.g. <c>"en"</c>).</summary>
    public static EspeakPhonemeTable Load(string path, string tableName)
    {
        try
        {
            return new EspeakPhonemeTable(File.ReadAllBytes(path), tableName);
        }
        catch (Exception ex) when (ex is not HartsyInferenceException)
        {
            Logs.Error($"Failed to load espeak phoneme table '{path}' ({tableName}): {ex.Message}");
            throw new HartsyInferenceException($"Failed to load espeak phoneme table '{path}'.", ex);
        }
    }

    /// <summary>Parses already-loaded phontab bytes for <paramref name="tableName"/>.</summary>
    public static EspeakPhonemeTable Parse(byte[] phontab, string tableName) => new(phontab, tableName);

    /// <summary>Returns the phoneme record for <paramref name="code"/>, or false if that code is unused.</summary>
    public bool TryGet(int code, out EspeakPhoneme phoneme)
    {
        if ((uint)code < 256 && _present[code])
        {
            phoneme = _byCode[code];
            return true;
        }
        phoneme = default;
        return false;
    }

    /// <summary>Decodes a sequence of phoneme code bytes back to their concatenated mnemonic text (espeak <c>DecodePhonemes</c>), skipping codes that are not in the table.</summary>
    public string Decode(IReadOnlyList<byte> codes)
    {
        System.Text.StringBuilder sb = new(codes.Count * 2);
        for (int i = 0; i < codes.Count; i++)
            if (TryGet(codes[i], out EspeakPhoneme ph))
                sb.Append(ph.MnemonicText);
        return sb.ToString();
    }

    /// <summary>Looks up the code for a packed mnemonic (espeak <c>PhonemeCode</c>), or 0 when not found.</summary>
    public int CodeForMnemonic(uint mnemonic)
    {
        for (int code = 0; code <= MaxCode; code++)
            if (_present[code] && _byCode[code].Mnemonic == mnemonic)
                return _byCode[code].Code;
        return 0;
    }

    private List<TableInfo> ParseTables(byte[] data)
    {
        int nTables = data[0];
        int p = 4;
        List<TableInfo> tables = new(nTables);
        for (int i = 0; i < nTables; i++)
        {
            int nPhonemes = data[p];
            int includes = data[p + 1];
            p += 4;
            string name = ReadName(data, p);
            p += TableNameLength;
            tables.Add(new TableInfo(name, includes, p, nPhonemes));
            p += nPhonemes * PhonemeRecordSize;
        }
        return tables;
    }

    // Port of SelectPhonemeTable/SetUpPhonemeTable: recurse into the included base table, then overlay this table.
    private void BuildActive(byte[] data, List<TableInfo> tables, int number)
    {
        TableInfo table = tables[number];
        if (table.Includes > 0)
            BuildActive(data, tables, table.Includes - 1);

        int p = table.EntriesOffset;
        for (int ix = 0; ix < table.NumPhonemes; ix++)
        {
            EspeakPhoneme ph = EspeakPhoneme.Read(data, p);
            _byCode[ph.Code] = ph;
            _present[ph.Code] = true;
            if (ph.Code > MaxCode)
                MaxCode = ph.Code;
            p += PhonemeRecordSize;
        }
    }

    private static string ReadName(byte[] data, int offset)
    {
        int end = offset;
        int limit = offset + TableNameLength;
        while (end < limit && data[end] != 0)
            end++;
        return System.Text.Encoding.ASCII.GetString(data, offset, end - offset);
    }

    private readonly record struct TableInfo(string Name, int Includes, int EntriesOffset, int NumPhonemes);
}
