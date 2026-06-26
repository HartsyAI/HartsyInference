using System.Reflection;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.Phonemizer.Espeak;

/// <summary>Maps espeak phoneme mnemonics to IPA, loaded from an embedded table generated offline by aligning
/// espeak-ng's ascii and IPA phoneme output. The TTS models consume IPA, while the ported translation pipeline emits
/// phoneme codes whose mnemonics come from the phoneme table; this bridges the two.</summary>
internal sealed class EspeakIpaMap
{
    private const string ResourcePrefix = "HartsyInference.Phonemizer.Resources.ipa_phoneme_map";

    private readonly Dictionary<string, string> _map;

    private EspeakIpaMap(Dictionary<string, string> map) => _map = map;

    /// <summary>Loads the embedded mnemonic-to-IPA table for a phoneme table (e.g. <c>en</c>, <c>en-us</c>). The mapping
    /// differs per table (British <c>əʊ</c> vs American <c>oʊ</c>); falls back to the generic table.</summary>
    public static EspeakIpaMap Load(string table = "en")
    {
        Assembly asm = typeof(EspeakIpaMap).Assembly;
        Stream? s = asm.GetManifestResourceStream($"{ResourcePrefix}.{table}.tsv")
            ?? asm.GetManifestResourceStream($"{ResourcePrefix}.tsv")
            ?? throw new HartsyInferenceException($"Embedded IPA map resource for table '{table}' is missing.");
        using Stream _ = s;
        using StreamReader reader = new(s, System.Text.Encoding.UTF8);
        Dictionary<string, string> map = new(StringComparer.Ordinal);
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            int tab = line.IndexOf('\t');
            if (tab <= 0) continue;
            map[line[..tab]] = line[(tab + 1)..];
        }
        return new EspeakIpaMap(map);
    }

    /// <summary>Converts a sequence of phoneme codes to an IPA string, mapping each code through its phoneme-table
    /// mnemonic. Codes without an IPA mapping fall back to their printable mnemonic so nothing is silently dropped.</summary>
    public string ToIpa(IReadOnlyList<byte> codes, EspeakPhonemeTable phonemeTable)
    {
        System.Text.StringBuilder sb = new(codes.Count * 2);
        for (int i = 0; i < codes.Count; i++)
        {
            if (!phonemeTable.TryGet(codes[i], out EspeakPhoneme ph))
                continue;
            string mnemonic = StripControl(ph.MnemonicText);
            if (mnemonic.Length == 0)
                continue;
            if (_map.TryGetValue(mnemonic, out string? ipa))
            {
                sb.Append(ipa);
                continue;
            }
            // espeak appends liaison/length markers ('#', '%', '-') and variant digits ('@2', '02') to a phoneme's
            // internal mnemonic; its IPA output collapses these to the base phoneme. Fall back to the base IPA.
            sb.Append(FallbackIpa(mnemonic));
        }
        return sb.ToString();
    }

    // Resolve an unmapped mnemonic to IPA by progressively stripping trailing markers / variant digits to the base
    // phoneme; if nothing matches, pass the mnemonic through so nothing is silently dropped.
    private string FallbackIpa(string mnemonic)
    {
        string m = mnemonic.TrimEnd('#', '%', '-');
        if (m != mnemonic && _map.TryGetValue(m, out string? ipa)) return ipa;
        if (m.Length > 1 && char.IsAsciiDigit(m[^1]) && _map.TryGetValue(m[..^1], out string? digitIpa)) return digitIpa;
        return mnemonic;
    }

    private static string StripControl(string mnemonic)
    {
        int n = 0;
        Span<char> buf = stackalloc char[mnemonic.Length];
        foreach (char c in mnemonic)
            if (c >= ' ') buf[n++] = c;
        return new string(buf[..n]);
    }
}
