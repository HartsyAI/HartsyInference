using System.Reflection;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.Phonemizer.Espeak;

/// <summary>Maps espeak phoneme mnemonics to IPA, loaded from an embedded table generated offline by aligning
/// espeak-ng's ascii and IPA phoneme output. The TTS models consume IPA, while the ported translation pipeline emits
/// phoneme codes whose mnemonics come from the phoneme table; this bridges the two.</summary>
internal sealed class EspeakIpaMap
{
    private const string ResourceName = "HartsyInference.Phonemizer.Resources.ipa_phoneme_map.tsv";

    private readonly Dictionary<string, string> _map;

    private EspeakIpaMap(Dictionary<string, string> map) => _map = map;

    /// <summary>Loads the embedded mnemonic-to-IPA table.</summary>
    public static EspeakIpaMap Load()
    {
        Assembly asm = typeof(EspeakIpaMap).Assembly;
        using Stream? s = asm.GetManifestResourceStream(ResourceName)
            ?? throw new HartsyInferenceException($"Embedded IPA map resource '{ResourceName}' is missing.");
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
            // espeak appends liaison/length markers ('#', '%', '-') to a phoneme's internal mnemonic that its IPA
            // output drops; fall back to the base phoneme's IPA after stripping them.
            string baseMnem = mnemonic.TrimEnd('#', '%', '-');
            if (baseMnem.Length != mnemonic.Length && _map.TryGetValue(baseMnem, out string? baseIpa))
                sb.Append(baseIpa);
            else
                sb.Append(mnemonic);
        }
        return sb.ToString();
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
