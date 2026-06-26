using HartsyInference.Core.Logging;

namespace HartsyInference.Phonemizer.Espeak;

/// <summary>A language/voice variant resolved from an espeak-ng <c>lang/&lt;group&gt;/&lt;name&gt;</c> file. Captures the
/// pieces that change the phonemized output: which phoneme table to use (<c>phonemes</c>), which conditional dictionary
/// rules to enable (<c>dictrules</c> → <c>dict_condition</c> bits), the American flap-t option (<c>option reduce_t</c>),
/// and phoneme <c>replace</c>s. English (en-gb) sets none of these; <c>en-us</c> switches to the American phoneme table
/// and enables conditions 3 and 6.</summary>
internal sealed class EspeakVoiceVariant
{
    /// <summary>Base language whose <c>&lt;name&gt;_dict</c> is loaded (region stripped: <c>en-us</c> → <c>en</c>).</summary>
    public string DictName { get; private init; } = "en";

    /// <summary>Phoneme table to resolve from <c>phontab</c> (e.g. <c>en</c> or <c>en-us</c>).</summary>
    public string PhonemeTable { get; private init; } = "en";

    /// <summary>Bit set of enabled dictionary conditions (from <c>dictrules</c>).</summary>
    public int DictCondition { get; private init; }

    /// <summary>American flap-t reduction (<c>option reduce_t</c>).</summary>
    public bool ReduceT { get; private init; }

    /// <summary>Phoneme replacements: <c>(condition, fromMnemonic, toMnemonic)</c> applied to the output.</summary>
    public IReadOnlyList<(int Cond, string From, string To)> Replaces { get; private init; } = [];

    /// <summary>Resolves the variant for <paramref name="language"/> (e.g. "en", "en-us") from the espeak data dir. If
    /// no lang file is found, returns a sensible default (dict + table = the language's base, no conditions).</summary>
    public static EspeakVoiceVariant Resolve(string dataDir, string language)
    {
        string baseName = language.Split('-')[0];
        EspeakVoiceVariant fallback = new() { DictName = baseName, PhonemeTable = baseName };

        string? file = FindLangFile(dataDir, language);
        if (file is null)
            return fallback;

        try
        {
            return Parse(File.ReadAllLines(file), baseName);
        }
        catch (Exception ex)
        {
            Logs.Error($"Failed to parse espeak variant '{file}': {ex.Message}");
            return fallback;
        }
    }

    private static EspeakVoiceVariant Parse(string[] lines, string baseName)
    {
        string phonemes = baseName;
        int dictCondition = 0;
        bool reduceT = false;
        List<(int, string, string)> replaces = new();

        foreach (string raw in lines)
        {
            string line = raw.Trim();
            if (line.Length == 0 || line[0] == '/') continue;
            string[] tok = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            switch (tok[0])
            {
                case "phonemes" when tok.Length >= 2:
                    phonemes = tok[1];
                    break;
                case "dictrules":
                    for (int i = 1; i < tok.Length; i++)
                        if (int.TryParse(tok[i], out int n) && n is > 0 and < 32) dictCondition |= 1 << n;
                    break;
                case "option" when tok.Length >= 2 && tok[1] == "reduce_t":
                    reduceT = tok.Length < 3 || tok[2] != "0";
                    break;
                case "replace" when tok.Length >= 4:
                    if (int.TryParse(tok[1], out int cond)) replaces.Add((cond, tok[2], tok[3]));
                    break;
            }
        }

        return new EspeakVoiceVariant
        {
            DictName = baseName,
            PhonemeTable = phonemes,
            DictCondition = dictCondition,
            ReduceT = reduceT,
            Replaces = replaces,
        };
    }

    // espeak lang files live under lang/<group>/<name>; the name may differ in case from the language code
    // (en-us -> file "en-US"). Match by the file's `language` directive, falling back to a filename match.
    private static string? FindLangFile(string dataDir, string language)
    {
        string langRoot = Path.Combine(dataDir, "lang");
        if (!Directory.Exists(langRoot)) return null;

        string target = language.ToLowerInvariant();
        // Prefer the file whose primary `language` directive (the first one) matches at the best priority. A file may
        // list a language as a low-priority alternate (en-US has `language en 3`), so an exact match must win.
        string? best = null;
        int bestPriority = int.MaxValue;
        foreach (string file in Directory.EnumerateFiles(langRoot, "*", SearchOption.AllDirectories))
        {
            // Exact filename match (e.g. "en", "en-US") is authoritative.
            if (string.Equals(Path.GetFileName(file), language, StringComparison.OrdinalIgnoreCase))
                return file;

            foreach (string line in ReadHeadLines(file, 12))
            {
                string t = line.Trim();
                if (!t.StartsWith("language ", StringComparison.Ordinal)) continue;
                string[] tok = t.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
                if (tok.Length < 2 || !string.Equals(tok[1], target, StringComparison.OrdinalIgnoreCase)) continue;
                int priority = tok.Length >= 3 && int.TryParse(tok[2], out int pr) ? pr : 5;
                if (priority < bestPriority) { bestPriority = priority; best = file; }
            }
        }
        return best;
    }

    private static IEnumerable<string> ReadHeadLines(string file, int count)
    {
        using StreamReader r = new(file);
        for (int i = 0; i < count; i++)
        {
            string? line = r.ReadLine();
            if (line is null) yield break;
            yield return line;
        }
    }
}
