using System.Text;

namespace HartsyInference.Audio.Frontends;

/// <summary>Converts a CMUdict ARPAbet phone sequence (e.g. <c>["HH","AH0","L","OW1"]</c>) to an IPA string
/// (e.g. <c>həˈloʊ</c>) in the symbol set Kokoro / StyleTTS2 consume. Stress digits map to IPA stress marks
/// (1→ˈ, 2→ˌ, 0→none); the reduced vowels AH0/ER0 become ə/ɚ. The mark is placed immediately before the
/// stressed vowel — an approximation of misaki's per-syllable placement, intelligible but not identical.</summary>
public static class ArpabetToIpa
{
    private static readonly Dictionary<string, string> Consonants = new(StringComparer.Ordinal)
    {
        ["B"] = "b", ["CH"] = "tʃ", ["D"] = "d", ["DH"] = "ð", ["F"] = "f", ["G"] = "ɡ", ["HH"] = "h",
        ["JH"] = "dʒ", ["K"] = "k", ["L"] = "l", ["M"] = "m", ["N"] = "n", ["NG"] = "ŋ", ["P"] = "p",
        ["R"] = "ɹ", ["S"] = "s", ["SH"] = "ʃ", ["T"] = "t", ["TH"] = "θ", ["V"] = "v", ["W"] = "w",
        ["Y"] = "j", ["Z"] = "z", ["ZH"] = "ʒ",
    };

    private static readonly Dictionary<string, string> Vowels = new(StringComparer.Ordinal)
    {
        ["AA"] = "ɑ", ["AE"] = "æ", ["AO"] = "ɔ", ["AW"] = "aʊ", ["AY"] = "aɪ", ["EH"] = "ɛ", ["EY"] = "eɪ",
        ["IH"] = "ɪ", ["IY"] = "i", ["OW"] = "oʊ", ["OY"] = "ɔɪ", ["UH"] = "ʊ", ["UW"] = "u",
        // AH and ER are stress-dependent (reduced vs full) — handled in MapBase.
    };

    /// <summary>Converts one word's ARPAbet phones to a contiguous IPA string (no spaces).</summary>
    public static string ConvertWord(IReadOnlyList<string> phones)
    {
        StringBuilder sb = new();
        foreach (string raw in phones)
        {
            string p = raw.Trim().ToUpperInvariant();
            if (p.Length == 0)
            {
                continue;
            }
            int stress = -1;
            string basePhone = p;
            char last = p[^1];
            if (last is '0' or '1' or '2')
            {
                stress = last - '0';
                basePhone = p[..^1];
            }
            string ipa = MapBase(basePhone, stress);
            if (ipa.Length == 0)
            {
                continue;
            }
            if (stress == 1)
            {
                sb.Append('ˈ');
            }
            else if (stress == 2)
            {
                sb.Append('ˌ');
            }
            sb.Append(ipa);
        }
        return sb.ToString();
    }

    private static string MapBase(string basePhone, int stress)
    {
        switch (basePhone)
        {
            case "AH": return stress == 0 ? "ə" : "ʌ";
            case "ER": return stress == 0 ? "ɚ" : "ɝ";
        }
        if (Vowels.TryGetValue(basePhone, out string? v))
        {
            return v;
        }
        return Consonants.TryGetValue(basePhone, out string? c) ? c : "";
    }
}
