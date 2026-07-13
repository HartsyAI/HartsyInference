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
        // Affricates use misaki's single-char symbols (ʧ/ʤ), NOT decomposed tʃ/dʒ — Kokoro's vocab was
        // trained on the misaki phoneme set, where each diphthong/affricate is ONE token. Feeding the
        // two-char IPA (t+ʃ, d+ʒ) makes the model articulate two separate phonemes → mispronunciation.
        ["B"] = "b", ["CH"] = "ʧ", ["D"] = "d", ["DH"] = "ð", ["F"] = "f", ["G"] = "ɡ", ["HH"] = "h",
        ["JH"] = "ʤ", ["K"] = "k", ["L"] = "l", ["M"] = "m", ["N"] = "n", ["NG"] = "ŋ", ["P"] = "p",
        ["R"] = "ɹ", ["S"] = "s", ["SH"] = "ʃ", ["T"] = "t", ["TH"] = "θ", ["V"] = "v", ["W"] = "w",
        ["Y"] = "j", ["Z"] = "z", ["ZH"] = "ʒ",
    };

    private static readonly Dictionary<string, string> Vowels = new(StringComparer.Ordinal)
    {
        // Diphthongs use misaki's single-char symbols (A/I/W/O/Y), NOT the decomposed IPA (eɪ/aɪ/aʊ/oʊ/ɔɪ):
        // Kokoro's vocab has BOTH, but the model was trained with the single tokens for diphthongs; the plain
        // vowels (e, ɪ, a, ʊ, o) are separate monophthong tokens, so "lazy" lˈeɪzi → "l-eh-ih-zee" ≈ "easy".
        //   AW=aʊ→W, AY=aɪ→I, EY=eɪ→A, OW=oʊ→O, OY=ɔɪ→Y  (misaki US gold set).
        ["AA"] = "ɑ", ["AE"] = "æ", ["AO"] = "ɔ", ["AW"] = "W", ["AY"] = "I", ["EH"] = "ɛ", ["EY"] = "A",
        ["IH"] = "ɪ", ["IY"] = "i", ["OW"] = "O", ["OY"] = "Y", ["UH"] = "ʊ", ["UW"] = "u",
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
            // misaki writes the r-colored vowels as a vowel + ɹ (ɜɹ / əɹ), not the single glyphs ɝ/ɚ
            // (which aren't the tokens Kokoro was trained on): "bird" bˈɜɹd, "butter" bˈʌTəɹ.
            case "ER": return stress == 0 ? "əɹ" : "ɜɹ";
        }
        if (Vowels.TryGetValue(basePhone, out string? v))
        {
            return v;
        }
        return Consonants.TryGetValue(basePhone, out string? c) ? c : "";
    }
}
