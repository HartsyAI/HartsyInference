using System.Text;
using HartsyInference.Audio.Phonemizer.Espeak;

namespace HartsyInference.Audio.Models.Zonos;

/// <summary>Tokenizes text into the Zonos phoneme-embedder ids (vocab 189). Matches
/// <c>zonos/conditioning.py</c> exactly: espeak-ng phonemization (per-language, with stress, punctuation
/// preserved) → a per-Unicode-codepoint lookup into a fixed 185-symbol table (punctuation + ASCII letters + IPA
/// letters), offset by the 4 special tokens (PAD 0, UNK 1, BOS 2, EOS 3), with unknown codepoints mapped to UNK,
/// then wrapped <c>[BOS, …ids…, EOS]</c>. The table is character-level over the whole IPA string, so spaces and
/// punctuation are tokenized too.</summary>
/// <param name="phonemizer">espeak-ng phonemizer (reused across calls).</param>
/// <param name="language">espeak voice code, e.g. <c>"en-us"</c>.</param>
public sealed class ZonosPhonemeTokenizer(EspeakPhonemizer phonemizer, string language)
{
    public const int PadId = 0;
    public const int UnkId = 1;
    public const int BosId = 2;
    public const int EosId = 3;
    private const int SpecialCount = 4;

    // Unicode codepoints of the 185 symbols (in id order: id = SpecialCount + index), from
    // zonos.conditioning.symbols = [*_punctuation, *_letters, *_letters_ipa]. Two entries share codepoint 39
    // (apostrophe); as in the Python dict comprehension the later index wins.
    private static readonly int[] SymbolCodepoints =
    [
        59, 58, 44, 46, 33, 63, 161, 191, 8212, 8230, 34, 171, 187, 8220, 8221, 40, 41, 32, 42, 126, 45, 47, 92, 38,
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
        97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
        119, 120, 121, 122,
        593, 592, 594, 230, 595, 665, 946, 596, 597, 231, 599, 598, 240, 676, 601, 600, 602, 603, 604, 605, 606, 607,
        644, 609, 608, 610, 667, 614, 615, 295, 613, 668, 616, 618, 669, 621, 620, 619, 622, 671, 625, 623, 624, 331,
        627, 626, 628, 248, 629, 632, 952, 339, 630, 664, 633, 634, 638, 635, 640, 641, 637, 642, 643, 648, 679, 649,
        650, 651, 11377, 652, 611, 612, 653, 967, 654, 655, 657, 656, 658, 660, 673, 661, 674, 448, 449, 450, 451,
        712, 716, 720, 721, 700, 692, 688, 689, 690, 695, 736, 740, 734, 8595, 8593, 8594, 8599, 8600, 39, 809, 39, 7547,
    ];

    private static readonly Dictionary<int, int> SymbolToId = BuildTable();

    private readonly EspeakPhonemizer _phonemizer = phonemizer;
    private readonly string _language = language;

    /// <summary>Phonemizes <paramref name="text"/> and returns the wrapped phoneme ids <c>[BOS, …, EOS]</c>.</summary>
    public int[] Tokenize(string text)
    {
        string ipa = _phonemizer.PhonemizeToIpa(text, _language, preservePunctuation: true);
        return TokenizeIpa(NormalizePunctuation(ipa));
    }

    /// <summary>Drops the space the engine's espeak inserts before a punctuation mark, matching the Python
    /// phonemizer's <c>preserve_punctuation</c> re-attachment (e.g. <c>"oʊ , ðɪs"</c> → <c>"oʊ, ðɪs"</c>).</summary>
    public static string NormalizePunctuation(string ipa)
    {
        StringBuilder sb = new(ipa.Length);
        for (int i = 0; i < ipa.Length; i++)
        {
            char c = ipa[i];
            if (c == ' ' && i + 1 < ipa.Length && IsAttachablePunctuation(ipa[i + 1])) continue;
            sb.Append(c);
        }
        return sb.ToString();
    }

    private static bool IsAttachablePunctuation(char c) =>
        c is ';' or ':' or ',' or '.' or '!' or '?' or ')' or '…' or '»' or '”';

    /// <summary>Maps an already-phonemized IPA string to wrapped phoneme ids <c>[BOS, …, EOS]</c>.</summary>
    public static int[] TokenizeIpa(string ipa)
    {
        List<int> ids = new(ipa.Length + 2) { BosId };
        foreach (Rune rune in ipa.EnumerateRunes())
            ids.Add(SymbolToId.TryGetValue(rune.Value, out int id) ? id : UnkId);
        ids.Add(EosId);
        return ids.ToArray();
    }

    private static Dictionary<int, int> BuildTable()
    {
        Dictionary<int, int> map = new(SymbolCodepoints.Length);
        for (int i = 0; i < SymbolCodepoints.Length; i++)
            map[SymbolCodepoints[i]] = SpecialCount + i;   // later duplicate wins (matches the Python dict)
        return map;
    }
}
