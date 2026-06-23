using System.Collections.Generic;

namespace HartsyInference.Audio.Frontends;

/// <summary>Language front-end for GPT-SoVITS — the engine equivalent of upstream <c>cleaner.clean_text</c>.
/// Routes text to the per-language G2P and returns the phoneme symbols, the per-character phoneme counts
/// (<c>word2ph</c>, non-null only for character languages), and the (caller-normalized) text. The phoneme ids
/// (<see cref="GptSoVitsSymbols.ToSequence"/>) feed s1's text span and s2's MRTE text stream; <c>word2ph</c>
/// expands the BERT character features (<see cref="GptSoVitsSymbols.ExpandBertToPhonemes"/>).</summary>
public static class GptSoVitsFrontend
{
    /// <summary>Supported language codes (those with a ported G2P).</summary>
    public static readonly IReadOnlyList<string> SupportedLanguages = ["zh", "en"];

    /// <summary>Cleans + phonemizes <paramref name="text"/> for the given language code. Any phone outside the
    /// symbol set becomes <c>UNK</c> (matching upstream). Throws for languages whose G2P isn't ported yet.</summary>
    public static (List<string> Phones, List<int>? Word2Ph) CleanText(string text, string language)
    {
        (List<string> phones, List<int>? word2ph) = language switch
        {
            "zh" => Wrap(ChineseG2P.G2P(text)),
            "en" => (GptSoVitsEnglishG2P.G2P(text), (List<int>?)null),
            _ => throw new System.NotSupportedException(
                $"GPT-SoVITS G2P for language '{language}' is not yet available (ported: {string.Join(", ", SupportedLanguages)}). "
                + "Japanese (OpenJTalk) and Korean require their dictionaries; supply phoneme ids directly meanwhile."),
        };

        for (int i = 0; i < phones.Count; i++)
            if (GptSoVitsSymbols.IdOf(phones[i]) < 0) phones[i] = "UNK";
        return (phones, word2ph);
    }

    private static (List<string>, List<int>?) Wrap((List<string> Phones, List<int> Word2Ph) zh)
        => (zh.Phones, zh.Word2Ph);
}
