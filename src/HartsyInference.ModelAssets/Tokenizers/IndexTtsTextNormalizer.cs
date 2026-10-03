using System.Text;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Text pre-processing for <see cref="IndexTtsTokenizer"/>'s Unigram SentencePiece model: a faithful port
/// of the reference <c>tokenize_by_CJK_char</c> (<c>indextts/utils/common.py</c>) — every CJK code point becomes
/// its own space-separated token, and every run of non-CJK text is uppercased.</summary>
/// <remarks>The uppercasing is NOT cosmetic: the real <c>bpe.model</c>'s ~12k-entry vocabulary contains zero
/// lowercase English word pieces (verified directly against the downloaded checkpoint and the real
/// <c>sentencepiece</c> library) — every English training example was uppercased before BPE training, so
/// lowercase/mixed-case input tokenizes almost entirely to <c>&lt;unk&gt;</c>. An earlier version of this method
/// only inserted boundary spaces and never uppercased, which produced unintelligible generations end-to-end
/// despite every other component loading and running without error. Decode-side case restoration
/// (<c>de_tokenized_by_CJK_char</c>'s <c>do_lower_case</c>) is intentionally not ported — nothing in this
/// pipeline decodes token ids back to display text.</remarks>
public static class IndexTtsTextNormalizer
{
    /// <summary>Splits <paramref name="text"/> so every CJK code point is its own token and every run of non-CJK
    /// text is uppercased, then rejoins with single spaces — e.g. <c>"你好世界是 hello world 的中文"</c> becomes
    /// <c>"你 好 世 界 是 HELLO WORLD 的 中 文"</c>, exactly matching the reference.</summary>
    public static string InjectCjkBoundaries(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        text = text.Trim();
        if (text.Length == 0) return string.Empty;

        List<string> segments = [];
        StringBuilder current = new();
        for (int i = 0; i < text.Length;)
        {
            int len = char.IsSurrogatePair(text, i) ? 2 : 1;
            int codePoint = char.ConvertToUtf32(text, i);
            if (IsCjkCodePoint(codePoint))
            {
                if (current.Length > 0) { segments.Add(current.ToString()); current.Clear(); }
                segments.Add(text.Substring(i, len));
            }
            else
            {
                current.Append(text, i, len);
            }
            i += len;
        }
        if (current.Length > 0) segments.Add(current.ToString());

        List<string> parts = new(segments.Count);
        foreach (string segment in segments)
        {
            string trimmed = segment.Trim();
            if (trimmed.Length > 0) parts.Add(trimmed.ToUpperInvariant());
        }
        return string.Join(' ', parts);
    }

    /// <summary>The reference's exact CJK range set (<c>indextts/utils/common.py</c>'s <c>CJK_RANGE_PATTERN</c>,
    /// itself sourced from nltk): Hangul Jamo, the CJK radicals/symbols/Hiragana/Katakana/Bopomofo/Unified
    /// Ideographs/Yi super-block, Hangul Syllables, CJK Compatibility Ideographs, CJK Compatibility Forms,
    /// Halfwidth Jamo/Forms, and the supplementary CJK planes — deliberately much broader than a bare "CJK
    /// Unified Ideographs" range.</summary>
    private static bool IsCjkCodePoint(int cp) =>
        (cp >= 0x1100 && cp <= 0x11FF) ||
        (cp >= 0x2E80 && cp <= 0xA4CF) ||
        (cp >= 0xA840 && cp <= 0xD7AF) ||
        (cp >= 0xF900 && cp <= 0xFAFF) ||
        (cp >= 0xFE30 && cp <= 0xFE4F) ||
        (cp >= 0xFF65 && cp <= 0xFFDC) ||
        (cp >= 0x20000 && cp <= 0x2FFFF);
}
