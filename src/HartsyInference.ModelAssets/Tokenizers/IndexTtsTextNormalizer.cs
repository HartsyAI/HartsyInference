using System.Text;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Text pre-processing for <see cref="IndexTtsTokenizer"/>'s Unigram SentencePiece model: inserting
/// whitespace at CJK/non-CJK boundaries so the segmenter doesn't merge a Chinese run into neighboring Latin
/// text or punctuation.</summary>
/// <remarks>Phase-1 scope: boundary spacing only. IndexTTS's own text frontend additionally supports explicit
/// pinyin-bracket annotations (e.g. <c>zhong1 guo2</c>) for pronunciation override; that is not implemented here.</remarks>
public static class IndexTtsTextNormalizer
{
    /// <summary>Inserts a space at every boundary between a CJK Unified Ideograph and a non-CJK character (and
    /// vice versa), leaving CJK-CJK and non-CJK–non-CJK runs untouched.</summary>
    public static string InjectCjkBoundaries(ReadOnlySpan<char> text)
    {
        if (text.IsEmpty) return string.Empty;
        StringBuilder sb = new(text.Length + 8);
        bool? prevIsCjk = null;
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            bool isCjk = IsCjkCodePoint(c);
            if (prevIsCjk is bool prev && prev != isCjk && !char.IsWhiteSpace(c) &&
                sb.Length > 0 && !char.IsWhiteSpace(sb[^1]))
            {
                sb.Append(' ');
            }
            sb.Append(c);
            prevIsCjk = isCjk;
        }
        return sb.ToString();
    }

    /// <summary>CJK Unified Ideographs plus the common extension blocks used by IndexTTS's training corpus (Chinese).</summary>
    private static bool IsCjkCodePoint(char c) =>
        (c >= 0x4E00 && c <= 0x9FFF) ||   // CJK Unified Ideographs
        (c >= 0x3400 && c <= 0x4DBF) ||   // CJK Extension A
        (c >= 0xF900 && c <= 0xFAFF) ||   // CJK Compatibility Ideographs
        (c >= 0x3000 && c <= 0x303F) ||   // CJK Symbols and Punctuation
        (c >= 0xFF00 && c <= 0xFFEF);     // Halfwidth/Fullwidth Forms (CJK punctuation variants)
}
