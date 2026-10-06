using System.Text;

namespace HartsyInference.Audio.Frontends;

/// <summary>Kokoro-82M v1.0's Mandarin front-end: a port of misaki's <c>zh.ZHG2P</c> (hexgrad/misaki, Apache-2.0) on
/// its legacy path. Arabic numbers become Chinese numerals (cn2an), full-width punctuation becomes ASCII, each Han run
/// is cut into words by jieba and each word read by pypinyin, and each syllable becomes IPA with tone arrows. Words are
/// joined by spaces; text outside Han runs passes through as written. Dictionaries come from jieba 0.42.1 and
/// pypinyin 0.55.0 (both MIT) and are loaded by the caller, so nothing third-party ships in the binary.</summary>
public sealed class KokoroMandarinG2P
{
    private readonly JiebaSegmenter _jieba;
    private readonly PinyinConverter _pinyin;

    private KokoroMandarinG2P(JiebaSegmenter jieba, PinyinConverter pinyin)
    {
        _jieba = jieba;
        _pinyin = pinyin;
    }

    /// <summary>Builds the front-end from jieba's <c>dict.txt</c> and <c>finalseg/prob_emit.py</c> and pypinyin's
    /// <c>phrases_dict.json</c> and <c>pinyin_dict.json</c>.</summary>
    public static KokoroMandarinG2P FromStreams(Stream jiebaDictionary, Stream jiebaProbEmit, Stream pinyinPhrases,
        Stream pinyinSingles)
        => new(JiebaSegmenter.FromStreams(jiebaDictionary, jiebaProbEmit),
            PinyinConverter.FromStreams(pinyinPhrases, pinyinSingles));

    /// <summary>As <see cref="FromStreams"/>, from files on disk.</summary>
    public static KokoroMandarinG2P FromFiles(string jiebaDictionaryPath, string jiebaProbEmitPath,
        string pinyinPhrasesPath, string pinyinSinglesPath)
        => new(JiebaSegmenter.FromFiles(jiebaDictionaryPath, jiebaProbEmitPath),
            PinyinConverter.FromFiles(pinyinPhrasesPath, pinyinSinglesPath));

    /// <summary>The phoneme string Kokoro takes for <paramref name="text"/> (<c>ZHG2P()(text)[0]</c>).</summary>
    public string ToIpa(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (PythonStrip(text).Length == 0) return "";
        text = PythonStrip(MapPunctuation(ChineseNumberNormalizer.Transform(text)));
        StringBuilder sb = new(text.Length * 4);
        int i = 0;
        while (i < text.Length)
        {
            int start = i;
            bool han = IsHan(text[i]);
            while (i < text.Length && IsHan(text[i]) == han) i++;
            string segment = text[start..i];
            if (!han)
            {
                sb.Append(segment);
                continue;
            }
            List<string> words = _jieba.Cut(segment);
            for (int w = 0; w < words.Count; w++)
            {
                if (w > 0) sb.Append(' ');
                AppendWord(words[w], sb);
            }
        }
        // misaki drops the non-syllabic mark U+032F (ai̯ → ai) everywhere, text outside Han runs included.
        return sb.Replace("̯", "").ToString();
    }

    /// <summary>The jieba words of a Han run, as <c>jieba.lcut</c> cuts it.</summary>
    internal List<string> Segment(string han) => _jieba.Cut(han);

    /// <summary>The tone-number pinyin of a word, as <c>lazy_pinyin(word, Style.TONE3, neutral_tone_with_five=True)</c>
    /// returns it.</summary>
    internal List<string> Pinyin(string word) => _pinyin.LazyPinyin(word);

    // ZHG2P.word2ipa. A syllable with no reading (a character pypinyin lacks) makes misaki raise; here it is dropped.
    private void AppendWord(string word, StringBuilder sb)
    {
        foreach (string syllable in _pinyin.LazyPinyin(word))
        {
            string? ipa = PinyinToIpa.Convert(syllable);
            if (ipa is not null) sb.Append(ipa);
        }
    }

    // ZHG2P.map_punctuation.
    private static string MapPunctuation(string text) => new StringBuilder(text)
        .Replace("、", ", ").Replace("，", ", ").Replace("。", ". ").Replace("．", ". ").Replace("！", "! ")
        .Replace("：", ": ").Replace("；", "; ").Replace("？", "? ").Replace("«", " “").Replace("»", "” ")
        .Replace("《", " “").Replace("》", "” ").Replace("「", " “").Replace("」", "” ").Replace("【", " “")
        .Replace("】", "” ").Replace("（", " (").Replace("）", ") ").ToString();

    private static bool IsHan(char c) => c >= '一' && c <= '鿿';

    // Python str.strip(): trims what str.isspace() calls whitespace.
    private static string PythonStrip(string text)
    {
        int start = 0, end = text.Length;
        while (start < end && JiebaSegmenter.IsPythonSpace(text[start])) start++;
        while (end > start && JiebaSegmenter.IsPythonSpace(text[end - 1])) end--;
        return text[start..end];
    }
}
