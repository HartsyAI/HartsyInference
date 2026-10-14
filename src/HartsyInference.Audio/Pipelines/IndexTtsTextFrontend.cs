using System.Text.RegularExpressions;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Audio.Pipelines;

/// <summary>The upstream IndexTTS text path from raw text to the per-segment token ids the GPT embeds, for both
/// tokenizer families. SentencePiece (1.5 and 2.0): <c>TextTokenizer.tokenize</c> (normalize, CJK split + uppercase,
/// pieces), <c>split_segments</c>, ids. Tiktoken (2.5): <c>infer_generator</c>'s character map, normalize, lowercase,
/// pronunciation tokens, <c>split_text_by_tokens</c> and <c>encode("&lt;|lang|&gt; " + segment)</c>. Every returned id
/// list has the start/stop text sentinels removed, as <c>UnifiedVoice.prepare_gpt_inputs</c> does before re-wrapping.</summary>
/// <remarks>Not ported: the 2.5 Japanese/Spanish NeMo normalizer and Japanese morpheme spacing; those languages pass
/// through un-normalized.</remarks>
public sealed partial class IndexTtsTextFrontend
{
    /// <summary>Upstream <c>max_text_tokens_per_segment</c> default.</summary>
    public const int DefaultMaxTextTokensPerSegment = 120;

    private const int StartTextToken = 0;
    private const int StopTextToken = 1;

    /// <summary>Creates the frontend over an upstream-equivalent normalizer.</summary>
    public IndexTtsTextFrontend(IndexTtsFrontNormalizer normalizer)
    {
        ArgumentNullException.ThrowIfNull(normalizer);
        Normalizer = normalizer;
    }

    /// <summary>The normalizer this frontend runs.</summary>
    public IndexTtsFrontNormalizer Normalizer { get; }

    /// <summary>IndexTTS-1.5 / 2.0: segments of GPT text ids.</summary>
    public List<int[]> SegmentSentencePiece(string text, IndexTtsTokenizer tokenizer, int maxTokensPerSegment = DefaultMaxTextTokensPerSegment,
        int quickStreamingTokens = 0)
    {
        List<IndexTts2TextSegmenter.Token[]> segments = SplitSentencePiece(text, tokenizer, maxTokensPerSegment, quickStreamingTokens);
        List<int[]> result = new(segments.Count);
        foreach (IndexTts2TextSegmenter.Token[] segment in segments)
        {
            int[] ids = new int[segment.Length];
            for (int i = 0; i < ids.Length; i++) ids[i] = segment[i].Id;
            result.Add(StripTextSentinels(ids));
        }
        return result;
    }

    /// <summary>IndexTTS-2.5: segments of GPT text ids, each starting with the language-tag token.</summary>
    /// <param name="encode">Tiktoken encode with every special token allowed (<c>allowed_special="all"</c>).</param>
    /// <param name="textPositionCapacity">Rows of the GPT's <c>text_pos_embedding</c> (602 for the released 2.5).</param>
    public List<int[]> SegmentTiktoken(string text, string language, Func<string, int[]> encode, int textPositionCapacity,
        int maxTokensPerSegment = DefaultMaxTextTokensPerSegment)
    {
        ArgumentNullException.ThrowIfNull(encode);
        string prefix = LanguagePrefix(language);
        List<string> segments = SplitByTokens(PrepareTiktoken(text, language).Prepared, maxTokensPerSegment, textPositionCapacity, prefix,
            s => encode(s).Length);
        List<int[]> result = new(segments.Count);
        foreach (string segment in segments) result.Add(StripTextSentinels(EncodeTiktokenSegment(segment, prefix, encode)));
        return result;
    }

    /// <summary><c>TextTokenizer.tokenize</c> + <c>split_segments</c>: tokenizer pieces cut into segments.</summary>
    internal List<IndexTts2TextSegmenter.Token[]> SplitSentencePiece(string text, IndexTtsTokenizer tokenizer, int maxTokensPerSegment,
        int quickStreamingTokens)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        string? prepared = PrepareSentencePiece(text);
        if (prepared is null) return [];
        IReadOnlyList<(string Piece, int Id)> pieces = tokenizer.EncodeToPieces(prepared);
        IndexTts2TextSegmenter.Token[] tokens = new IndexTts2TextSegmenter.Token[pieces.Count];
        for (int i = 0; i < tokens.Length; i++) tokens[i] = new IndexTts2TextSegmenter.Token(pieces[i].Piece, pieces[i].Id);
        return IndexTts2TextSegmenter.Split(tokens, maxTokensPerSegment, quickStreamingTokens);
    }

    /// <summary>The string <c>TextTokenizer.encode</c> hands to SentencePiece, or null for empty text. A text that is
    /// one character after stripping is encoded raw, without normalization or uppercasing.</summary>
    internal string? PrepareSentencePiece(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (text.Length == 0) return null;
        if (CodePointCount(text.Trim()) == 1) return text;
        return IndexTtsTextNormalizer.InjectCjkBoundaries(Normalizer.Normalize(text));
    }

    /// <summary>The 2.5 text stages of <c>infer_generator</c>, in order.</summary>
    internal TiktokenStages PrepareTiktoken(string text, string language)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(language);
        string lang = language.ToLowerInvariant();
        string cleaned = IndexTtsFrontNormalizer.ReplaceChars(text, chinese: false);
        string? normalized = lang is "zh" or "zhen" or "en" ? Normalizer.Normalize(cleaned) : null;
        string current = normalized ?? cleaned;
        if (lang is "ja" or "zh" or "zhen" or "en") current = current.ToLowerInvariant();
        if (lang == "es") current = current.ToUpperInvariant();
        string lowered = current;
        string annotated = ApplyPronunciationAnnotations(lowered);
        string prepared = SpecialTokenPattern().Replace(annotated, static m => "<|" + m.Groups[1].Value.ToUpperInvariant() + "|>");
        return new TiktokenStages(cleaned, normalized, lowered, annotated, prepared);
    }

    /// <summary>Upstream <c>lang_prefix</c>: <c>"&lt;|en|&gt; "</c>.</summary>
    public static string LanguagePrefix(string language)
    {
        ArgumentNullException.ThrowIfNull(language);
        return "<|" + language.ToLowerInvariant() + "|> ";
    }

    /// <summary><c>encode(lang_prefix + segment)</c> followed by the stop pad, exactly the row upstream builds.</summary>
    internal static int[] EncodeTiktokenSegment(string segment, string prefix, Func<string, int[]> encode)
    {
        int[] ids = encode(prefix + segment);
        int[] padded = new int[ids.Length + 1];
        ids.CopyTo(padded, 0);
        padded[^1] = StopTextToken;
        return padded;
    }

    /// <summary><c>prepare_gpt_inputs</c>' <c>valid_mask</c>: drops every start (0) and stop (1) text token. In the 2.5
    /// tiktoken vocabulary these ids are also <c>!</c> and <c>"</c>, which upstream therefore never feeds the GPT.</summary>
    public static int[] StripTextSentinels(int[] ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        int kept = 0;
        foreach (int id in ids) if (id != StartTextToken && id != StopTextToken) kept++;
        if (kept == ids.Length) return ids;
        int[] result = new int[kept];
        int j = 0;
        foreach (int id in ids) if (id != StartTextToken && id != StopTextToken) result[j++] = id;
        return result;
    }

    /// <summary>Upstream <c>apply_pronunciation_annotations</c>: <c>&lt;word|PRON&gt;</c> becomes
    /// <c>&lt;|SPECIAL_TOKEN_1|&gt;PRON&lt;|SPECIAL_TOKEN_1|&gt;</c> (<c>_2</c> when the word has Chinese characters); a kana
    /// reading is inlined between spaces instead.</summary>
    internal static string ApplyPronunciationAnnotations(string text) =>
        IndexTtsFrontNormalizer.PronunciationAnnotationPattern().Replace(text, static m =>
        {
            string pronunciation = m.Groups[2].Value.ToUpperInvariant();
            if (HiraganaPattern().IsMatch(pronunciation) || KatakanaPattern().IsMatch(pronunciation)) return " " + pronunciation + " ";
            string token = ChineseChar().IsMatch(m.Groups[1].Value) ? "SPECIAL_TOKEN_2" : "SPECIAL_TOKEN_1";
            return "<|" + token + "|>" + pronunciation + "<|" + token + "|>";
        });

    /// <summary>Upstream <c>split_text_by_tokens</c>: whole text if it fits the budget, otherwise punctuation-bounded
    /// chunks (pronunciation-token spans kept whole, over-long chunks cut per character) greedily packed.</summary>
    internal static List<string> SplitByTokens(string text, int maxTokens, int capacity, string prefix, Func<string, int> tokenLength)
    {
        int budget = Math.Max(1, Math.Min(maxTokens, capacity - 2) - tokenLength(prefix));
        if (tokenLength(text) <= budget) return [text];

        List<string> chunks = [];
        int pos = 0;
        foreach (Match m in ProtectedSpanPattern().Matches(text))
        {
            if (m.Index > pos) AddChunks(text[pos..m.Index], budget, tokenLength, chunks);
            chunks.Add(m.Value);
            pos = m.Index + m.Length;
        }
        if (pos < text.Length) AddChunks(text[pos..], budget, tokenLength, chunks);

        List<string> segments = [];
        string current = "";
        foreach (string chunk in chunks)
        {
            if (current.Length > 0 && tokenLength(current + chunk) > budget)
            {
                segments.Add(current);
                current = chunk;
            }
            else
            {
                current += chunk;
            }
        }
        if (current.Length > 0) segments.Add(current);
        return segments.Count > 0 ? segments : [text];
    }

    private static void AddChunks(string piece, int budget, Func<string, int> tokenLength, List<string> chunks)
    {
        foreach (string part in AfterPunctuationPattern().Split(piece))
        {
            if (part.Length == 0) continue;
            if (tokenLength(part) <= budget)
            {
                chunks.Add(part);
                continue;
            }
            string current = "";
            for (int i = 0; i < part.Length;)
            {
                int len = char.IsSurrogatePair(part, i) ? 2 : 1;
                string ch = part.Substring(i, len);
                i += len;
                if (current.Length > 0 && tokenLength(current + ch) > budget)
                {
                    chunks.Add(current);
                    current = ch;
                }
                else
                {
                    current += ch;
                }
            }
            if (current.Length > 0) chunks.Add(current);
        }
    }

    private static int CodePointCount(string s)
    {
        int count = 0;
        for (int i = 0; i < s.Length; i += char.IsSurrogatePair(s, i) ? 2 : 1) count++;
        return count;
    }

    /// <summary>The 2.5 text after each <c>infer_generator</c> step.</summary>
    internal readonly record struct TiktokenStages(string Cleaned, string? Normalized, string Lowered, string Annotated, string Prepared);

    [GeneratedRegex(@"<\|([^|]+)\|>")]
    private static partial Regex SpecialTokenPattern();

    [GeneratedRegex(@"<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>")]
    private static partial Regex ProtectedSpanPattern();

    [GeneratedRegex(@"(?<=[，。！？、；：,\.!\?;:\n])")]
    private static partial Regex AfterPunctuationPattern();

    [GeneratedRegex(@"^[぀-ゟ]+$")]
    private static partial Regex HiraganaPattern();

    [GeneratedRegex(@"^[゠-ヿ]+$")]
    private static partial Regex KatakanaPattern();

    [GeneratedRegex(@"[一-鿿]")]
    private static partial Regex ChineseChar();
}
