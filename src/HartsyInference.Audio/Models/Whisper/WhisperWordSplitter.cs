using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Audio.Models.Whisper;

/// <summary>Groups Whisper text tokens into words by OpenAI's rules: a word starts at a token with a leading space or at a
/// stray punctuation mark, a script written without spaces gets one word per character, and attached punctuation joins its
/// neighbour without taking part in the word's timing.</summary>
internal static class WhisperWordSplitter
{
    private const char Replacement = (char)0xFFFD;
    private const string Prepended = "\"'“¡¿([{-";
    private const string Appended = "\"'.。,，!！?？:：”)]}、";
    private const string AsciiPunctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

    /// <summary>Languages whose words are not separated by spaces.</summary>
    private static readonly HashSet<string> Unspaced = new HashSet<string>(StringComparer.Ordinal) { "zh", "ja", "th", "lo", "my", "yue" };

    /// <summary>Known text ready to tokenize: every whitespace run becomes one space and the ends are trimmed, so a line
    /// break cannot fuse two words. Null when only whitespace is left.</summary>
    internal static string? NormalizeKnownText(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return null;
        }
        return string.Join(' ', text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
    }

    /// <summary>Splits <paramref name="tokens"/>, the text tokens of one transcript with no special tokens among them, into
    /// words; <paramref name="decode"/> turns a run of token ids into text. <paramref name="language"/> picks the space-free
    /// rule and may be a locale code such as <c>zh-CN</c>; with none, the text's own script decides.</summary>
    internal static IReadOnlyList<WhisperWordTokens> Split(Func<ReadOnlyMemory<int>, string> decode, ReadOnlyMemory<int> tokens, string? language)
    {
        List<WhisperWordTokens> pieces = SplitOnUnicode(decode, tokens);
        bool unspaced = string.IsNullOrWhiteSpace(language)
            ? MostlyUnspacedScript(pieces)
            : Unspaced.Contains(WhisperTokenizer.NormalizeLanguageCode(language));
        if (unspaced)
        {
            return MergePunctuation(pieces);
        }

        List<WhisperWordTokens> words = new List<WhisperWordTokens>();
        foreach (WhisperWordTokens piece in pieces)
        {
            if (words.Count == 0 || piece.Text.StartsWith(' ') || AsciiPunctuation.Contains(piece.Text.Trim()))
            {
                words.Add(piece);
            }
            else
            {
                WhisperWordTokens last = words[^1];
                words[^1] = new WhisperWordTokens(last.Text + piece.Text, last.TokenCount + piece.TokenCount);
            }
        }
        return MergePunctuation(words);
    }

    /// <summary>Cuts the tokens into runs that each decode to whole characters: a multi-byte character can span several
    /// tokens and decodes to a replacement mark until the run is complete. A replacement mark the whole text also has is
    /// invalid data, not an unfinished character, and does not hold the run open.</summary>
    private static List<WhisperWordTokens> SplitOnUnicode(Func<ReadOnlyMemory<int>, string> decode, ReadOnlyMemory<int> tokens)
    {
        List<WhisperWordTokens> pieces = new List<WhisperWordTokens>();
        string whole = decode(tokens);
        int start = 0;
        int offset = 0;
        for (int end = 1; end <= tokens.Length; end++)
        {
            string decoded = decode(tokens[start..end]);
            int mark = decoded.IndexOf(Replacement);
            bool complete = mark < 0 || end == tokens.Length || (offset + mark < whole.Length && whole[offset + mark] == Replacement);
            if (complete)
            {
                pieces.Add(new WhisperWordTokens(decoded, end - start));
                offset += decoded.Length;
                start = end;
            }
        }
        return pieces;
    }

    /// <summary>Whether most letters of the text belong to a script written without spaces, for a transcript with no
    /// language to say so.</summary>
    private static bool MostlyUnspacedScript(List<WhisperWordTokens> pieces)
    {
        int letters = 0;
        int unspaced = 0;
        foreach (WhisperWordTokens piece in pieces)
        {
            foreach (char c in piece.Text)
            {
                if (!char.IsLetter(c))
                {
                    continue;
                }
                letters++;
                if (c is (>= '฀' and <= '໿') or (>= 'က' and <= '႟') or (>= '぀' and <= 'ヿ')
                    or (>= '㐀' and <= '䶿') or (>= '一' and <= '鿿'))
                {
                    unspaced++;
                }
            }
        }
        return letters > 0 && unspaced * 2 > letters;
    }

    /// <summary>Joins an opening mark to the word after it and a closing mark to the word before it, then drops what was
    /// absorbed. Only the text and token count grow: the word keeps the timing of its own tokens, not the mark's.</summary>
    private static List<WhisperWordTokens> MergePunctuation(List<WhisperWordTokens> words)
    {
        string[] text = new string[words.Count];
        int[] counts = new int[words.Count];
        int[] leads = new int[words.Count];
        int[] trails = new int[words.Count];
        for (int k = 0; k < words.Count; k++)
        {
            text[k] = words[k].Text;
            counts[k] = words[k].TokenCount;
        }

        int following = words.Count - 1;
        for (int previous = words.Count - 2; previous >= 0; previous--)
        {
            if (text[previous].StartsWith(' ') && Prepended.Contains(text[previous].Trim()))
            {
                text[following] = text[previous] + text[following];
                leads[following] += counts[previous];
                counts[following] += counts[previous];
                text[previous] = string.Empty;
                counts[previous] = 0;
            }
            else
            {
                following = previous;
            }
        }

        int last = 0;
        for (int next = 1; next < words.Count; next++)
        {
            if (!text[last].EndsWith(' ') && Appended.Contains(text[next]))
            {
                text[last] += text[next];
                trails[last] += counts[next];
                counts[last] += counts[next];
                text[next] = string.Empty;
                counts[next] = 0;
            }
            else
            {
                last = next;
            }
        }

        List<WhisperWordTokens> merged = new List<WhisperWordTokens>(words.Count);
        for (int k = 0; k < words.Count; k++)
        {
            if (counts[k] > 0)
            {
                merged.Add(new WhisperWordTokens(text[k], counts[k], leads[k], trails[k]));
            }
        }
        return merged;
    }
}
