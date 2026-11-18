namespace HartsyInference.Audio.Models.Whisper;

/// <summary>Groups a run of Whisper text tokens into words, following OpenAI's rules: a word starts at a token that begins with a space (or at a stray punctuation mark), a script written without spaces gets one word per character, and attached punctuation joins its neighbour.</summary>
internal static class WhisperWordSplitter
{
    private const char Replacement = (char)0xFFFD;
    private const string Prepended = "\"'“¡¿([{-";
    private const string Appended = "\"'.。,，!！?？:：”)]}、";

    /// <summary>Languages whose words are not separated by spaces.</summary>
    private static readonly HashSet<string> Unspaced = new HashSet<string>(StringComparer.Ordinal) { "zh", "ja", "th", "lo", "my", "yue" };

    /// <summary>Known text ready to tokenize: every run of whitespace (newlines, tabs, repeated spaces) becomes one space and the ends are trimmed, so a line break cannot fuse the last word of a line to the first of the next. Null when nothing but whitespace is left.</summary>
    internal static string? NormalizeKnownText(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return null;
        }
        return string.Join(' ', text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
    }

    /// <summary>Splits <paramref name="tokens"/>, the text tokens of one transcript with no special tokens among them, into words; <paramref name="decode"/> turns a run of token ids into text. <paramref name="language"/> picks the space-free rule; null counts as spaced.</summary>
    internal static IReadOnlyList<WhisperWordTokens> Split(Func<ReadOnlyMemory<int>, string> decode, ReadOnlyMemory<int> tokens, string? language)
    {
        List<WhisperWordTokens> pieces = SplitOnUnicode(decode, tokens);
        if (language is not null && Unspaced.Contains(language.ToLowerInvariant()))
        {
            return MergePunctuation(pieces);
        }

        List<WhisperWordTokens> words = new List<WhisperWordTokens>();
        foreach (WhisperWordTokens piece in pieces)
        {
            bool startsWord = words.Count == 0 || piece.Text.StartsWith(' ') || IsPunctuation(piece.Text.Trim());
            if (startsWord)
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

    /// <summary>Cuts the tokens into runs that each decode to whole characters, since a multi-byte character can span several tokens and decodes to a replacement mark until the run is complete.</summary>
    private static List<WhisperWordTokens> SplitOnUnicode(Func<ReadOnlyMemory<int>, string> decode, ReadOnlyMemory<int> tokens)
    {
        List<WhisperWordTokens> pieces = new List<WhisperWordTokens>();
        int start = 0;
        for (int end = 1; end <= tokens.Length; end++)
        {
            string decoded = decode(tokens[start..end]);
            if (!decoded.Contains(Replacement) || end == tokens.Length)
            {
                pieces.Add(new WhisperWordTokens(decoded, end - start));
                start = end;
            }
        }
        return pieces;
    }

    /// <summary>Joins an opening mark to the word after it and a closing mark to the word before it, then drops what was absorbed.</summary>
    private static List<WhisperWordTokens> MergePunctuation(List<WhisperWordTokens> words)
    {
        string[] text = new string[words.Count];
        int[] counts = new int[words.Count];
        for (int k = 0; k < words.Count; k++)
        {
            text[k] = words[k].Text;
            counts[k] = words[k].TokenCount;
        }

        int following = words.Count - 1;
        for (int previous = words.Count - 2; previous >= 0; previous--)
        {
            if (text[previous].StartsWith(' ') && text[previous].Trim().Length > 0 && Prepended.Contains(text[previous].Trim()))
            {
                text[following] = text[previous] + text[following];
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
            if (!text[last].EndsWith(' ') && text[next].Length > 0 && Appended.Contains(text[next]))
            {
                text[last] += text[next];
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
                merged.Add(new WhisperWordTokens(text[k], counts[k]));
            }
        }
        return merged;
    }

    /// <summary>Whether the text is one or more ASCII punctuation marks.</summary>
    private static bool IsPunctuation(string text)
    {
        if (text.Length == 0)
        {
            return false;
        }
        foreach (char c in text)
        {
            if (!char.IsAscii(c) || (!char.IsPunctuation(c) && !char.IsSymbol(c)))
            {
                return false;
            }
        }
        return true;
    }
}
