using System.Text;

namespace HartsyInference.Audio.Frontends;

/// <summary>Cuts a passage into sentences so a text-to-speech model can be given one at a time.
///
/// <para>Why a caller would want that: synthesis of a whole reply cannot start until the whole reply exists,
/// and cannot finish until every word of it is generated. Split it, and the first sentence can be spoken while
/// the rest is still being written — the listener hears something after one sentence's worth of work instead of
/// the entire passage's. For a model with no incremental decode loop of its own, like Piper, this is the only
/// way to stream at all.</para>
///
/// <para>The split is deliberately conservative. Cutting where a sentence does not end is the expensive
/// mistake: the two halves are synthesized with separate prosody and the seam is audible, whereas failing to
/// cut only costs latency. So a period is a boundary only when what precedes it does not look like an
/// abbreviation or an initial, and only when what follows starts a new sentence.</para></summary>
public static class SentenceSplitter
{
    /// <summary>Words that end in a period without ending a sentence. Lower-cased; the match ignores case.
    ///
    /// <para>Deliberately short. Every entry here is a period that will not be cut on, so a wrong entry costs
    /// nothing but a longer first chunk, while a missing entry costs an audible seam mid-sentence. The list
    /// covers titles, the calendar, and the handful of Latin abbreviations that show up in ordinary prose.</para></summary>
    private static readonly HashSet<string> Abbreviations = new(StringComparer.OrdinalIgnoreCase)
    {
        "mr", "mrs", "ms", "dr", "prof", "rev", "hon", "st", "sr", "jr", "mt",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
        "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun",
        "no", "vs", "etc", "eg", "ie", "al", "approx", "dept", "est", "fig", "inc", "ltd", "min", "max",
        "am", "pm", "ca", "cf",
    };

    /// <summary>Shortest sentence worth emitting on its own, in characters.
    ///
    /// <para>A fragment shorter than this is glued to the sentence after it rather than synthesized alone.
    /// "Yes." on its own is a quarter second of audio wrapped in a whole model invocation, and a run of them
    /// turns a smooth reply into a stutter of separately-voiced words.</para></summary>
    public const int MinSentenceLength = 24;

    /// <summary>Splits text into sentences, longest-safe-first.</summary>
    /// <param name="text">The passage to split. Null, empty or whitespace yields nothing.</param>
    /// <param name="minLength">Shortest sentence to emit alone; shorter ones merge forward. Defaults to
    /// <see cref="MinSentenceLength"/>.</param>
    /// <returns>The sentences, in order, each trimmed, with their terminating punctuation kept. Concatenating
    /// them with a single space reproduces the input's words in order.</returns>
    /// <remarks>The last piece is returned even when it is short and even when it has no terminator, because a
    /// caller streaming a reply has to speak the tail; it is the only piece allowed to be under
    /// <paramref name="minLength"/>.</remarks>
    public static IReadOnlyList<string> Split(string? text, int minLength = MinSentenceLength)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return [];
        }

        List<string> sentences = [];
        StringBuilder current = new();
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            current.Append(c);
            if (!IsTerminator(c) || !EndsSentence(text, i))
            {
                continue;
            }
            // Keep a closing quote or bracket with the sentence it closes.
            while (i + 1 < text.Length && IsTrailingPunctuation(text[i + 1]))
            {
                current.Append(text[++i]);
            }
            string candidate = current.ToString().Trim();
            if (candidate.Length == 0)
            {
                current.Clear();
                continue;
            }
            // Too short to stand alone: leave it in the builder and let the next sentence carry it.
            if (candidate.Length < minLength)
            {
                continue;
            }
            sentences.Add(candidate);
            current.Clear();
        }

        string tail = current.ToString().Trim();
        if (tail.Length > 0)
        {
            sentences.Add(tail);
        }
        return sentences;
    }

    /// <summary>The three characters that can end a sentence. A colon or semicolon can too, in principle, but
    /// cutting there changes the intonation of what follows, which is exactly the seam this avoids.</summary>
    private static bool IsTerminator(char c) => c is '.' or '!' or '?';

    private static bool IsTrailingPunctuation(char c) => c is '"' or '\'' or ')' or ']' or '”' or '’' or '.' or '!' or '?';

    /// <summary>Whether the terminator at <paramref name="index"/> really ends a sentence.</summary>
    /// <remarks>Three ways it does not: it is the last character of an abbreviation, it is the dot of an
    /// initial or a decimal number, or nothing that looks like a new sentence follows it. Exclamation and
    /// question marks are unambiguous and only need the last of those.</remarks>
    private static bool EndsSentence(string text, int index)
    {
        if (text[index] == '.')
        {
            if (IsAbbreviation(text, index) || IsNumericDot(text, index))
            {
                return false;
            }
        }
        return StartsNewSentence(text, index);
    }

    /// <summary>True when the word ending at this period is a known abbreviation, or a single letter — an
    /// initial, as in "J. R. R." — which no sentence ever legitimately ends with.</summary>
    private static bool IsAbbreviation(string text, int index)
    {
        int start = index;
        while (start > 0 && char.IsLetter(text[start - 1]))
        {
            start--;
        }
        int length = index - start;
        if (length == 0)
        {
            return false;
        }
        if (length == 1)
        {
            return true;
        }
        return Abbreviations.Contains(text.AsSpan(start, length).ToString());
    }

    /// <summary>True for the dot inside a number — "3.5", "1.000" — which is a decimal point, not a stop.</summary>
    private static bool IsNumericDot(string text, int index) =>
        index > 0 && char.IsDigit(text[index - 1]) && index + 1 < text.Length && char.IsDigit(text[index + 1]);

    /// <summary>True when what follows the terminator looks like the start of another sentence: whitespace and
    /// then an upper-case letter, a digit, or an opening quote. End of input counts.</summary>
    private static bool StartsNewSentence(string text, int index)
    {
        int i = index + 1;
        while (i < text.Length && IsTrailingPunctuation(text[i]))
        {
            i++;
        }
        if (i >= text.Length)
        {
            return true;
        }
        if (!char.IsWhiteSpace(text[i]))
        {
            return false;
        }
        while (i < text.Length && char.IsWhiteSpace(text[i]))
        {
            i++;
        }
        if (i >= text.Length)
        {
            return true;
        }
        char next = text[i];
        return char.IsUpper(next) || char.IsDigit(next) || next is '"' or '\'' or '“' or '‘' or '(' or '[';
    }
}
