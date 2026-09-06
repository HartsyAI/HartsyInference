namespace HartsyInference.Engine.Audio.Wake;

/// <summary>Separates the wake phrase from the command that followed it.
///
/// <para>Transcription runs over the wake word and the command together — it has to, because the useful audio
/// starts before the detector fires and text-independent speaker verification degrades badly on a phrase that
/// short. So the transcript a satellite receives begins with the words that woke it: "Hey Jarvis, what time is
/// it?". Handing that to an assistant invites it to answer the greeting instead of the question, which small
/// models reliably do.</para>
///
/// <para>The engine already knows which head fired, so it can say which words to remove rather than leaving
/// every consumer to guess. The full transcript is still reported alongside — it is what was said, and a caller
/// doing its own parsing should not have it silently edited.</para></summary>
public static class WakePhrase
{
    /// <summary>Turns a head name into the words it listens for: <c>hey_jarvis</c> becomes <c>hey jarvis</c>.</summary>
    /// <remarks>Underscores and hyphens are the two separators the stock and in-engine-trained heads use.</remarks>
    public static string FromWordKey(string word) =>
        string.IsNullOrWhiteSpace(word) ? "" : word.Replace('_', ' ').Replace('-', ' ').Trim();

    /// <summary>Removes a leading wake phrase from a transcript, returning what the user actually asked.</summary>
    /// <param name="transcript">The transcript as recognised, wake phrase included.</param>
    /// <param name="word">The head name that fired, e.g. <c>hey_jarvis</c>.</param>
    /// <returns>The command, or the transcript unchanged when the phrase is not at the front of it. Empty when
    /// the transcript was nothing but the wake phrase.</returns>
    /// <remarks>Matching ignores case, punctuation and repeated whitespace, because a recogniser writes the
    /// same spoken words several ways — "Hey Jarvis,", "Hey, Jarvis!", "hey jarvis" — and a literal comparison
    /// would strip the phrase only some of the time, which is worse than never stripping it.
    ///
    /// <para>Only a leading match is removed. The same words later in a sentence ("tell me about hey jarvis")
    /// are part of the question.</para></remarks>
    public static string Strip(string? transcript, string? word)
    {
        if (string.IsNullOrWhiteSpace(transcript))
        {
            return "";
        }
        string phrase = FromWordKey(word ?? "");
        if (phrase.Length == 0)
        {
            return transcript.Trim();
        }

        string[] phraseWords = Tokenize(phrase);
        if (phraseWords.Length == 0)
        {
            return transcript.Trim();
        }

        // Walk the transcript one comparable word at a time, tracking where each ends, so the tail can be cut
        // out of the ORIGINAL text — preserving its capitalisation and punctuation rather than handing back a
        // flattened reconstruction.
        int matched = 0;
        int cutAt = -1;
        int i = 0;
        while (i < transcript.Length && matched < phraseWords.Length)
        {
            while (i < transcript.Length && !char.IsLetterOrDigit(transcript[i]))
            {
                i++;
            }
            int start = i;
            while (i < transcript.Length && char.IsLetterOrDigit(transcript[i]))
            {
                i++;
            }
            if (i == start)
            {
                break;
            }
            if (!transcript.AsSpan(start, i - start).Equals(phraseWords[matched], StringComparison.OrdinalIgnoreCase))
            {
                return transcript.Trim();
            }
            matched++;
            cutAt = i;
        }

        if (matched < phraseWords.Length || cutAt < 0)
        {
            return transcript.Trim();
        }
        return transcript[cutAt..].TrimStart(' ', '\t', ',', '.', '!', '?', ':', ';', '-', '—', '\r', '\n').Trim();
    }

    private static string[] Tokenize(string phrase) =>
        phrase.Split([' ', '\t', '_', '-'], StringSplitOptions.RemoveEmptyEntries);
}
