using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Long-text chunking for Drama 3, ported from fish-speech's <c>fish_omni/chunking.py</c> (the hosted
/// server's rules). Text is cut into chunks of about <see cref="ChunkBytes"/> UTF-8 bytes with at most
/// <see cref="ChunkMaxSpeakers"/> speakers, preferring sentence ends, then clause marks, then spaces; special tokens,
/// paired tags and <c>[bracket cues]</c> are never split.</summary>
public static partial class FishAudioDrama3Chunking
{
    /// <summary>Chunk size threshold in UTF-8 bytes; a protected span longer than this stays whole.</summary>
    public const int ChunkBytes = 300;

    /// <summary>Most speakers one chunk may carry.</summary>
    public const int ChunkMaxSpeakers = 3;

    /// <summary>Frames per chunk when the request sets no cap, about 27.9 s at 2048 samples per frame and 44.1 kHz.</summary>
    public const int ChunkMaxNewFrames = 600;

    private const string SentenceEnds = ".!?;。！？；…\n";
    private const string ClauseMarks = ",:，：、";
    private const string Spaces = " ";

    [GeneratedRegex(@"<\|[^|>]*\|>|<([A-Za-z][\w-]*)\b[^>]*>.*?</\1>|\[[^\[\]]*\]", RegexOptions.Singleline)]
    private static partial Regex ProtectedSpan();

    /// <summary>Generation chunks for <paramref name="text"/>, in order. Untagged text is split by length; tagged text is
    /// split into speaker segments (long ones cut further, each piece keeping its tag) and regrouped, starting a new
    /// chunk when a segment would bring in one speaker too many or exceed <paramref name="maxBytes"/>.</summary>
    public static List<string> Split(string text, int maxBytes = ChunkBytes, int maxSpeakers = ChunkMaxSpeakers)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (!FishAudioDrama3Text.SpeakerTag().IsMatch(text))
        {
            return SplitByLength(text, maxBytes);
        }
        List<(string Speaker, string Segment)> segments = [];
        foreach ((string speaker, string body) in ParseSpeakerSegments(text))
        {
            foreach (string piece in SplitByLength(body, maxBytes))
            {
                segments.Add((speaker, $"<|speaker:{speaker}|> {piece}"));
            }
        }
        List<string> chunks = [];
        List<string> batch = [];
        List<string> speakers = [];
        int size = 0;
        foreach ((string speaker, string segment) in segments)
        {
            bool newSpeaker = !speakers.Contains(speaker);
            bool tooManySpeakers = newSpeaker && speakers.Count >= maxSpeakers;
            bool tooLong = batch.Count > 0 && size + Utf8Length(segment) > maxBytes;
            if (tooManySpeakers || tooLong)
            {
                chunks.Add(string.Join("\n", batch));
                batch.Clear();
                speakers.Clear();
                size = 0;
                newSpeaker = true;
            }
            batch.Add(segment);
            if (newSpeaker)
            {
                speakers.Add(speaker);
            }
            size += Utf8Length(segment);
        }
        if (batch.Count > 0)
        {
            chunks.Add(string.Join("\n", batch));
        }
        return chunks;
    }

    /// <summary>Chunks of about <paramref name="length"/> bytes packed sentence by sentence. A sentence still longer
    /// than <paramref name="length"/> is cut at clause marks, then spaces, then on raw length; adjacent pieces are then
    /// merged back up to <paramref name="length"/>. Text short enough is returned unchanged.</summary>
    public static List<string> SplitByLength(string text, int length = ChunkBytes)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        if (FishAudioDrama3Text.IsBlank(text))
        {
            return [];
        }
        if (Utf8Length(text) <= length)
        {
            return [text];
        }
        List<(string Text, bool Protected)> pieces = BreakAtMarks(ProtectedPieces(text), 0, SentenceEnds);
        pieces = BreakAtMarks(pieces, length, ClauseMarks);
        pieces = BreakAtMarks(pieces, length, Spaces);
        pieces = BreakByLength(pieces, length);

        List<string> chunks = [];
        StringBuilder current = new();
        foreach ((string piece, bool _) in pieces)
        {
            if (current.Length > 0 && Utf8Length(current.ToString() + piece) > length)
            {
                chunks.Add(current.ToString());
                current.Clear();
            }
            current.Append(piece);
        }
        chunks.Add(current.ToString());
        List<string> result = [];
        foreach (string chunk in chunks)
        {
            string trimmed = FishAudioDrama3Text.Trim(chunk);
            if (trimmed.Length > 0)
            {
                result.Add(trimmed);
            }
        }
        return result;
    }

    /// <summary>(speaker id, text) per tagged segment; text before the first tag belongs to speaker 0, and blank
    /// segments are dropped.</summary>
    public static List<(string Speaker, string Text)> ParseSpeakerSegments(string text)
    {
        string[] parts = FishAudioDrama3Text.SpeakerTag().Split(text);
        List<(string, string)> segments = [];
        if (!FishAudioDrama3Text.IsBlank(parts[0]))
        {
            segments.Add(("0", parts[0]));
        }
        for (int i = 1; i + 1 < parts.Length; i += 2)
        {
            if (!FishAudioDrama3Text.IsBlank(parts[i + 1]))
            {
                segments.Add((parts[i], parts[i + 1]));
            }
        }
        return segments;
    }

    /// <summary>UTF-8 byte length of <paramref name="text"/>.</summary>
    public static int Utf8Length(string text) => Encoding.UTF8.GetByteCount(text);

    private static List<(string, bool)> ProtectedPieces(string text)
    {
        List<(string, bool)> pieces = [];
        int last = 0;
        foreach (Match match in ProtectedSpan().Matches(text))
        {
            if (match.Index > last)
            {
                pieces.Add((text[last..match.Index], false));
            }
            pieces.Add((match.Value, true));
            last = match.Index + match.Length;
        }
        if (last < text.Length)
        {
            pieces.Add((text[last..], false));
        }
        return pieces;
    }

    // Splits each free piece longer than length bytes after every mark, keeping the mark with the text before it.
    private static List<(string, bool)> BreakAtMarks(List<(string Text, bool Protected)> pieces, int length, string marks)
    {
        List<(string, bool)> output = [];
        foreach ((string text, bool isProtected) in pieces)
        {
            if (isProtected || Utf8Length(text) <= length)
            {
                output.Add((text, isProtected));
                continue;
            }
            int start = 0;
            for (int index = 0; index < text.Length; index++)
            {
                if (BreaksAt(text, index, marks))
                {
                    output.Add((text[start..(index + 1)], false));
                    start = index + 1;
                }
            }
            if (start < text.Length)
            {
                output.Add((text[start..], false));
            }
        }
        return output;
    }

    // A mark breaks the text unless it is the point of a decimal such as 3.14.
    private static bool BreaksAt(string text, int index, string marks)
    {
        char c = text[index];
        if (marks.IndexOf(c) < 0)
        {
            return false;
        }
        return !(c == '.' && index > 0 && index < text.Length - 1 && char.IsDigit(text[index - 1]) && char.IsDigit(text[index + 1]));
    }

    // Splits the free pieces still longer than length on code-point boundaries.
    private static List<(string, bool)> BreakByLength(List<(string Text, bool Protected)> pieces, int length)
    {
        List<(string, bool)> output = [];
        foreach ((string text, bool isProtected) in pieces)
        {
            if (isProtected || Utf8Length(text) <= length)
            {
                output.Add((text, isProtected));
                continue;
            }
            StringBuilder current = new();
            int size = 0;
            foreach (Rune rune in text.EnumerateRunes())
            {
                int runeBytes = rune.Utf8SequenceLength;
                if (current.Length > 0 && size + runeBytes > length)
                {
                    output.Add((current.ToString(), false));
                    current.Clear();
                    size = 0;
                }
                current.Append(rune.ToString());
                size += runeBytes;
            }
            output.Add((current.ToString(), false));
        }
        return output;
    }
}
