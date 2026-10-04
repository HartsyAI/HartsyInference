namespace HartsyInference.Audio.Models.Kokoro;

/// <summary>Splits a Kokoro phoneme string into pieces the model can take whole: PLBERT has 512 positions, two of
/// them the BOS/EOS pads, so the reference <c>KPipeline</c> caps a chunk at 510 phonemes. Mirrors its
/// <c>waterfall_last</c>: cut after the last sentence end (<c>!.?…</c>), else the last <c>:;</c>, else the last
/// <c>,—</c>, stepping past a closing <c>)</c> or <c>”</c>; with none of those, the last space, and only for one
/// unbroken run longer than the cap, a hard cut. A newline is always a boundary: the reference phonemizes and
/// synthesizes each input line separately.</summary>
public static class KokoroPhonemeChunker
{
    /// <summary>Longest phoneme string synthesized whole, as in the reference pipeline.</summary>
    public const int MaxPhonemes = 510;

    private static readonly string[] Waterfall = ["!.?…", ":;", ",—"];
    private const string Bumps = ")”";

    /// <summary>The non-empty, trimmed chunks of <paramref name="phonemes"/>, each at most
    /// <paramref name="maxLength"/> characters, in order.</summary>
    public static IReadOnlyList<string> Split(string phonemes, int maxLength = MaxPhonemes)
    {
        ArgumentNullException.ThrowIfNull(phonemes);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxLength, 1);
        List<string> chunks = new();
        foreach (string line in phonemes.Split('\n'))
        {
            SplitLine(line, maxLength, chunks);
        }
        return chunks;
    }

    private static void SplitLine(string line, int maxLength, List<string> chunks)
    {
        int start = SkipSpaces(line, 0);
        int end = line.Length;
        while (end > start && char.IsWhiteSpace(line[end - 1])) end--;
        while (end - start > maxLength)
        {
            int cut = FindCut(line, start, start + maxLength);
            string chunk = line[start..cut].TrimEnd(' ');
            if (chunk.Length > 0) chunks.Add(chunk);
            start = SkipSpaces(line, cut);
        }
        if (end > start) chunks.Add(line[start..end]);
    }

    /// <summary>Exclusive cut index in <c>(start, limit]</c> for the chunk beginning at <paramref name="start"/>.</summary>
    private static int FindCut(string s, int start, int limit)
    {
        foreach (string marks in Waterfall)
        {
            for (int i = limit - 1; i >= start; i--)
            {
                if (marks.IndexOf(s[i]) < 0) continue;
                int cut = i + 1;
                if (cut < limit && Bumps.IndexOf(s[cut]) >= 0) cut++;
                return cut;
            }
        }
        // The space itself may sit at limit: the chunk before it is then exactly the cap.
        for (int i = Math.Min(limit, s.Length - 1); i > start; i--)
        {
            if (s[i] == ' ') return i;
        }
        return limit;
    }

    private static int SkipSpaces(string s, int i)
    {
        while (i < s.Length && char.IsWhiteSpace(s[i])) i++;
        return i;
    }
}
