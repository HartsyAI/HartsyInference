namespace HartsyInference.LLM.OutputParsing;

/// <summary>Finds stop markers in a growing text buffer and reports how much of its tail must be held back because it could still grow into a marker.</summary>
internal static class MarkerScanner
{
    /// <summary>Scans <paramref name="text"/>: the earliest full marker (longest on a tie) and the earliest suffix start that is a proper prefix of some marker. Each index is -1 when absent.</summary>
    public static void Scan(string text, string[] markers, out int matchAt, out int markerIndex, out int holdFrom)
    {
        matchAt = -1;
        markerIndex = -1;
        for (int m = 0; m < markers.Length; m++)
        {
            int at = text.IndexOf(markers[m], StringComparison.Ordinal);
            if (at < 0) continue;
            if (matchAt < 0 || at < matchAt || (at == matchAt && markers[m].Length > markers[markerIndex].Length))
            {
                matchAt = at;
                markerIndex = m;
            }
        }
        holdFrom = FindHoldFrom(text, markers);
    }

    private static int FindHoldFrom(string text, string[] markers)
    {
        int longest = 0;
        foreach (string marker in markers) longest = Math.Max(longest, marker.Length);
        for (int start = Math.Max(0, text.Length - longest + 1); start < text.Length; start++)
        {
            ReadOnlySpan<char> suffix = text.AsSpan(start);
            foreach (string marker in markers)
            {
                if (marker.Length > suffix.Length && marker.AsSpan().StartsWith(suffix, StringComparison.Ordinal)) return start;
            }
        }
        return -1;
    }
}
