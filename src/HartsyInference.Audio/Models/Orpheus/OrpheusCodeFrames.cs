namespace HartsyInference.Audio.Models.Orpheus;

/// <summary>Pure index bookkeeping that turns Orpheus's flat emitted-token stream into the SNAC 3-layer
/// hierarchy — the one part of Orpheus that must be exactly right, so it lives apart from the pipeline and
/// is unit-tested without weights (the same split as <c>MusicGenDelay</c>).</summary>
public static class OrpheusCodeFrames
{
    /// <summary>Keeps tokens after the last <paramref name="codeStart"/> occurrence, drops
    /// <paramref name="endOfSpeech"/> markers, and trims to a whole number of <paramref name="tokensPerFrame"/>
    /// groups.</summary>
    public static int[] ExtractAudioCodes(IReadOnlyList<int> generated, int codeStart, int endOfSpeech,
        int tokensPerFrame)
    {
        int start = -1;
        for (int i = generated.Count - 1; i >= 0; i--)
            if (generated[i] == codeStart) { start = i; break; }
        int from = start >= 0 ? start + 1 : 0;

        List<int> row = new(generated.Count - from);
        for (int i = from; i < generated.Count; i++)
            if (generated[i] != endOfSpeech) row.Add(generated[i]);

        int n = (row.Count / tokensPerFrame) * tokensPerFrame;
        return row.GetRange(0, n).ToArray();
    }

    /// <summary>Redistributes 7-token groups into the SNAC hierarchy — position 0 → L1, {1,4} → L2,
    /// {2,3,5,6} → L3 — subtracting the per-position offset <c>audioCodeBase + p*codebookSize</c> and
    /// clamping into <c>[0, codebookSize)</c>. Returns three streams of length <c>groups</c>, <c>2*groups</c>,
    /// <c>4*groups</c>.</summary>
    public static (int[] L1, int[] L2, int[] L3) Redistribute(int[] codes, int audioCodeBase, int codebookSize)
    {
        int groups = codes.Length / 7;
        int[] l1 = new int[groups];
        int[] l2 = new int[groups * 2];
        int[] l3 = new int[groups * 4];
        for (int i = 0; i < groups; i++)
        {
            int b = i * 7;
            l1[i] = Dequant(codes[b], 0, audioCodeBase, codebookSize);
            l2[i * 2] = Dequant(codes[b + 1], 1, audioCodeBase, codebookSize);
            l3[i * 4] = Dequant(codes[b + 2], 2, audioCodeBase, codebookSize);
            l3[i * 4 + 1] = Dequant(codes[b + 3], 3, audioCodeBase, codebookSize);
            l2[i * 2 + 1] = Dequant(codes[b + 4], 4, audioCodeBase, codebookSize);
            l3[i * 4 + 2] = Dequant(codes[b + 5], 5, audioCodeBase, codebookSize);
            l3[i * 4 + 3] = Dequant(codes[b + 6], 6, audioCodeBase, codebookSize);
        }
        return (l1, l2, l3);
    }

    private static int Dequant(int token, int position, int audioCodeBase, int codebookSize)
    {
        int code = token - audioCodeBase - position * codebookSize;
        return Math.Clamp(code, 0, codebookSize - 1);
    }
}
