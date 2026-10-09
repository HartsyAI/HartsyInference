namespace HartsyInference.LLM.Generation.Speculation;

/// <summary>Materializes the searchable context shared by the draft providers:
/// the last <paramref name="maxLookback"/> tokens of prompt plus generated output.</summary>
internal static class SpeculationWindow
{
    internal static int[] Tail(int[] promptIds, IReadOnlyList<int> generated, int maxLookback)
    {
        int totalLen = promptIds.Length + generated.Count;
        int searchFloor = Math.Max(0, totalLen - maxLookback);
        int windowLen = totalLen - searchFloor;
        int[] context = new int[windowLen];
        for (int i = 0; i < windowLen; i++)
        {
            int idx = searchFloor + i;
            context[i] = idx < promptIds.Length ? promptIds[idx] : generated[idx - promptIds.Length];
        }
        return context;
    }
}
