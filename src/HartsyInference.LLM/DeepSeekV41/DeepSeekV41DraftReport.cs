using System.Text;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>What the checkpoint's draft stack holds, with the routed experts a partial one lacks.</summary>
/// <param name="Status">Absent, complete or incomplete.</param>
/// <param name="MissingExpertsByLayer">For an incomplete draft: draft layer index (0-based within <c>mtp</c>) to its missing or partial expert ids, ascending.</param>
/// <param name="ExpertsPerLayer">Routed experts each draft layer should carry.</param>
public sealed record DeepSeekV41DraftReport(
    DeepSeekV41DraftStatus Status,
    IReadOnlyDictionary<int, IReadOnlyList<int>> MissingExpertsByLayer,
    int ExpertsPerLayer)
{
    /// <summary>A one-line refusal naming each incomplete draft layer and its missing experts, or an empty string when nothing blocks the draft.</summary>
    public string DescribeMissing()
    {
        if (Status != DeepSeekV41DraftStatus.Incomplete)
            return "";
        StringBuilder text = new();
        foreach (KeyValuePair<int, IReadOnlyList<int>> layer in MissingExpertsByLayer.OrderBy(static pair => pair.Key))
        {
            if (text.Length > 0)
                text.Append("; ");
            text.Append($"mtp.{layer.Key} lacks {layer.Value.Count} of {ExpertsPerLayer} routed experts ({FormatRanges(layer.Value)})");
        }
        return text.ToString();
    }

    private static string FormatRanges(IReadOnlyList<int> sorted)
    {
        List<string> parts = new();
        int index = 0;
        while (index < sorted.Count)
        {
            int start = sorted[index];
            int end = start;
            while (index + 1 < sorted.Count && sorted[index + 1] == end + 1)
            {
                end = sorted[++index];
            }
            parts.Add(start == end ? $"{start}" : $"{start}-{end}");
            index++;
        }
        return string.Join(", ", parts);
    }
}
