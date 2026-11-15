using System.Text.Json;

namespace HartsyInference.GenCapture;

/// <summary>Batched-decode proof read from an attention telemetry observation file.
/// Two paged admissions are not proof. A proof needs a <c>scheduler.decode.batch</c> event with <c>Count</c> of at least 2
/// inside the measured request window, from a session that dropped no observations.</summary>
internal sealed record AttentionDecodeProof(
    int DecodeEvents,
    int MeasuredDecodeEvents,
    int MaxCountMeasured,
    int StepsWithTwoOrMoreMeasured,
    IReadOnlyDictionary<int, int> MeasuredCountHistogram,
    int WarmupStepsWithTwoOrMore,
    int PagedAdmissionEvents,
    int PagedLayers,
    int PagedKvHeads,
    int PagedHeadDim)
{
    public bool Proven => StepsWithTwoOrMoreMeasured > 0;

    /// <summary>Reads <paramref name="path"/>. Events at or after <paramref name="measuredStart"/> belong to the measured request;
    /// the caller must confirm that the session dropped nothing, because indices are only valid for an undropped capture.</summary>
    public static AttentionDecodeProof Read(string path, long measuredStart)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(path));
        int index = 0, decode = 0, measuredDecode = 0, maxMeasured = 0, twoOrMore = 0, warmupTwoOrMore = 0, paged = 0;
        int layers = 0, kvHeads = 0, headDim = 0;
        HashSet<int> pagedLayers = [];
        SortedDictionary<int, int> histogram = new();
        foreach (JsonElement element in document.RootElement.EnumerateArray())
        {
            string site = element.TryGetProperty("Site", out JsonElement siteValue) ? siteValue.GetString() ?? "" : "";
            bool measured = index >= measuredStart;
            if (site == "scheduler.decode.batch")
            {
                int count = element.GetProperty("Count").GetInt32();
                decode++;
                if (measured)
                {
                    measuredDecode++;
                    maxMeasured = Math.Max(maxMeasured, count);
                    histogram[count] = histogram.GetValueOrDefault(count) + 1;
                    if (count >= 2) twoOrMore++;
                }
                else if (count >= 2) warmupTwoOrMore++;
            }
            else if (site == "scheduler.paged") paged++;
            else if (site == "paged.keys")
            {
                int layer = element.GetProperty("Layer").GetInt32();
                if (layer >= 0) pagedLayers.Add(layer);
                kvHeads = element.GetProperty("KvHeads").GetInt32();
                headDim = element.GetProperty("QkHeadDim").GetInt32();
            }
            index++;
        }
        layers = pagedLayers.Count;
        return new AttentionDecodeProof(decode, measuredDecode, maxMeasured, twoOrMore,
            histogram, warmupTwoOrMore, paged, layers, kvHeads, headDim);
    }
}
