using System.Text.Json;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The real tensor headers of the 48 official shards (deepseek-ai/DeepSeek-V4.1-Flash @ dba1be0a), folded into
/// templates: one entry per distinct (name pattern, dtype, shape) with the index ranges it repeats over.</summary>
/// <remarks>Read from the shard headers themselves, so the sums below are the real checkpoint's, not a model of it.
/// 96,085 names expand from 12 KB of templates.</remarks>
internal static class DeepSeekV41HeaderTemplates
{
    /// <summary>Index <c>metadata.total_size</c> of the official checkpoint.</summary>
    public const long OfficialTotalSize = 510_286_023_000;

    /// <summary>Tensor count over the official shards.</summary>
    public const int OfficialTensorCount = 96_085;

    /// <summary>Every tensor of the official checkpoint as (name, byte length).</summary>
    public static List<KeyValuePair<string, long>> ExpandOfficial()
    {
        using JsonDocument document = JsonDocument.Parse(DeepSeekV41Fixtures.Read("header_templates.json"));
        List<KeyValuePair<string, long>> tensors = new();
        foreach (JsonElement template in document.RootElement.EnumerateArray())
        {
            string key = template.GetProperty("key").GetString()!;
            long bytes = template.GetProperty("bytes").GetInt64();
            List<int[]> axes = template.GetProperty("ranges").EnumerateArray()
                .Select(static axis => axis.EnumerateArray().SelectMany(static range => Enumerable.Range(
                    range[0].GetInt32(), range[1].GetInt32() - range[0].GetInt32() + 1)).ToArray())
                .ToList();
            foreach (int[] indices in Product(axes))
            {
                string name = key;
                foreach (int index in indices)
                {
                    int hash = name.IndexOf('#', StringComparison.Ordinal);
                    name = string.Concat(name.AsSpan(0, hash), index.ToString(System.Globalization.CultureInfo.InvariantCulture),
                        name.AsSpan(hash + 1));
                }
                tensors.Add(new KeyValuePair<string, long>(name, bytes));
            }
        }
        return tensors;
    }

    private static IEnumerable<int[]> Product(List<int[]> axes)
    {
        IEnumerable<int[]> result = [Array.Empty<int>()];
        foreach (int[] axis in axes)
            result = result.SelectMany(prefix => axis.Select(value => prefix.Append(value).ToArray()));
        return result;
    }
}
