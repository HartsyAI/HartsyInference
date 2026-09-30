using HartsyInference.Core.Engram;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.IO;

namespace HartsyInference.LLM.DeepSeekV41.Engram;

/// <summary>Builds row stores for the Engram tables of an opened DeepSeek-V4.1 checkpoint.</summary>
public static class EngramTableStores
{
    /// <summary>The official-checkpoint row layout of <paramref name="table"/>.</summary>
    public static OfficialFp8E8M0RowLayout OfficialLayout(DeepSeekV41EngramTable table)
    {
        ArgumentNullException.ThrowIfNull(table);
        if (table.Scale is null || table.ScaleSource is null)
            throw new HartsyInferenceException($"Engram layer {table.Layer} has no e8m0 scale tensor, so its rows cannot be decoded.");
        if (!ReferenceEquals(table.EmbedSource, table.ScaleSource))
            throw new HartsyInferenceException($"Engram layer {table.Layer} keeps its payload and scales in different shards; the official layout expects one.");
        return new OfficialFp8E8M0RowLayout(table.Rows, table.Embed.FileOffset, table.Scale.FileOffset);
    }

    /// <summary>A store over the whole table of <paramref name="table"/> (or only <paramref name="owned"/>), reading from its pread handle.</summary>
    public static EngramTableStore Open(DeepSeekV41EngramTable table, EngramBacking backing, long budgetBytes, EngramRowRange? owned = null)
    {
        OfficialFp8E8M0RowLayout layout = OfficialLayout(table);
        return new EngramTableStore(layout, table.EmbedSource, backing, owned ?? EngramRowRange.All(table.Rows), budgetBytes);
    }
}
