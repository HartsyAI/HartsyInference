using HartsyInference.Core.Engram;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.DeepSeekV41.Engram;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41.Engram;

/// <summary>The factory that wires a checkpoint's Engram table (header offsets, pread handle) into a row store.</summary>
public sealed class EngramTableStoresTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "engram-stores-" + Guid.NewGuid().ToString("N"));

    public EngramTableStoresTests() => Directory.CreateDirectory(_directory);

    public void Dispose() => Directory.Delete(_directory, true);

    [Fact]
    public async Task Open_ReadsTheTinyCheckpointsTableThroughItsHeaderOffsets()
    {
        TinyDeepSeekV41Checkpoint.Write(_directory);
        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(_directory);
        DeepSeekV41EngramTable table = checkpoint.EngramTable(TinyDeepSeekV41Checkpoint.EngramLayer);

        OfficialFp8E8M0RowLayout layout = EngramTableStores.OfficialLayout(table);
        Assert.Equal(table.Embed.FileOffset, layout.GetSlice(0).BaseOffset);
        Assert.Equal(table.Scale!.FileOffset, layout.GetSlice(1).BaseOffset);
        Assert.Equal(TinyDeepSeekV41Checkpoint.EngramRows, layout.Rows);

        using EngramTableStore store = EngramTableStores.Open(table, EngramBacking.HostResident, 1L << 20);
        await store.LoadAllAsync();
        long[] rows = Enumerable.Range(0, TinyDeepSeekV41Checkpoint.EngramRows).Select(r => (long)r).ToArray();
        ushort[] dest = new ushort[rows.Length * 256];
        Array.Fill(dest, (ushort)0xFFFF);
        store.Gather(rows, dest);
        Assert.All(dest, value => Assert.Equal(0, value)); // the tiny table is all-zero codes
        Assert.Throws<ArgumentOutOfRangeException>(() => store.Gather([TinyDeepSeekV41Checkpoint.EngramRows], dest));
    }
}
