using HartsyInference.Core.Engram;
using HartsyInference.Core.IO;
using Xunit;

namespace HartsyInference.Core.Tests.Engram;

/// <summary>Store behavior on a synthetic shard-format file: same byte layout as the real checkpoint shards, 96 rows.</summary>
public sealed class EngramTableStoreTests : IDisposable
{
    private const long EmbedOffset = 664;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "engram-store-" + Guid.NewGuid().ToString("N"));
    private readonly SyntheticTables _tables;
    private readonly string _officialPath;
    private readonly long _scaleOffset;

    public EngramTableStoreTests()
    {
        Directory.CreateDirectory(_dir);
        _tables = SyntheticTables.Load(EngramFixtures.RowDecode.GetProperty("synthetic"));
        _officialPath = Path.Combine(_dir, "official.safetensors");
        _scaleOffset = EmbedOffset + _tables.Rows * 256L;
        byte[] file = new byte[_scaleOffset + _tables.Rows * 8L];
        _tables.Fp8.CopyTo(file, EmbedOffset);
        _tables.E8m0.CopyTo(file, _scaleOffset);
        File.WriteAllBytes(_officialPath, file);
    }

    public void Dispose() => Directory.Delete(_dir, true);

    private OfficialFp8E8M0RowLayout Layout() => new(_tables.Rows, EmbedOffset, _scaleOffset);

    private static long Budget(int rows) => rows * 264L;

    [Fact]
    public void Gather_MatchesPythonForEveryRow()
    {
        using PreadByteSource source = new(_officialPath);
        using EngramTableStore store = new(Layout(), source, EngramBacking.Storage, EngramRowRange.All(_tables.Rows), Budget(_tables.Rows));
        long[] rows = Enumerable.Range(0, _tables.Rows).Select(r => (long)r).ToArray();
        ushort[] dest = new ushort[rows.Length * 256];
        store.Gather(rows, dest);
        for (int r = 0; r < rows.Length; r++)
            EngramFixtures.AssertBf16Equal(_tables.Expected(SyntheticTables.Kind.Official, r), dest.AsSpan(r * 256, 256), $"row {r}");
    }

    [Fact]
    public void Gather_KeepsRequestOrderAndRepeats()
    {
        using PreadByteSource source = new(_officialPath);
        using EngramTableStore store = new(Layout(), source, EngramBacking.Storage, EngramRowRange.All(_tables.Rows), Budget(8));
        long[] rows = [40, 3, 40, 95, 3, 3];
        ushort[] dest = new ushort[rows.Length * 256];
        store.Gather(rows, dest);
        for (int i = 0; i < rows.Length; i++)
            EngramFixtures.AssertBf16Equal(_tables.Expected(SyntheticTables.Kind.Official, (int)rows[i]), dest.AsSpan(i * 256, 256), $"slot {i}");
        EngramStoreStats stats = store.Stats;
        Assert.Equal(6, stats.RowsRequested);
        Assert.Equal(3, stats.UniqueRows);
        Assert.Equal(3, stats.CacheMisses);
        Assert.Equal(3, stats.ResidentRows);
    }

    [Fact]
    public async Task Prefetch_LoadsSoGatherDoesNoIo()
    {
        CountingSource counting = new(new PreadByteSource(_officialPath));
        using EngramTableStore store = new(Layout(), counting, EngramBacking.Storage, EngramRowRange.All(_tables.Rows), Budget(32));
        long[] rows = [10, 11, 50, 10];
        await store.PrefetchAsync(rows);
        long reads = counting.Reads;
        Assert.True(reads > 0);
        Assert.Equal(1, store.Stats.PrefetchCalls);
        ushort[] dest = new ushort[rows.Length * 256];
        store.Gather(rows, dest);
        Assert.Equal(reads, counting.Reads);
        Assert.Equal(0, store.Stats.GatherMisses);
        for (int i = 0; i < rows.Length; i++)
            EngramFixtures.AssertBf16Equal(_tables.Expected(SyntheticTables.Kind.Official, (int)rows[i]), dest.AsSpan(i * 256, 256), $"slot {i}");
    }

    [Fact]
    public void Storage_EvictsLeastRecentlyUsedRows()
    {
        using PreadByteSource source = new(_officialPath);
        using EngramTableStore store = new(Layout(), source, EngramBacking.Storage, EngramRowRange.All(_tables.Rows), Budget(4));
        ushort[] dest = new ushort[256];
        foreach (long row in new long[] { 0, 1, 2, 3 })
            store.Gather([row], dest);
        Assert.Equal(4, store.Stats.ResidentRows);
        store.Gather([0], dest); // 0 is now most recent; 1 is the LRU
        store.Gather([4], dest); // evicts 1
        EngramStoreStats stats = store.Stats;
        Assert.Equal(1, stats.Evictions);
        Assert.Equal(4, stats.ResidentRows);
        long missesBefore = stats.GatherMisses;
        store.Gather([0], dest); // still resident
        Assert.Equal(missesBefore, store.Stats.GatherMisses);
        store.Gather([1], dest); // evicted, must reload and still decode right
        Assert.Equal(missesBefore + 1, store.Stats.GatherMisses);
        EngramFixtures.AssertBf16Equal(_tables.Expected(SyntheticTables.Kind.Official, 1), dest, "reloaded row 1");
    }

    [Fact]
    public void Storage_BatchOfDistinctRowsLargerThanTheCacheThrows()
    {
        using PreadByteSource source = new(_officialPath);
        using EngramTableStore store = new(Layout(), source, EngramBacking.Storage, EngramRowRange.All(_tables.Rows), Budget(4));
        Assert.Equal(4, store.MaxBatchRows);
        long[] rows = [0, 1, 2, 3, 4];
        ushort[] dest = new ushort[rows.Length * 256];
        Assert.Throws<InvalidOperationException>(() => store.Gather(rows, dest));
    }

    [Fact]
    public async Task RowsOutsideTheOwnedRangeThrow()
    {
        using PreadByteSource source = new(_officialPath);
        using EngramTableStore store = new(Layout(), source, EngramBacking.Storage, new EngramRowRange(10, 20), Budget(20));
        ushort[] dest = new ushort[256];
        Assert.Throws<ArgumentOutOfRangeException>(() => store.Gather([9], dest));
        Assert.Throws<ArgumentOutOfRangeException>(() => store.Gather([30], dest));
        Assert.Throws<ArgumentOutOfRangeException>(() => store.Gather([-1], dest));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => store.PrefetchAsync(new long[] { 10, 30 }));
        store.Gather([10], dest);
        store.Gather([29], dest);
        EngramFixtures.AssertBf16Equal(_tables.Expected(SyntheticTables.Kind.Official, 29), dest, "row 29");
    }

    [Fact]
    public async Task HostResident_LoadAllThenServesEverythingWithoutIo()
    {
        CountingSource counting = new(new PreadByteSource(_officialPath));
        using EngramTableStore store = new(Layout(), counting, EngramBacking.HostResident, EngramRowRange.All(_tables.Rows), Budget(_tables.Rows));
        await store.LoadAllAsync();
        Assert.Equal(_tables.Rows, store.Stats.ResidentRows);
        long reads = counting.Reads;
        long[] rows = [95, 0, 47, 47];
        ushort[] dest = new ushort[rows.Length * 256];
        store.Gather(rows, dest);
        Assert.Equal(reads, counting.Reads);
        Assert.Equal(0, store.Stats.Evictions);
        for (int i = 0; i < rows.Length; i++)
            EngramFixtures.AssertBf16Equal(_tables.Expected(SyntheticTables.Kind.Official, (int)rows[i]), dest.AsSpan(i * 256, 256), $"slot {i}");
    }

    [Fact]
    public void ConcurrentGathersOfOverlappingRowsAllDecodeCorrectly()
    {
        using PreadByteSource source = new(_officialPath);
        using EngramTableStore store = new(Layout(), source, EngramBacking.Storage, EngramRowRange.All(_tables.Rows), Budget(48));
        Parallel.For(0, 8, new ParallelOptions { MaxDegreeOfParallelism = 8 }, worker =>
        {
            Random rng = new(worker);
            long[] rows = new long[12];
            ushort[] dest = new ushort[rows.Length * 256];
            for (int iter = 0; iter < 60; iter++)
            {
                for (int i = 0; i < rows.Length; i++)
                    rows[i] = rng.Next(_tables.Rows);
                store.Gather(rows, dest);
                for (int i = 0; i < rows.Length; i++)
                    EngramFixtures.AssertBf16Equal(_tables.Expected(SyntheticTables.Kind.Official, (int)rows[i]), dest.AsSpan(i * 256, 256), $"worker {worker} slot {i}");
            }
        });
    }

    private sealed class CountingSource : IWeightByteSource
    {
        private readonly IWeightByteSource _inner;
        private long _reads;

        public CountingSource(IWeightByteSource inner) => _inner = inner;

        /// <summary>Individual ranges read (a batch of N ranges counts N).</summary>
        public long Reads => Interlocked.Read(ref _reads);

        public long Length => _inner.Length;

        public void ReadAt(long offset, Span<byte> destination)
        {
            Interlocked.Increment(ref _reads);
            _inner.ReadAt(offset, destination);
        }

        public Task ReadBatchAsync(IReadOnlyList<ByteRange> ranges, Memory<byte> destination, CancellationToken cancellationToken = default)
        {
            Interlocked.Add(ref _reads, ranges.Count);
            return _inner.ReadBatchAsync(ranges, destination, cancellationToken);
        }

        public void Dispose() => _inner.Dispose();
    }
}
