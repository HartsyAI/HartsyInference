using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Opening the 475 GiB DeepSeek-V4.1-Flash checkpoint must cost header bytes only. The directory can be the real
/// download or a sparse-file replica built from the real headers, since only headers are ever read. RSS is judged as
/// growth because the test host alone sits near 200 MiB before the checkpoint is touched.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class ShardSetHeaderOnlyTests
{
    private const string DirEnvVar = "HARTSY_DSV41_FLASH_DIR";
    private const long OfficialTotalBytes = 510_286_023_000;
    private const int OfficialTensorCount = 96_085;
    private const int OfficialShardCount = 48;
    private const long RssGrowthLimitBytes = 128L << 20;

    private readonly ITestOutputHelper _output;

    public ShardSetHeaderOnlyTests(ITestOutputHelper output) => _output = output;

    private static string CheckpointDir =>
        Environment.GetEnvironmentVariable(DirEnvVar) ?? Path.Combine(TestPaths.ModelsDir, "DeepSeek-V4.1-Flash");

    [Fact]
    public void OpenIndex_OfficialCheckpoint_ReadsHeadersOnlyAndNeverMaps()
    {
        if (!RealWeightGate.Require(_output.WriteLine, Path.Combine(CheckpointDir, "model.safetensors.index.json"))) return;

        long before = ResidentBytes();
        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenIndex(CheckpointDir,
            new ShardSetOptions { ExpectedTotalBytes = OfficialTotalBytes });

        long rss = ResidentBytes();
        int mappings = MappingsUnder(CheckpointDir);
        _output.WriteLine($"{set.Inventory.Count} tensors, {set.Shards.Count} shards, {set.TotalTensorBytes} bytes, "
            + $"RSS {before >> 20} -> {rss >> 20} MiB, {mappings} mappings of the checkpoint");

        Assert.Equal(OfficialTensorCount, set.Inventory.Count);
        Assert.Equal(OfficialShardCount, set.Shards.Count);
        Assert.Equal(OfficialTotalBytes, set.TotalTensorBytes);
        Assert.Equal(0, set.MappedShardCount);
        Assert.Equal(0, mappings);
        Assert.True(rss - before < RssGrowthLimitBytes,
            $"RSS grew {(rss - before) >> 20} MiB opening the checkpoint; the limit is {RssGrowthLimitBytes >> 20} MiB.");
    }

    [Fact]
    public void OpenIndex_OfficialCheckpoint_CoversTheDtypesThePlanNeeds()
    {
        if (!RealWeightGate.Require(_output.WriteLine, Path.Combine(CheckpointDir, "model.safetensors.index.json"))) return;

        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenIndex(CheckpointDir);
        HashSet<DType> seen = set.Inventory.Values.Select(location => location.DType).ToHashSet();
        _output.WriteLine(string.Join(", ", seen.Select(dtype => dtype.Name).Order()));

        Assert.Contains(DType.F8E4M3, seen);
        Assert.Contains(DType.F8E8M0, seen);
        Assert.Contains(DType.I8, seen);
        Assert.Contains(DType.BF16, seen);
        Assert.Contains(DType.F32, seen);
        Assert.DoesNotContain(seen, dtype => dtype.IsFnuz);
    }

    private static long ResidentBytes()
    {
        foreach (string line in File.ReadLines("/proc/self/status"))
        {
            if (line.StartsWith("VmRSS:", StringComparison.Ordinal))
                return long.Parse(line.Split(' ', StringSplitOptions.RemoveEmptyEntries)[1]) * 1024;
        }
        throw new InvalidOperationException("VmRSS not found in /proc/self/status.");
    }

    private static int MappingsUnder(string directory)
    {
        string prefix = Path.GetFullPath(directory);
        return File.ReadLines("/proc/self/maps").Count(line => line.Contains(prefix, StringComparison.Ordinal));
    }
}
