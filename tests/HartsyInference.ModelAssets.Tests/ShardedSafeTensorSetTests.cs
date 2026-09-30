using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

public sealed class ShardedSafeTensorSetTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"st_shards_{Guid.NewGuid():N}");

    public ShardedSafeTensorSetTests() => Directory.CreateDirectory(_dir);

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private string PathOf(string name) => Path.Combine(_dir, name);

    [Fact]
    public void OpenIndex_HappyPath_BuildsCompleteInventoryWithoutMapping()
    {
        ShardTestFiles.WriteThreeShardSet(_dir, out long total);

        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenIndex(_dir);

        Assert.Equal(5, set.Inventory.Count);
        Assert.Equal(3, set.Shards.Count);
        Assert.Equal(total, set.TotalTensorBytes);
        Assert.Equal(total, set.DeclaredTotalSize);
        Assert.Equal(0, set.MappedShardCount);
        TensorLocation b1 = set.Inventory["b.1"];
        Assert.Equal("s2.safetensors", b1.Shard.FileName);
        Assert.Equal(DType.F32, b1.DType);
        Assert.Equal(4, b1.ByteLength);
    }

    [Fact]
    public unsafe void GetTensor_MapsOnlyTheOwningShardAndReadsTheRightBytes()
    {
        ShardTestFiles.WriteThreeShardSet(_dir, out _);
        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenIndex(_dir);

        Tensor tensor = set.GetTensor("b.1");

        Assert.Equal(1, set.MappedShardCount);
        Assert.True(set.Shards[1].IsMapped);
        Assert.Equal(9f, tensor.AsReadOnlySpan<float>()[0]);
        Assert.Equal(12f, set.GetTensor("c.0").AsReadOnlySpan<float>()[2]);
        Assert.Equal(2, set.MappedShardCount);
    }

    [Fact]
    public void GetTensor_UnknownName_Throws()
    {
        ShardTestFiles.WriteThreeShardSet(_dir, out _);
        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenIndex(_dir);

        Assert.Throws<KeyNotFoundException>(() => set.GetTensor("nope"));
    }

    [Fact]
    public void OpenIndex_MissingShardFile_NamesTheFile()
    {
        ShardTestFiles.WriteThreeShardSet(_dir, out _);
        File.Delete(PathOf("s2.safetensors"));

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => ShardedSafeTensorSet.OpenIndex(_dir));

        Assert.Contains("s2.safetensors", error.Message);
        Assert.Contains("missing on disk", error.Message);
    }

    [Fact]
    public void OpenIndex_KeyInIndexButNotInHeader_IsReported()
    {
        Dictionary<string, string> map = ShardTestFiles.WriteThreeShardSet(_dir, out long total);
        map["ghost.weight"] = "s3.safetensors";
        ShardTestFiles.WriteIndex(_dir, map, total);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => ShardedSafeTensorSet.OpenIndex(_dir));

        Assert.Contains("in the index but not in the header", error.Message);
        Assert.Contains("ghost.weight", error.Message);
    }

    [Fact]
    public void OpenIndex_KeyInHeaderButNotInIndex_IsReported()
    {
        Dictionary<string, string> map = ShardTestFiles.WriteThreeShardSet(_dir, out long total);
        map.Remove("a.1");
        ShardTestFiles.WriteIndex(_dir, map, total);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => ShardedSafeTensorSet.OpenIndex(_dir));

        Assert.Contains("in shard headers but not in the index", error.Message);
        Assert.Contains("a.1", error.Message);
    }

    [Fact]
    public void OpenIndex_KeyInAnotherShardThanIndexed_IsReported()
    {
        Dictionary<string, string> map = ShardTestFiles.WriteThreeShardSet(_dir, out long total);
        ShardTestFiles.WriteShard(PathOf("s3.safetensors"), ShardTestFiles.F32("c.0", 10, 11, 12), ShardTestFiles.F32("c.1", 1));
        map["c.1"] = "s3.safetensors";
        map["c.0"] = "s1.safetensors";
        ShardTestFiles.WriteIndex(_dir, map, total + 4);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => ShardedSafeTensorSet.OpenIndex(_dir));

        Assert.Contains("different shard than the index says", error.Message);
    }

    [Fact]
    public void OpenIndex_DuplicateKeyAcrossShards_NamesBothShards()
    {
        Dictionary<string, string> map = ShardTestFiles.WriteThreeShardSet(_dir, out long total);
        ShardTestFiles.WriteShard(PathOf("s3.safetensors"),
            ShardTestFiles.F32("c.0", 10, 11, 12), ShardTestFiles.F32("a.0", 1, 2, 3, 4));
        ShardTestFiles.WriteIndex(_dir, map, total + 16);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => ShardedSafeTensorSet.OpenIndex(_dir));

        Assert.Contains("more than one shard", error.Message);
        Assert.Contains("'a.0' in s1.safetensors and s3.safetensors", error.Message);
    }

    [Fact]
    public void OpenIndex_TotalSizeMismatch_ReportsBothNumbers()
    {
        Dictionary<string, string> map = ShardTestFiles.WriteThreeShardSet(_dir, out long total);
        ShardTestFiles.WriteIndex(_dir, map, total + 100);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => ShardedSafeTensorSet.OpenIndex(_dir));

        Assert.Contains($"sum to {total}", error.Message);
        Assert.Contains($"total_size is {total + 100}", error.Message);
    }

    [Fact]
    public void OpenIndex_TotalSizeMismatch_CanBeRelaxed()
    {
        Dictionary<string, string> map = ShardTestFiles.WriteThreeShardSet(_dir, out long total);
        ShardTestFiles.WriteIndex(_dir, map, total + 100);

        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenIndex(_dir, new ShardSetOptions { RequireTotalSizeMatch = false });

        Assert.Equal(5, set.Inventory.Count);
    }

    [Fact]
    public void OpenIndex_ExpectedTotalBytes_IsEnforcedIndependentlyOfTheIndex()
    {
        ShardTestFiles.WriteThreeShardSet(_dir, out long total);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
            () => ShardedSafeTensorSet.OpenIndex(_dir, new ShardSetOptions { ExpectedTotalBytes = total + 1 }));

        Assert.Contains("were expected", error.Message);
    }

    [Fact]
    public void OpenIndex_TruncatedShard_FailsInsteadOfMisleading()
    {
        ShardTestFiles.WriteThreeShardSet(_dir, out _);
        using (FileStream stream = new FileStream(PathOf("s1.safetensors"), FileMode.Open, FileAccess.Write))
            stream.SetLength(stream.Length - 4);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => ShardedSafeTensorSet.OpenIndex(_dir));

        Assert.Contains("truncated", error.Message);
    }

    [Fact]
    public void OpenIndex_AggregatesSeveralProblemsIntoOneError()
    {
        Dictionary<string, string> map = ShardTestFiles.WriteThreeShardSet(_dir, out long total);
        map["ghost"] = "s3.safetensors";
        map.Remove("a.1");
        ShardTestFiles.WriteIndex(_dir, map, total + 1);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => ShardedSafeTensorSet.OpenIndex(_dir));

        Assert.Contains("ghost", error.Message);
        Assert.Contains("a.1", error.Message);
        Assert.Contains("total_size", error.Message);
    }

    [Fact]
    public void OpenIndex_WeightMapPathEscape_IsRejected()
    {
        ShardTestFiles.WriteIndex(_dir, new Dictionary<string, string> { ["x"] = "../evil.safetensors" }, null);

        Assert.Throws<HartsyInferenceException>(() => ShardedSafeTensorSet.OpenIndex(_dir));
    }

    [Fact]
    public void OpenIndex_NoIndexFile_Throws()
    {
        Assert.Throws<FileNotFoundException>(() => ShardedSafeTensorSet.OpenIndex(_dir));
    }

    [Fact]
    public void OpenFiles_OddNames_BuildsInventoryWithoutAnIndex()
    {
        ShardTestFiles.WriteShard(PathOf("weights.00.safetensors"), ShardTestFiles.F32("x", 1, 2));
        ShardTestFiles.WriteShard(PathOf("weights.01.safetensors"), ShardTestFiles.F32("y", 3));

        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenFiles(
            [PathOf("weights.00.safetensors"), PathOf("weights.01.safetensors")]);

        Assert.Equal(2, set.Inventory.Count);
        Assert.Null(set.DeclaredTotalSize);
        Assert.Equal(12, set.TotalTensorBytes);
    }

    [Fact]
    public void OpenFiles_DuplicateKeyAcrossFiles_IsReported()
    {
        ShardTestFiles.WriteShard(PathOf("a.safetensors"), ShardTestFiles.F32("x", 1));
        ShardTestFiles.WriteShard(PathOf("b.safetensors"), ShardTestFiles.F32("x", 2));

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
            () => ShardedSafeTensorSet.OpenFiles([PathOf("a.safetensors"), PathOf("b.safetensors")]));

        Assert.Contains("'x' in a.safetensors and b.safetensors", error.Message);
    }

    [Fact]
    public void OpenFiles_MissingFile_Throws()
    {
        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
            () => ShardedSafeTensorSet.OpenFiles([PathOf("gone.safetensors")]));

        Assert.Contains("gone.safetensors", error.Message);
    }

    [Fact]
    public void PreadOnlyShard_IsNeverMappedAndServesBytesThroughTheByteSource()
    {
        ShardTestFiles.WriteThreeShardSet(_dir, out _);
        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenIndex(_dir,
            new ShardSetOptions { PreadOnlyShards = ["s2.safetensors"] });

        InvalidOperationException refusal = Assert.Throws<InvalidOperationException>(() => set.GetTensor("b.0"));
        Assert.Contains("GetByteSource", refusal.Message);

        TensorLocation location = set.Inventory["b.1"];
        Assert.True(location.Shard.PreadOnly);
        byte[] bytes = new byte[location.ByteLength];
        set.GetByteSource(location.Shard).ReadAt(location.FileOffset, bytes);
        Assert.Equal(9f, BitConverter.ToSingle(bytes));
        Assert.False(location.Shard.IsMapped);
        Assert.Equal(0, set.MappedShardCount);

        Assert.Equal(1f, set.GetTensor("a.0").AsReadOnlySpan<float>()[0]);
        Assert.Equal(1, set.MappedShardCount);
    }

    [Fact]
    public void PreadOnlyShards_NamingAnUnknownShard_Throws()
    {
        ShardTestFiles.WriteThreeShardSet(_dir, out _);

        Assert.Throws<ArgumentException>(() => ShardedSafeTensorSet.OpenIndex(_dir,
            new ShardSetOptions { PreadOnlyShards = ["typo.safetensors"] }));
    }

    [Fact]
    public void GetTensor_Fnuz_IsRefusedWithAClearError()
    {
        ShardTestFiles.WriteShard(PathOf("fnuz.safetensors"),
            new ShardTestFiles.Entry("w", "F8_E4M3FNUZ", [2], [1, 2]));
        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenFiles([PathOf("fnuz.safetensors")]);

        Assert.Equal(DType.F8E4M3Fnuz, set.Inventory["w"].DType);
        UnsupportedModelException error = Assert.Throws<UnsupportedModelException>(() => set.GetTensor("w"));

        Assert.Contains("FNUZ", error.Message);
        Assert.Equal(0, set.MappedShardCount);
    }

    [Fact]
    public void Dispose_UnmapsAndBlocksFurtherUse()
    {
        ShardTestFiles.WriteThreeShardSet(_dir, out _);
        ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenIndex(_dir);
        set.GetTensor("a.0");

        set.Dispose();
        set.Dispose();

        Assert.Equal(0, set.MappedShardCount);
        Assert.Throws<ObjectDisposedException>(() => set.GetTensor("a.0"));
    }
}
