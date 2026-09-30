using System.Text;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

public sealed class ShardedSafeTensorSetLargeOffsetTests : IDisposable
{
    private const long FiveGiB = 5L << 30;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"st_large_{Guid.NewGuid():N}");

    public ShardedSafeTensorSetLargeOffsetTests() => Directory.CreateDirectory(_dir);

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    [Fact]
    public void OffsetsBeyondFourGiB_AreExactAndReadableByMapAndByPread()
    {
        string path = Path.Combine(_dir, "big.safetensors");
        long tailOffset = FiveGiB;
        string header = $$$"""
            {"head":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},
            "tail":{"dtype":"F32","shape":[1],"data_offsets":[{{{tailOffset}}},{{{tailOffset + 4}}}]}}
            """;
        byte[] json = Encoding.UTF8.GetBytes(header.Trim().ReplaceLineEndings(string.Empty));
        long dataStart = 8 + json.Length;
        using (FileStream stream = new FileStream(path, FileMode.Create, FileAccess.ReadWrite))
        {
            stream.Write(BitConverter.GetBytes((ulong)json.Length));
            stream.Write(json);
            stream.Write(BitConverter.GetBytes(1.5f));
            stream.Position = dataStart + tailOffset;
            stream.Write(BitConverter.GetBytes(-2.25f));
        }

        using ShardedSafeTensorSet mapped = ShardedSafeTensorSet.OpenFiles([path]);
        TensorLocation tail = mapped.Inventory["tail"];
        Assert.Equal(dataStart + tailOffset, tail.FileOffset);
        Assert.True(tail.FileOffset > uint.MaxValue);
        Assert.Equal(-2.25f, mapped.GetTensor("tail").AsReadOnlySpan<float>()[0]);
        Assert.Equal(1.5f, mapped.GetTensor("head").AsReadOnlySpan<float>()[0]);

        using ShardedSafeTensorSet pread = ShardedSafeTensorSet.OpenFiles([path],
            new ShardSetOptions { PreadOnlyShards = ["big.safetensors"] });
        byte[] bytes = new byte[4];
        pread.GetByteSource(pread.Shards[0]).ReadAt(pread.Inventory["tail"].FileOffset, bytes);
        Assert.Equal(-2.25f, BitConverter.ToSingle(bytes));
        Assert.Equal(0, pread.MappedShardCount);
    }
}
