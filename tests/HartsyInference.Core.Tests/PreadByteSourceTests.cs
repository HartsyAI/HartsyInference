using HartsyInference.Core.Exceptions;
using HartsyInference.Core.IO;
using Xunit;

namespace HartsyInference.Core.Tests;

public sealed class PreadByteSourceTests : IDisposable
{
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"pread_{Guid.NewGuid():N}.bin");
    private readonly byte[] _content;

    public PreadByteSourceTests()
    {
        _content = new byte[4096];
        new Random(7).NextBytes(_content);
        File.WriteAllBytes(_path, _content);
    }

    public void Dispose() => File.Delete(_path);

    [Fact]
    public void ReadAt_PastEndOfFile_ThrowsInsteadOfPartialData()
    {
        using PreadByteSource source = new PreadByteSource(_path);

        Assert.Throws<HartsyInferenceException>(() => source.ReadAt(4090, new byte[10]));
        Assert.Throws<HartsyInferenceException>(() => source.ReadAt(-1, new byte[1]));
    }

    [Fact]
    public async Task ReadBatchAsync_PacksRangesBackToBackInOrder()
    {
        using PreadByteSource source = new PreadByteSource(_path);
        List<ByteRange> ranges = [new ByteRange(3000, 50), new ByteRange(10, 20), new ByteRange(2048, 5)];
        byte[] destination = new byte[75];

        await source.ReadBatchAsync(ranges, destination);

        Assert.Equal(_content.AsSpan(3000, 50).ToArray(), destination[..50]);
        Assert.Equal(_content.AsSpan(10, 20).ToArray(), destination[50..70]);
        Assert.Equal(_content.AsSpan(2048, 5).ToArray(), destination[70..]);
    }

    [Fact]
    public async Task ReadBatchAsync_ReadFailsMidBatch_ThrowsAfterAllReadsSettled()
    {
        using PreadByteSource source = new PreadByteSource(_path);
        using (FileStream truncate = new FileStream(_path, FileMode.Open, FileAccess.Write, FileShare.ReadWrite))
            truncate.SetLength(1024);
        List<ByteRange> ranges = Enumerable.Range(0, 200).Select(i => new ByteRange(i * 16, 16)).ToList();

        await Assert.ThrowsAsync<HartsyInferenceException>(() => source.ReadBatchAsync(ranges, new byte[200 * 16]));
    }
}
