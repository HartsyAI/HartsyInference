using System.Runtime.InteropServices;
using HartsyInference.Core.Backends.Gpu;
using HartsyInference.Core.Memory.Host;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Residency;

public sealed unsafe class MappedUploadStagingTests : IDisposable
{
    private const long ChunkBytes = 1000;
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"mapped-staging-{Guid.NewGuid():N}.bin");
    private readonly List<nint> _ownedNative = [];

    public MappedUploadStagingTests()
    {
        byte[] bytes = new byte[2500];
        new Random(7).NextBytes(bytes);
        File.WriteAllBytes(_path, bytes);
    }

    public void Dispose()
    {
        foreach (nint block in _ownedNative)
            NativeMemory.AlignedFree((void*)block);
        File.Delete(_path);
    }

    [Fact]
    public void StageChunk_MappedTensor_StagedBytesEqualMappedBytes()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);
        Tensor mapped = MappedTensor(handle, 2500);
        using MappedUploadStaging staging = new(ChunkBytes);

        Assert.True(MappedUploadStaging.IsMappedView(mapped));
        List<byte> staged = [];
        for (long offset = 0; offset < 2500;)
        {
            int count = staging.StageChunk(mapped, offset, 2500 - offset, out void* host);
            staged.AddRange(new ReadOnlySpan<byte>(host, count).ToArray());
            staging.MarkCopyComplete();
            offset += count;
        }

        Assert.Equal(File.ReadAllBytes(_path), staged.ToArray());
        Assert.Equal(ChunkBytes, staging.Capacity);
    }

    [Fact]
    public void IsMappedView_ViewOfMappedTensor_IsStaged()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);
        Tensor mapped = MappedTensor(handle, 2500);
        Tensor view = mapped.Reshape(new TensorShape(50, 50));

        Assert.True(MappedUploadStaging.IsMappedView(view));
    }

    [Fact]
    public void IsMappedView_OwnedTensor_IsNotStaged()
    {
        using Tensor owned = new(new TensorShape(2500), DType.U8);

        Assert.False(MappedUploadStaging.IsMappedView(owned));
    }

    [Fact]
    public void IsMappedView_BorrowedNonMappedTensor_IsNotStaged()
    {
        nint block = (nint)NativeMemory.AlignedAlloc(2500, 64);
        _ownedNative.Add(block);
        Tensor borrowed = new((void*)block, new TensorShape(2500), DType.U8);

        Assert.False(MappedUploadStaging.IsMappedView(borrowed));
    }

    [Fact]
    public void StageChunk_BeforeCopyCompletes_Throws()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);
        Tensor mapped = MappedTensor(handle, 2500);
        using MappedUploadStaging staging = new(ChunkBytes);

        staging.StageChunk(mapped, 0, 2500, out _);

        Assert.Throws<InvalidOperationException>(() => staging.StageChunk(mapped, ChunkBytes, 2500 - ChunkBytes, out _));
        Assert.Throws<InvalidOperationException>(() => staging.Dispose());
        staging.MarkCopyComplete();
        staging.StageChunk(mapped, ChunkBytes, 2500 - ChunkBytes, out _);
        staging.MarkCopyComplete();
    }

    [Fact]
    public void StageChunk_OutOfRange_Throws()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);
        Tensor mapped = MappedTensor(handle, 2500);
        using MappedUploadStaging staging = new(ChunkBytes);

        Assert.Throws<ArgumentOutOfRangeException>(() => staging.StageChunk(mapped, 2400, 200, out _));
    }

    private static Tensor MappedTensor(MmapHandle handle, long byteCount)
    {
        Tensor tensor = new(handle.PointerAt(0), new TensorShape(byteCount), DType.U8);
        tensor.SetKeepAlive(handle);
        return tensor;
    }
}
