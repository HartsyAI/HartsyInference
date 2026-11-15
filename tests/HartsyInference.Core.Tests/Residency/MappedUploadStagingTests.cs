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
    private readonly List<string> _extraFiles = [];

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
        foreach (string file in _extraFiles)
            File.Delete(file);
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

    [Fact]
    public void StageChunk_FileBackedTensor_StagedBytesEqualMappedBytes()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);
        Tensor mapped = MappedTensor(handle, 2500);
        Tensor fileBacked = MappedTensor(handle, 2500);
        TensorFileExtents.Register(fileBacked, handle, 0);
        using MappedUploadStaging staging = new(ChunkBytes);

        byte[] viaMapping = StageAll(staging, mapped);
        byte[] viaFile = StageAll(staging, fileBacked);

        Assert.Equal(File.ReadAllBytes(_path), viaFile);
        Assert.Equal(viaMapping, viaFile);
    }

    [Fact]
    public void StageChunk_FileBackedTensorAtOffset_ReadsFromItsFileOffset()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);
        Tensor tensor = new(handle.PointerAt(100), new TensorShape(1000), DType.U8);
        tensor.SetKeepAlive(handle);
        TensorFileExtents.Register(tensor, handle, 100);
        using MappedUploadStaging staging = new(ChunkBytes);

        Assert.Equal(File.ReadAllBytes(_path).AsSpan(100, 1000).ToArray(), StageAll(staging, tensor));
    }

    [Fact]
    public void StageChunk_ViewOfFileBackedTensor_ReadsAtTheViewsOffsetInTheFile()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);
        Tensor root = MappedTensor(handle, 2500);
        TensorFileExtents.Register(root, handle, 0);
        Tensor reshaped = root.Reshape(new TensorShape(50, 50));
        Tensor sub = new((byte*)root.DataPointer + 100, new TensorShape(1000), DType.U8);
        sub.SetKeepAlive(root);
        using MappedUploadStaging staging = new(ChunkBytes);

        Assert.Equal(File.ReadAllBytes(_path), StageAll(staging, reshaped));
        Assert.Equal(File.ReadAllBytes(_path).AsSpan(100, 1000).ToArray(), StageAll(staging, sub));
    }

    [Fact]
    public void StageChunk_FileBackedTensor_ReadsTheRegisteredFileNotTheMapping()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);
        string decoy = WriteDecoyFile();
        using MmapHandle decoyHandle = MmapHandle.OpenRead(decoy);
        Tensor tensor = MappedTensor(handle, 2500);
        TensorFileExtents.Register(tensor, decoyHandle, 0);
        using MappedUploadStaging staging = new(ChunkBytes);

        Assert.Equal(File.ReadAllBytes(decoy), StageAll(staging, tensor));
    }

    [Fact]
    public void StageChunk_TensorWithoutFileExtent_ReadsTheMapping()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);
        WriteDecoyFile();
        Tensor tensor = MappedTensor(handle, 2500);
        using MappedUploadStaging staging = new(ChunkBytes);

        Assert.False(TensorFileExtents.TryGet(tensor, out _, out _));
        Assert.Equal(File.ReadAllBytes(_path), StageAll(staging, tensor));
    }

    [Fact]
    public void StageChunk_ReloadedCheckpointAtSamePath_ReadsTheNewFile()
    {
        using MappedUploadStaging staging = new(ChunkBytes);
        byte[] original = File.ReadAllBytes(_path);
        byte[] first = StageMappedFile(staging, _path);
        byte[] replacement = new byte[2500];
        new Random(11).NextBytes(replacement);
        File.WriteAllBytes(_path, replacement);

        byte[] second = StageMappedFile(staging, _path);

        Assert.Equal(original, first);
        Assert.Equal(replacement, second);
    }

    private static byte[] StageMappedFile(MappedUploadStaging staging, string path)
    {
        using MmapHandle handle = MmapHandle.OpenRead(path);
        Tensor tensor = MappedTensor(handle, 2500);
        TensorFileExtents.Register(tensor, handle, 0);
        return StageAll(staging, tensor);
    }

    private string WriteDecoyFile()
    {
        string decoy = Path.Combine(Path.GetTempPath(), $"mapped-staging-decoy-{Guid.NewGuid():N}.bin");
        _extraFiles.Add(decoy);
        File.WriteAllBytes(decoy, Enumerable.Repeat((byte)0xAB, 2500).ToArray());
        return decoy;
    }

    private static byte[] StageAll(MappedUploadStaging staging, Tensor source)
    {
        long total = source.DType.ComputeByteCount(source.ElementCount);
        List<byte> staged = [];
        for (long offset = 0; offset < total;)
        {
            int count = staging.StageChunk(source, offset, total - offset, out void* host);
            staged.AddRange(new ReadOnlySpan<byte>(host, count).ToArray());
            staging.MarkCopyComplete();
            offset += count;
        }
        return staged.ToArray();
    }

    private static Tensor MappedTensor(MmapHandle handle, long byteCount)
    {
        Tensor tensor = new(handle.PointerAt(0), new TensorShape(byteCount), DType.U8);
        tensor.SetKeepAlive(handle);
        return tensor;
    }
}
