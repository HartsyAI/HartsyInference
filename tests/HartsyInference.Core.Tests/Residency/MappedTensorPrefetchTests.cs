using System.Runtime.InteropServices;
using HartsyInference.Core.Memory.Host;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Residency;

public sealed unsafe class MappedTensorPrefetchTests : IDisposable
{
    // Over two prefetch chunks, so one tensor is split across readers.
    private const int FileBytes = MappedTensorPrefetch.ChunkBytes * 2 + 12345;
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"mapped-prefetch-{Guid.NewGuid():N}.bin");
    private readonly byte[] _bytes = new byte[FileBytes];
    private readonly List<nint> _ownedNative = [];

    public MappedTensorPrefetchTests()
    {
        new Random(11).NextBytes(_bytes);
        File.WriteAllBytes(_path, _bytes);
    }

    public void Dispose()
    {
        foreach (nint block in _ownedNative)
            NativeMemory.AlignedFree((void*)block);
        File.Delete(_path);
    }

    [Fact]
    public void ReadAt_FromManyThreadsAtOnce_ReturnsTheFileBytes()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);
        int[] offsets = Enumerable.Range(0, 64).Select(i => i * (FileBytes / 64)).ToArray();

        Parallel.For(0, offsets.Length, new ParallelOptions { MaxDegreeOfParallelism = 16 }, i =>
        {
            byte[] read = new byte[4096];
            handle.ReadAt(offsets[i], read);
            Assert.Equal(_bytes.AsSpan(offsets[i], read.Length).ToArray(), read);
        });
    }

    [Fact]
    public void ReadAt_AfterDispose_ThrowsObjectDisposed()
    {
        MmapHandle handle = MmapHandle.OpenRead(_path);
        handle.Dispose();

        Assert.Throws<ObjectDisposedException>(() => handle.ReadAt(0, new byte[16]));
    }

    [Fact]
    public void Read_RegisteredTensor_ReadsEveryByte_AndSkipsUnregisteredOnes()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);
        Tensor registered = new(handle.PointerAt(100), new TensorShape(FileBytes - 100), DType.U8);
        registered.SetKeepAlive(handle);
        TensorFileExtents.Register(registered, handle, 100);
        Tensor mappedOnly = new(handle.PointerAt(0), new TensorShape(1000), DType.U8);
        mappedOnly.SetKeepAlive(handle);
        nint block = (nint)NativeMemory.AlignedAlloc(1000, 64);
        _ownedNative.Add(block);
        Tensor borrowed = new((void*)block, new TensorShape(1000), DType.U8);
        using Tensor owned = new(new TensorShape(1000), DType.U8);

        long read = MappedTensorPrefetch.Read([registered, mappedOnly, borrowed, owned], threads: 4);

        Assert.Equal(FileBytes - 100, read);
    }

    [Fact]
    public void Read_ZeroThreads_ReadsNothing()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);
        Tensor registered = new(handle.PointerAt(0), new TensorShape(FileBytes), DType.U8);
        registered.SetKeepAlive(handle);
        TensorFileExtents.Register(registered, handle, 0);

        Assert.Equal(0, MappedTensorPrefetch.Read([registered], threads: 0));
    }

    [Fact]
    public void Read_DisposedMapping_IsSkippedWithoutThrowing()
    {
        MmapHandle handle = MmapHandle.OpenRead(_path);
        Tensor registered = new(handle.PointerAt(0), new TensorShape(FileBytes), DType.U8);
        TensorFileExtents.Register(registered, handle, 0);
        handle.Dispose();

        Assert.Equal(0, MappedTensorPrefetch.Read([registered], threads: 4));
    }
}
