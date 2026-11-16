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
    public void ReadAt_DisposedWhileReadsRun_EachReadCompletesOrIsRefused()
    {
        MmapHandle handle = MmapHandle.OpenRead(_path);
        using ManualResetEventSlim started = new(false);
        int refusedReaders = 0, wrongReads = 0;
        // Each reader reads until Dispose refuses it; every read before that must return the file's bytes.
        Thread[] readers = Enumerable.Range(0, 8).Select(r => new Thread(() =>
        {
            byte[] read = new byte[MappedTensorPrefetch.ChunkBytes];
            for (int k = 0; k < 100_000; k++)
            {
                started.Set();
                try
                {
                    handle.ReadAt(0, read);
                }
                catch (ObjectDisposedException)
                {
                    Interlocked.Increment(ref refusedReaders);
                    return;
                }
                if (!_bytes.AsSpan(0, read.Length).SequenceEqual(read)) Interlocked.Increment(ref wrongReads);
            }
        })).ToArray();
        foreach (Thread reader in readers) reader.Start();

        started.Wait();
        handle.Dispose();
        foreach (Thread reader in readers) reader.Join();

        Assert.Equal(0, wrongReads);
        Assert.Equal(8, refusedReaders);
    }

    [Fact]
    public void Read_SecondCallOfTheSameSize_AllocatesNothingOnTheCaller()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);
        Tensor registered = new(handle.PointerAt(0), new TensorShape(FileBytes), DType.U8);
        registered.SetKeepAlive(handle);
        TensorFileExtents.Register(registered, handle, 0);
        List<Tensor> tensors = [registered];
        MappedTensorPrefetch.Read(tensors, threads: 4);   // starts the readers and sizes the chunk table

        long before = GC.GetAllocatedBytesForCurrentThread();
        long read = MappedTensorPrefetch.Read(tensors, threads: 4);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(FileBytes, read);
        Assert.Equal(0, allocated);
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
