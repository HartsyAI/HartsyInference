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
    public void Read_SecondCallOfTheSameSize_AllocatesNothingOnTheCallerThread()
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
    public void Begin_WaitForEachTensorInTurn_ReturnsOnceItIsRead_AndDisposeFinishesTheRest()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);
        List<Tensor> tensors = [];
        int[] offsets = [0, MappedTensorPrefetch.ChunkBytes, MappedTensorPrefetch.ChunkBytes * 2];
        foreach (int offset in offsets)
        {
            Tensor tensor = new(handle.PointerAt(offset), new TensorShape(4096), DType.U8);
            tensor.SetKeepAlive(handle);
            TensorFileExtents.Register(tensor, handle, offset);
            tensors.Add(tensor);
        }
        nint block = (nint)NativeMemory.AlignedAlloc(64, 64);
        _ownedNative.Add(block);
        tensors.Add(new Tensor((void*)block, new TensorShape(64), DType.U8));   // no file extent: nothing to wait for

        using (MappedTensorPrefetch.Batch batch = MappedTensorPrefetch.Begin(tensors, threads: 2))
        {
            for (int i = 0; i < tensors.Count; i++) batch.WaitFor(i);
            Assert.Throws<ArgumentOutOfRangeException>(() => batch.WaitFor(tensors.Count));
        }

        // The batch ended, so the next one starts rather than waiting on it (bounded, so a gate left held fails instead of hanging).
        Assert.Equal(4096 * 3, ReadWithin(tensors, threads: 2));
    }

    [Fact]
    public void Batch_DisposedTwiceOrThroughAStaleCopy_LeavesTheNextBatchAlone()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);
        Tensor registered = new(handle.PointerAt(0), new TensorShape(FileBytes), DType.U8);
        registered.SetKeepAlive(handle);
        TensorFileExtents.Register(registered, handle, 0);
        List<Tensor> tensors = [registered];

        MappedTensorPrefetch.Batch first = MappedTensorPrefetch.Begin(tensors, threads: 2);
        MappedTensorPrefetch.Batch copy = first;
        first.Dispose();
        first.Dispose();   // already ended: no second release of the gate

        long read = -1;
        Thread other = new(() =>
        {
            using MappedTensorPrefetch.Batch next = MappedTensorPrefetch.Begin(tensors, threads: 2);
            copy.Dispose();     // a stale copy must not end the running batch
            copy.WaitFor(0);    // nor read its state
            read = next.Finish();
        });
        other.Start();
        Assert.True(other.Join(TimeSpan.FromSeconds(30)), "the second batch did not start or finish");
        Assert.Equal(FileBytes, read);
        Assert.Equal(FileBytes, ReadWithin(tensors, threads: 2));
    }

    [Fact]
    public void Begin_WhileThisThreadHoldsABatch_Throws()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);
        Tensor registered = new(handle.PointerAt(0), new TensorShape(4096), DType.U8);
        registered.SetKeepAlive(handle);
        TensorFileExtents.Register(registered, handle, 0);

        using MappedTensorPrefetch.Batch batch = MappedTensorPrefetch.Begin([registered], threads: 1);
        Assert.Throws<InvalidOperationException>(() => MappedTensorPrefetch.Begin([registered], threads: 1));
    }

    // Read on another thread, failing after 30 s rather than hanging the run if a batch left the gate held.
    private static long ReadWithin(IReadOnlyList<Tensor> tensors, int threads)
    {
        long read = -1;
        Thread reader = new(() => read = MappedTensorPrefetch.Read(tensors, threads));
        reader.Start();
        Assert.True(reader.Join(TimeSpan.FromSeconds(30)), "the prefetch gate was still held");
        return read;
    }

    [Fact]
    public void Begin_ZeroThreads_IsAnEmptyBatchThatWaitsForNothing()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);
        Tensor registered = new(handle.PointerAt(0), new TensorShape(FileBytes), DType.U8);
        registered.SetKeepAlive(handle);
        TensorFileExtents.Register(registered, handle, 0);

        using MappedTensorPrefetch.Batch batch = MappedTensorPrefetch.Begin([registered], threads: 0);
        batch.WaitFor(0);
        batch.WaitFor(99);   // an empty batch has no tensors to check
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
