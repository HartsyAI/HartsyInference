using HartsyInference.Core.Tensors;
using HartsyInference.Gpu;
using Xunit;

namespace HartsyInference.Gpu.Tests;

/// <summary>The residency cache's own behaviour, on a fake device.
///
/// <para>The cache never touches a real API — it decides WHICH tensor should be on the device and when its buffer is
/// released, and delegates the five operations that need one. So the whole of it is testable against a fake buffer,
/// on any machine, with no GPU. That matters: this logic previously existed twice, in two backends whose tests both
/// needed hardware, which is exactly why the two copies drifted without anyone noticing.</para></summary>
public sealed class GpuResidencyCacheTests
{
    /// <summary>A device that records what was asked of it instead of doing it.</summary>
    private sealed class FakeCache : GpuResidencyCache<FakeCache.Buffer>
    {
        internal sealed class Buffer(int id, long bytes)
        {
            public int Id { get; } = id;
            public long Bytes { get; } = bytes;
            public bool Freed { get; set; }
        }

        private int _nextId;

        public List<Buffer> Allocated { get; } = [];
        public List<Buffer> FreedBuffers { get; } = [];
        public int Uploads { get; private set; }
        public int Downloads { get; private set; }
        public bool RefuseHostReads { get; set; }
        public HashSet<int> ExternallyOwnedIds { get; } = [];

        protected override Buffer AllocateDevice(long bytes)
        {
            Buffer buffer = new(++_nextId, bytes);
            Allocated.Add(buffer);
            return buffer;
        }

        protected override void FreeDevice(Buffer buffer, long bytes)
        {
            buffer.Freed = true;
            FreedBuffers.Add(buffer);
        }

        protected override void Upload(Buffer destination, Tensor source, long bytes) => Uploads++;

        protected override void DownloadSynced(nint hostDestination, Buffer source, long bytes) => Downloads++;

        protected override void MakeCurrent() { }

        protected override bool HostReadForbidden => RefuseHostReads;

        protected override bool IsExternallyOwned(Buffer buffer) => ExternallyOwnedIds.Contains(buffer.Id);

        /// <summary>Allocation without an upload, for tests exercising the cache's bookkeeping rather than transfers.</summary>
        public Buffer AllocateForTest(long bytes) => AllocateDevice(bytes);
    }

    private static Tensor NewTensor(int elements = 64) => new(new TensorShape(elements), DType.F32);

    [Fact]
    public void A_Cached_Tensor_Is_Served_Without_Another_Upload()
    {
        using FakeCache cache = new();
        using Tensor tensor = NewTensor();

        FakeCache.Buffer first = cache.CopyToDevice(tensor);
        cache.CacheActivation(tensor, first, GpuResidencyCache<FakeCache.Buffer>.ByteSize(tensor));
        FakeCache.Buffer second = cache.CopyToDevice(tensor);

        Assert.Same(first, second);
        Assert.Equal(1, cache.Uploads);
        Assert.Equal(1, cache.Misses);
        Assert.Equal(1, cache.Hits);
    }

    /// <summary>Reading the tensor from host is what ends its residency: the value comes back and the device copy is
    /// handed over. Without this a GPU-resident tensor would read as whatever its host buffer last held.</summary>
    [Fact]
    public void Reading_A_Resident_Tensor_Brings_It_Back_And_Releases_The_Buffer()
    {
        using FakeCache cache = new();
        using Tensor tensor = NewTensor();
        long bytes = GpuResidencyCache<FakeCache.Buffer>.ByteSize(tensor);

        FakeCache.Buffer buffer = cache.CopyToDevice(tensor);
        cache.CacheActivation(tensor, buffer, bytes);
        Assert.Equal(0, cache.D2hSyncCount);

        unsafe
        {
            _ = tensor.DataPointer;
        }

        Assert.Equal(1, cache.Downloads);
        Assert.Equal(1, cache.D2hSyncCount);
        Assert.True(buffer.Freed);
        Assert.False(cache.TryGetCached(tensor, out _));
    }

    /// <summary>An in-place op re-caches the same tensor with a new buffer. The binding left by the previous call
    /// closes over the OLD buffer, so if it is not cleared first it frees the buffer the tensor now points at.</summary>
    [Fact]
    public void Recaching_A_Tensor_Does_Not_Free_Its_New_Buffer()
    {
        using FakeCache cache = new();
        using Tensor tensor = NewTensor();
        long bytes = GpuResidencyCache<FakeCache.Buffer>.ByteSize(tensor);

        FakeCache.Buffer original = cache.CopyToDevice(tensor);
        cache.CacheActivation(tensor, original, bytes);

        FakeCache.Buffer replacement = cache.AllocateForTest(bytes);
        cache.CacheActivation(tensor, replacement, bytes);

        Assert.True(cache.TryGetCached(tensor, out FakeCache.Buffer? current));
        Assert.Same(replacement, current);
        Assert.False(replacement.Freed);
    }

    /// <summary>Teardown must release the buffer an in-place re-cache orphaned.
    ///
    /// <para>Re-caching overwrites the tensor's dictionary entry, and clearing its binding only nulls the callbacks —
    /// neither releases the buffer that was there before. It is then owned by nothing: unreachable from any cache,
    /// still owned by this instance. Walking the caches at teardown misses it, and on a real backend it survives to
    /// its finalizer, which destroys it against a device that no longer exists. That is a native crash, three layers
    /// from the cause, and it took three attempts to find — this test reproduces it without a GPU.</para></summary>
    [Fact]
    public void Teardown_Releases_A_Buffer_Orphaned_By_Recaching()
    {
        FakeCache cache = new();
        using Tensor tensor = NewTensor();
        long bytes = GpuResidencyCache<FakeCache.Buffer>.ByteSize(tensor);

        FakeCache.Buffer orphaned = cache.AllocateForTest(bytes);
        cache.CacheActivation(tensor, orphaned, bytes);

        FakeCache.Buffer current = cache.AllocateForTest(bytes);
        cache.CacheActivation(tensor, current, bytes);   // the first buffer is now owned by nothing

        cache.FreeAllCached();

        Assert.True(orphaned.Freed, "the re-cached tensor's previous buffer was never released");
        Assert.True(current.Freed);
    }

    [Fact]
    public void A_Preloaded_Weight_Outlives_An_Activation_Sweep()
    {
        using FakeCache cache = new();
        using Tensor weight = NewTensor();
        using Tensor activation = NewTensor();

        cache.PreloadWeight(weight);
        FakeCache.Buffer activationBuffer = cache.CopyToDevice(activation);
        cache.CacheActivation(activation, activationBuffer, GpuResidencyCache<FakeCache.Buffer>.ByteSize(activation));

        cache.OffloadActivations(long.MaxValue);

        Assert.True(cache.TryGetCached(weight, out _));
        Assert.False(cache.TryGetCached(activation, out _));
    }

    /// <summary>Pinning is what keeps cross-step device state put while everything around it is reclaimed.</summary>
    [Fact]
    public void Offload_Skips_A_Pinned_Activation()
    {
        using FakeCache cache = new();
        using Tensor kept = NewTensor();
        using Tensor evictable = NewTensor();
        long bytes = GpuResidencyCache<FakeCache.Buffer>.ByteSize(kept);

        cache.CacheActivation(kept, cache.AllocateForTest(bytes), bytes);
        cache.CacheActivation(evictable, cache.AllocateForTest(bytes), bytes);
        cache.PinActivation(kept);

        long freed = cache.OffloadActivations(long.MaxValue);

        Assert.Equal(bytes, freed);
        Assert.True(cache.TryGetCached(kept, out _));
        Assert.False(cache.TryGetCached(evictable, out _));
    }

    /// <summary>Two caches are two devices. A tensor resident on both must be released by each independently — a
    /// shared key would let one device's teardown drop the other's binding, and the surviving cache would then hand
    /// out a buffer nobody owns.</summary>
    [Fact]
    public void Two_Caches_Hold_Independent_Bindings_On_One_Tensor()
    {
        using FakeCache first = new();
        using FakeCache second = new();
        using Tensor shared = NewTensor();
        long bytes = GpuResidencyCache<FakeCache.Buffer>.ByteSize(shared);

        Assert.NotEqual(first.BindingKey, second.BindingKey);

        first.CacheActivation(shared, first.AllocateForTest(bytes), bytes);
        second.CacheActivation(shared, second.AllocateForTest(bytes), bytes);

        Assert.True(first.TryGetCached(shared, out _));
        Assert.True(second.TryGetCached(shared, out _));

        first.FreeAllCached();

        Assert.False(first.TryGetCached(shared, out _));
        Assert.True(second.TryGetCached(shared, out _));
    }

    /// <summary>A buffer an arena owns must never be freed one at a time; the arena frees it wholesale.</summary>
    [Fact]
    public void An_Externally_Owned_Buffer_Is_Not_Freed_Individually()
    {
        using FakeCache cache = new();
        using Tensor tensor = NewTensor();
        long bytes = GpuResidencyCache<FakeCache.Buffer>.ByteSize(tensor);

        FakeCache.Buffer buffer = cache.AllocateForTest(bytes);
        cache.ExternallyOwnedIds.Add(buffer.Id);
        cache.CacheActivation(tensor, buffer, bytes);

        cache.FreeAllCached();

        Assert.False(buffer.Freed);
        Assert.DoesNotContain(buffer, cache.FreedBuffers);
    }

    /// <summary>A host read while a graph is being recorded is a contract violation, and servicing it would drain a
    /// queue that is mid-record. It has to fail loudly rather than quietly do the dangerous thing.</summary>
    [Fact]
    public void A_Host_Read_During_Capture_Throws()
    {
        using FakeCache cache = new();
        using Tensor tensor = NewTensor();
        long bytes = GpuResidencyCache<FakeCache.Buffer>.ByteSize(tensor);

        cache.CacheActivation(tensor, cache.AllocateForTest(bytes), bytes);
        cache.RefuseHostReads = true;

        Assert.ThrowsAny<Exception>(() =>
        {
            unsafe
            {
                _ = tensor.DataPointer;
            }
        });
    }

    [Fact]
    public void Disposing_Releases_Everything_It_Still_Holds()
    {
        FakeCache cache = new();
        using Tensor weight = NewTensor();
        using Tensor activation = NewTensor();
        long bytes = GpuResidencyCache<FakeCache.Buffer>.ByteSize(weight);

        cache.PreloadWeight(weight);
        cache.CacheActivation(activation, cache.AllocateForTest(bytes), bytes);

        cache.Dispose();

        Assert.All(cache.Allocated, buffer => Assert.True(buffer.Freed));
        cache.Dispose();   // idempotent
    }
}
