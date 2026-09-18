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
        public List<(Tensor Tensor, Buffer Buffer)> Demoted { get; } = [];
        public List<(Tensor Tensor, Buffer Buffer)> Evicted { get; } = [];
        public bool BlockCallbacks { get; set; }

        /// <summary>Stands in for a backend that promotes a tensor it has seen uploaded twice.</summary>
        public bool PromoteOnSecondUpload { get; set; }

        private readonly Dictionary<Tensor, Buffer> _seen = new(ReferenceEqualityComparer.Instance);

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

        protected override void OnWeightDemoted(Tensor tensor, Buffer buffer) => Demoted.Add((tensor, buffer));

        protected override void OnActivationEvicted(Tensor tensor, Buffer buffer) => Evicted.Add((tensor, buffer));

        protected override bool TryEnterCallback() => !BlockCallbacks;

        protected override void OnTransientUploaded(Buffer buffer, Tensor source)
        {
            if (!PromoteOnSecondUpload)
            {
                return;
            }
            if (_seen.ContainsKey(source))
            {
                PromoteToWeight(source, buffer);
                return;
            }
            _seen[source] = buffer;
        }

        /// <summary>Allocation without an upload, for tests exercising the cache's bookkeeping rather than transfers.</summary>
        public Buffer AllocateForTest(long bytes) => AllocateDevice(bytes);
    }

    private static Tensor NewTensor(int elements = 64) => new(new TensorShape(elements), DType.F32);

    private static long Size(Tensor tensor) => GpuResidencyCache<FakeCache.Buffer>.ByteSize(tensor);

    /// <summary>A buffer displaced by a rebind must survive until the next op, because the op that displaced it may
    /// still be holding it as its own input and will release it in a <c>finally</c> that has not run yet.</summary>
    [Fact]
    public void A_Displaced_Buffer_Is_Not_Freed_Until_The_Next_Op_Sweeps()
    {
        using FakeCache cache = new();
        using Tensor tensor = NewTensor();

        FakeCache.Buffer first = cache.AllocateForTest(Size(tensor));
        cache.CacheActivation(tensor, first, Size(tensor));

        FakeCache.Buffer second = cache.AllocateForTest(Size(tensor));
        cache.CacheActivation(tensor, second, Size(tensor));

        Assert.False(first.Freed);      // the displacing op may still own it

        cache.SweepOrphans();

        Assert.True(first.Freed);
        Assert.False(second.Freed);
        Assert.Same(second, cache.CopyToDevice(tensor));
    }

    /// <summary>An in-place op writes through the buffer without replacing it. Whatever a backend hangs off that
    /// activation describes its contents — a producer-emitted quantized sidecar — so the write stales it just as a
    /// swap would, and the eviction hook has to fire even though nothing is displaced.</summary>
    [Fact]
    public void Rebinding_A_Tensor_To_The_Same_Buffer_Still_Reports_An_Eviction()
    {
        using FakeCache cache = new();
        using Tensor tensor = NewTensor();

        FakeCache.Buffer buffer = cache.AllocateForTest(Size(tensor));
        cache.CacheActivation(tensor, buffer, Size(tensor));
        cache.CacheActivation(tensor, buffer, Size(tensor));

        Assert.Single(cache.Evicted, entry => ReferenceEquals(entry.Buffer, buffer));
        cache.SweepOrphans();
        Assert.False(buffer.Freed);   // still bound, so nothing to reclaim
    }

    /// <summary>The in-place case: the displaced buffer IS the op's own input, so the op's cleanup frees it and the
    /// sweep must not free it again.</summary>
    [Fact]
    public void A_Displaced_Buffer_The_Caller_Releases_Is_Not_Freed_Twice()
    {
        using FakeCache cache = new();
        using Tensor tensor = NewTensor();

        FakeCache.Buffer input = cache.AllocateForTest(Size(tensor));
        cache.CacheActivation(tensor, input, Size(tensor));
        FakeCache.Buffer output = cache.AllocateForTest(Size(tensor));
        cache.CacheActivation(tensor, output, Size(tensor));

        cache.ReleaseIfNotCached(input, Size(tensor));   // the op's own finally
        cache.SweepOrphans();

        Assert.Single(cache.FreedBuffers, buffer => ReferenceEquals(buffer, input));
    }

    /// <summary>An arena owns its allocations wholesale, so a displaced one must never be handed back alone.</summary>
    [Fact]
    public void A_Displaced_Buffer_Owned_Elsewhere_Is_Never_Freed()
    {
        using FakeCache cache = new();
        using Tensor tensor = NewTensor();

        FakeCache.Buffer arena = cache.AllocateForTest(Size(tensor));
        cache.ExternallyOwnedIds.Add(arena.Id);
        cache.CacheActivation(tensor, arena, Size(tensor));
        cache.CacheActivation(tensor, cache.AllocateForTest(Size(tensor)), Size(tensor));

        cache.SweepOrphans();

        Assert.False(arena.Freed);
    }

    /// <summary>Teardown has to reach the parked set too: a parked buffer has already left the owned set, so the
    /// teardown sweep over that set does not see it.</summary>
    [Fact]
    public void Teardown_Frees_A_Buffer_Still_Parked()
    {
        FakeCache cache = new();
        using Tensor tensor = NewTensor();

        FakeCache.Buffer first = cache.AllocateForTest(Size(tensor));
        cache.CacheActivation(tensor, first, Size(tensor));
        cache.CacheActivation(tensor, cache.AllocateForTest(Size(tensor)), Size(tensor));

        cache.Dispose();

        Assert.True(first.Freed);
    }

    /// <summary>The core bug this guards: a resident weight whose tensor an op rebinds must stop being a weight.
    /// CopyToDevice checks weights first, so leaving the entry makes every later read return the PRE-op bytes and
    /// the device write is silently discarded.</summary>
    [Fact]
    public void An_Op_Writing_A_Resident_Weight_Demotes_It()
    {
        using FakeCache cache = new();
        using Tensor weight = NewTensor();

        cache.PreloadWeight(weight);
        FakeCache.Buffer stale = cache.CopyToDevice(weight);

        FakeCache.Buffer written = cache.AllocateForTest(Size(weight));
        cache.CacheActivation(weight, written, Size(weight));

        Assert.Same(written, cache.CopyToDevice(weight));
        Assert.NotSame(stale, cache.CopyToDevice(weight));
        Assert.Single(cache.Demoted, entry => ReferenceEquals(entry.Buffer, stale));
    }

    /// <summary>A demoted weight's cached conversions describe the pre-op contents, so they are stale too.</summary>
    [Fact]
    public void Demoting_A_Weight_Drops_Its_Cached_Conversions()
    {
        using FakeCache cache = new();
        using Tensor weight = NewTensor();

        cache.PreloadWeight(weight);
        FakeCache.Buffer cast = cache.AllocateForTest(16);
        cache.StoreWeightCast(weight, DType.F16, cast, 16);

        cache.CacheActivation(weight, cache.AllocateForTest(Size(weight)), Size(weight));

        Assert.False(cache.TryGetWeightCast(weight, DType.F16, out _));
        Assert.True(cast.Freed);
    }

    /// <summary>A tensor's binding outlives the cache that planted it. The callback must check before it touches
    /// anything — on CUDA this exact path threw during a model swap.</summary>
    [Fact]
    public void A_Callback_Arriving_After_Disposal_Does_Nothing()
    {
        FakeCache cache = new();
        Tensor tensor = NewTensor();

        cache.CacheActivation(tensor, cache.AllocateForTest(Size(tensor)), Size(tensor));
        cache.Dispose();
        int freedAtTeardown = cache.FreedBuffers.Count;

        tensor.Dispose();   // fires the binding planted before disposal

        Assert.Equal(freedAtTeardown, cache.FreedBuffers.Count);
    }

    /// <summary>The gate a backend closes while it is retiring.</summary>
    [Fact]
    public void A_Blocked_Callback_Leaves_The_Cache_Untouched()
    {
        using FakeCache cache = new();
        using Tensor tensor = NewTensor();

        FakeCache.Buffer buffer = cache.AllocateForTest(Size(tensor));
        cache.CacheActivation(tensor, buffer, Size(tensor));
        cache.BlockCallbacks = true;

        unsafe
        {
            _ = tensor.DataPointer;
        }

        Assert.Equal(0, cache.Downloads);
        Assert.False(buffer.Freed);
    }

    /// <summary>Promotion happens behind the caller's back, so host data stays authoritative: a later host write
    /// has to drop the device copy, or every read afterwards is served the pre-write bytes. The same silent
    /// wrong-answer bug as a weight an op writes through, arriving from the host side instead.</summary>
    [Fact]
    public void A_Host_Write_After_Promotion_Drops_The_Device_Copy()
    {
        using FakeCache cache = new();
        using Tensor tensor = NewTensor();
        cache.PromoteOnSecondUpload = true;

        cache.ReleaseIfNotCached(cache.CopyToDevice(tensor), Size(tensor));
        FakeCache.Buffer promoted = cache.CopyToDevice(tensor);
        cache.ReleaseIfNotCached(promoted, Size(tensor));
        Assert.Same(promoted, cache.CopyToDevice(tensor));

        tensor.AsSpan<float>()[0] = 42f;   // host write; funnels through the demotion binding

        Assert.True(promoted.Freed);
        Assert.NotSame(promoted, cache.CopyToDevice(tensor));   // re-uploaded, so it carries the new byte
        Assert.Equal(3, cache.Uploads);
        Assert.Single(cache.Demoted, entry => ReferenceEquals(entry.Buffer, promoted));
    }

    /// <summary>An explicit preload is the caller's decision, so a host read must not silently undo it.</summary>
    [Fact]
    public void A_Host_Read_After_An_Explicit_Preload_Keeps_The_Weight()
    {
        using FakeCache cache = new();
        using Tensor tensor = NewTensor();

        cache.PreloadWeight(tensor);
        FakeCache.Buffer resident = cache.CopyToDevice(tensor);

        unsafe
        {
            _ = tensor.DataPointer;
        }

        Assert.False(resident.Freed);
        Assert.Same(resident, cache.CopyToDevice(tensor));
    }

    /// <summary>Auto-promotion runs entirely through the upload hook: a tensor uploaded twice becomes a resident
    /// weight, and the caller's own release of that buffer is then correctly skipped.</summary>
    [Fact]
    public void A_Twice_Uploaded_Tensor_Can_Be_Promoted_By_The_Upload_Hook()
    {
        using FakeCache cache = new();
        using Tensor tensor = NewTensor();
        cache.PromoteOnSecondUpload = true;

        FakeCache.Buffer first = cache.CopyToDevice(tensor);
        cache.ReleaseIfNotCached(first, Size(tensor));
        FakeCache.Buffer second = cache.CopyToDevice(tensor);
        cache.ReleaseIfNotCached(second, Size(tensor));

        Assert.False(second.Freed);                        // the cache owns it now
        Assert.Same(second, cache.CopyToDevice(tensor));   // and serves it without a third upload
        Assert.Equal(2, cache.Uploads);
    }

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

    /// <summary>A cache over a DIFFERENT buffer type is still a different owner.
    ///
    /// <para>A static field inside a generic class is per closed type, so a counter declared on the generic gives
    /// <c>GpuResidencyCache&lt;ulong&gt;</c> and <c>GpuResidencyCache&lt;VulkanBuffer&gt;</c> one counter each and both
    /// hand out key 1. That is the "every device keyed 0" bug one level up, and it only appears once a second
    /// backend is on the base: a host tensor resident on both — a text encoder on one, a denoiser on the other —
    /// then carries both bindings under the same key, and their finalizer-cleanup buckets collide, so one backend's
    /// drain runs the other's device cleanup.</para></summary>
    [Fact]
    public void Caches_Over_Different_Buffer_Types_Do_Not_Share_Keys()
    {
        using FakeCache handleCache = new();
        using OtherBufferCache structCache = new();

        Assert.NotEqual(handleCache.BindingKey, structCache.BindingKey);
    }

    /// <summary>A second cache type, standing in for another backend's buffer handle.</summary>
    private sealed class OtherBufferCache : GpuResidencyCache<long>
    {
        protected override long AllocateDevice(long bytes) => bytes;
        protected override void FreeDevice(long buffer, long bytes) { }
        protected override void Upload(long destination, Tensor source, long bytes) { }
        protected override void DownloadSynced(nint hostDestination, long source, long bytes) { }
        protected override void MakeCurrent() { }
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
