using System.Collections.Concurrent;
using System.Runtime.ExceptionServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Memory;

namespace HartsyInference.Core.Tensors;

/// <summary>Core tensor type holding a pointer to unmanaged memory, shape, dtype, and device.</summary>
public sealed unsafe class Tensor : IDisposable
{
    private NativeBuffer? _ownedBuffer;
    private nint _dataPointer;

    /// <summary>Optional owner object kept alive for the lifetime of this tensor.</summary>
    /// <remarks>For tensors that borrow an external pointer (e.g. an mmap'd weight file), this roots the owner
    /// (e.g. the <c>MmapHandle</c>) so its finalizer can't run — and unmap the memory out from under the borrowed
    /// pointer — while the tensor is still reachable. Without this, a loader whose tensors outlive the loader
    /// reference itself (a common pattern when a helper returns only the tensor dictionary) would dangle the
    /// moment a GC collects the loader.</remarks>
    private object? _keepAlive;

    /// <summary>Byte size to lazily allocate on first CPU access (see <see cref="EnsureHostBuffer"/>); zero for borrowed tensors.</summary>
    private readonly long _byteSize;

    /// <summary>True when this tensor owns a lazily allocated host buffer (see <see cref="OwnsMemory"/>); false for borrowed pointers.</summary>
    /// <remarks>Drives <see cref="OwnsMemory"/> and the lazy-allocation path.</remarks>
    private readonly bool _ownsLazy;

    /// <summary>Set once on Dispose so the lazy host-buffer path can't resurrect freed memory.</summary>
    private bool _disposed;

    /// <summary>Backend-set callback: copies GPU→CPU then frees GPU pointer. Invoked lazily on first CPU data access.</summary>
    /// <remarks>Single-owner fast slot. Backends may null this directly ONLY for tensors they created themselves
    /// (activations, graph buffers) — those have exactly one owner. Multi-owner tensors (a shared host weight
    /// promoted on two backends) overflow into <see cref="_extraGpuBindings"/> via <see cref="SetGpuBinding"/>,
    /// which is the only correct plant path for cache/promotion machinery.</remarks>
    internal Action? _gpuSyncCallback;

    /// <summary>Backend-set callback: frees GPU pointer without D2H copy. Invoked on Dispose when GPU data was never synced to CPU.</summary>
    internal Action? _gpuDisposeCallback;

    /// <summary>Key identifying which backend context owns <see cref="_gpuDisposeCallback"/>.</summary>
    /// <remarks>The CUDA primary-context handle (one per device) the callback frees against. Set by the backend
    /// when it plants the callback; 0 for non-CUDA/context-less callbacks. Used to route this tensor's finalizer
    /// cleanup into the owning context's queue so the draining thread only ever touches its own backend's state
    /// (see <see cref="PendingFinalizerGpuCleanup"/>).</remarks>
    internal nint _gpuCleanupContext;

    /// <summary>Bindings past the first when MORE THAN ONE backend holds a device copy of this host tensor (shared
    /// model caches — Redux/TAESD encoders, audio converters — promoted on two GPUs). Before this existed the
    /// second backend's promotion overwrote the first's demote hook in the single slot above: the first backend's
    /// device copy then never demoted on host mutation and its cleanup routed into the wrong context bucket.
    /// Null in the overwhelmingly common single-owner case. Guarded by <c>lock (this)</c> — internal type, no
    /// external lockers, and a dedicated lock object would cost 8 bytes on every tensor for a rare case.</summary>
    private List<GpuBinding>? _extraGpuBindings;

    /// <summary>One backend's (sync, dispose, owner-key) triple for <see cref="_extraGpuBindings"/>.</summary>
    private readonly record struct GpuBinding(nint Key, Action Sync, Action Dispose);

    /// <summary>Registers backend callbacks for this tensor's device copy, keyed by the owning backend's cleanup
    /// key. First owner takes the fast single-slot fields; a DIFFERENT owner lands in the overflow list instead of
    /// overwriting; a repeat plant by the same owner replaces its own entry (the pre-existing semantics).</summary>
    internal void SetGpuBinding(nint key, Action sync, Action dispose)
    {
        bool disposeImmediately = false;
        lock (this)
        {
            if (_disposed)
            {
                disposeImmediately = true;
            }
            else if ((_gpuSyncCallback is null && _gpuDisposeCallback is null) || _gpuCleanupContext == key)
            {
                _gpuCleanupContext = key;
                _gpuSyncCallback = sync;
                _gpuDisposeCallback = dispose;
            }
            else
            {
                List<GpuBinding> extras = _extraGpuBindings ??= [];
                for (int i = 0; i < extras.Count; i++)
                {
                    if (extras[i].Key == key)
                    {
                        extras[i] = new GpuBinding(key, sync, dispose);
                        return;
                    }
                }
                extras.Add(new GpuBinding(key, sync, dispose));
            }
        }
        // Backends publish their cache entry before planting this binding. If disposal won the race, invoke the
        // supplied rollback outside the tensor lock instead of throwing and stranding that new device allocation.
        if (disposeImmediately)
            dispose();
    }

    /// <summary>Removes ONLY <paramref name="key"/>'s binding, leaving other backends' bindings intact — a bulk
    /// eviction on one backend must not strip a sibling's demote hook. Promotes the first overflow entry into the
    /// fast slot so the single-owner invariant is restored.</summary>
    internal void ClearGpuBinding(nint key)
    {
        lock (this)
        {
            List<GpuBinding>? extras = _extraGpuBindings;
            if (_gpuCleanupContext == key && (_gpuSyncCallback is not null || _gpuDisposeCallback is not null))
            {
                _gpuSyncCallback = null;
                _gpuDisposeCallback = null;
                _gpuCleanupContext = 0;
                if (extras is not null && extras.Count > 0)
                {
                    GpuBinding hoisted = extras[0];
                    extras.RemoveAt(0);
                    _gpuCleanupContext = hoisted.Key;
                    _gpuSyncCallback = hoisted.Sync;
                    _gpuDisposeCallback = hoisted.Dispose;
                }
                return;
            }
            if (extras is null)
            {
                return;
            }
            for (int i = 0; i < extras.Count; i++)
            {
                if (extras[i].Key == key)
                {
                    extras.RemoveAt(i);
                    return;
                }
            }
        }
    }

    /// <summary>Snapshots and empties the overflow bindings; null when there were none (the common case).</summary>
    private GpuBinding[]? TakeExtraBindings()
    {
        List<GpuBinding>? extras = _extraGpuBindings;
        if (extras is null)
        {
            return null;
        }
        lock (this)
        {
            if (extras.Count == 0)
            {
                return null;
            }
            GpuBinding[] snapshot = [.. extras];
            extras.Clear();
            return snapshot;
        }
    }

    /// <summary>Claims (and clears) the primary-slot sync callback under the same lock <see cref="SetGpuBinding"/>/
    /// <see cref="ClearGpuBinding"/> use, then returns it for the caller to invoke OUTSIDE the lock (mirrors
    /// <see cref="TakeExtraBindings"/> twelve lines up). Without this, two threads racing <c>EnsureCpuData</c> on a
    /// tensor shared by two backends (a weight promoted on both, mid-CFG-parallel-step) could both read the same
    /// non-null callback before either clears it and both invoke it — a double D2H-sync / double-free of the same
    /// GPU pointer. The unlocked null-check first keeps the overwhelmingly common no-GPU-shadow case a plain field
    /// read.</summary>
    private Action? ClaimPrimarySync()
    {
        if (_gpuSyncCallback is null)
        {
            return null;
        }
        lock (this)
        {
            Action? sync = _gpuSyncCallback;
            if (sync is null)
            {
                return null;
            }
            _gpuSyncCallback = null;
            _gpuDisposeCallback = null;
            _gpuCleanupContext = 0;
            return sync;
        }
    }

    /// <summary>GPU cleanup callbacks from finalizer-reclaimed tensors (never explicitly Disposed), queued rather than invoked inline.</summary>
    /// <remarks>The finalizer thread has no business making CUDA driver calls (stream enqueue order across threads
    /// is a race even though individual driver calls are thread-safe): a free enqueued from the finalizer thread
    /// can land between two dependent ops the inference thread assumed were adjacent in the stream, corrupting an
    /// in-flight buffer. Drained by the backend on the thread that actually owns the stream (see
    /// <c>CudaContext.EnsureCurrent</c>) so every GPU driver call still comes from one thread.
    ///
    /// <para>Partitioned by process-unique owning-backend key (not one global queue): a callback frees against, and
    /// mutates the unsynchronized State of, exactly one backend. With multiple GPU backends live in one process,
    /// draining every backend's callbacks from any thread let backend B's inference thread mutate backend A's
    /// non-thread-safe state while A's own thread did the same — a data race that leaked VRAM / threw / corrupted
    /// under concurrent multi-GPU load. Keying by backend registration means each thread drains only its own
    /// backend's bucket. Same-device CUDA backends still have distinct keys; bucket 0 holds context-less callbacks
    /// (e.g. Vulkan) and is drained only by the drain-all overload.</para></remarks>
    internal static readonly ConcurrentDictionary<nint, ConcurrentQueue<Action>> PendingFinalizerGpuCleanup = new();
    private static readonly ConcurrentDictionary<nint, byte> RetiredFinalizerGpuCleanupKeys = new();

    /// <summary>Queues a finalizer GPU-cleanup callback under its process-unique owning-backend bucket.</summary>
    internal static void EnqueueFinalizerGpuCleanup(nint ownerKey, Action cleanup)
    {
        if (RetiredFinalizerGpuCleanupKeys.ContainsKey(ownerKey)) return;
        ConcurrentQueue<Action> queue = PendingFinalizerGpuCleanup.GetOrAdd(
            ownerKey, static _ => new ConcurrentQueue<Action>());
        // Close the race with backend retirement: if retirement removed the published bucket while this enqueue
        // held a local reference, dropping the callback is correct (the backend already freed its entire cache).
        if (RetiredFinalizerGpuCleanupKeys.ContainsKey(ownerKey))
        {
            PendingFinalizerGpuCleanup.TryRemove(ownerKey, out _);
            return;
        }
        queue.Enqueue(cleanup);
    }

    /// <summary>Drains and invokes the finalizer GPU-cleanup callbacks owned by <paramref name="ownerKey"/> only.</summary>
    /// <remarks>Call from the thread that owns that backend's CUDA context/stream — it touches no other backend's state.</remarks>
    public static void DrainPendingFinalizerGpuCleanup(nint ownerKey)
    {
        if (PendingFinalizerGpuCleanup.TryGetValue(ownerKey, out ConcurrentQueue<Action>? queue))
        {
            while (queue.TryDequeue(out Action? cleanup))
            {
                InvokeFinalizerGpuCleanup(cleanup);
            }
        }
    }

    /// <summary>Drains every backend's finalizer GPU-cleanup callbacks, regardless of owner.</summary>
    /// <remarks>Only safe when the caller can service all live backends (single-backend / shutdown paths).
    /// Concurrent multi-GPU inference must use the keyed overload so a thread never runs another backend's cleanup.</remarks>
    public static void DrainPendingFinalizerGpuCleanup()
    {
        foreach (ConcurrentQueue<Action> queue in PendingFinalizerGpuCleanup.Values)
        {
            while (queue.TryDequeue(out Action? cleanup))
            {
                InvokeFinalizerGpuCleanup(cleanup);
            }
        }
    }

    /// <summary>Drops (without invoking) all queued finalizer GPU-cleanup callbacks for <paramref name="ownerKey"/>.</summary>
    /// <remarks>Call at backend teardown AFTER the backend has freed all its cached device memory: the queued
    /// callbacks can only reference caches that were just emptied, so running them is a no-op at best — and the
    /// CUDA driver reuses primary-context handles, so leaving them queued makes the NEXT backend on the same
    /// device drain and execute another backend's stale callbacks during its own construction (the GGUF
    /// model-switch NRE).</remarks>
    public static void DiscardPendingFinalizerGpuCleanup(nint ownerKey)
    {
        PendingFinalizerGpuCleanup.TryRemove(ownerKey, out _);
    }

    /// <summary>Permanently retires a process-unique backend key, dropping its current bucket and suppressing any
    /// late tensor finalizer that races backend teardown from recreating an undrainable static queue.</summary>
    internal static void RetireFinalizerGpuCleanup(nint ownerKey)
    {
        RetiredFinalizerGpuCleanupKeys[ownerKey] = 0;
        PendingFinalizerGpuCleanup.TryRemove(ownerKey, out _);
    }

    /// <summary>Runs one queued cleanup, containing any failure so it can't take down an unrelated op.</summary>
    /// <remarks>A queued callback can be arbitrarily stale (its backend torn down, its object graph resurrected
    /// post-finalization), and the drain runs inside whatever innocent op happened to bind the context next — one
    /// throwing callback must not take that op (or a whole backend construction) down with it.</remarks>
    private static void InvokeFinalizerGpuCleanup(Action cleanup)
    {
        try
        {
            cleanup();
        }
        catch (Exception ex)
        {
            Logging.Logs.Error("Finalizer GPU-cleanup callback failed (stale backend state?); continuing drain.", ex);
        }
    }

    /// <summary>Creates a new owned tensor; the host buffer is NOT allocated here, only lazily on first CPU access.</summary>
    /// <remarks>Allocated (zeroed) lazily via <see cref="DataPointer"/>/<see cref="AsSpan{T}"/>/etc. GPU-resident
    /// activations (whose data lives on the device and is freed without ever being read on the host) therefore
    /// never pay for a host malloc+memset.</remarks>
    public Tensor(TensorShape shape, DType dtype, DeviceKind device = default)
    {
        Shape = shape;
        DType = dtype;
        Device = device;
        _byteSize = dtype.ComputeByteCount(shape.ElementCount);
        _ownsLazy = true;
    }

    /// <summary>Creates a tensor that borrows memory from an external pointer (e.g., mmap'd weights).</summary>
    public Tensor(void* dataPointer, TensorShape shape, DType dtype, DeviceKind device = default)
    {
        _dataPointer = (nint)dataPointer;
        Shape = shape;
        DType = dtype;
        Device = device;
        _byteSize = dtype.ComputeByteCount(shape.ElementCount);
        _ownedBuffer = null;
        _ownsLazy = false;
    }

    /// <summary>Roots <paramref name="owner"/> for this tensor's lifetime (see <see cref="_keepAlive"/>). Used by
    /// borrowing loaders to pin the backing store (e.g. an <c>MmapHandle</c>) against premature GC/finalization.</summary>
    internal void SetKeepAlive(object? owner) => _keepAlive = owner;

    /// <summary>The owner object (if any) kept alive for this tensor's lifetime. See <see cref="SetKeepAlive"/>.</summary>
    internal object? KeepAliveOwner => _keepAlive;

    /// <summary>
    /// Returns whether this tensor and <paramref name="other"/> have overlapping materialized host-storage
    /// ranges, without allocating a host buffer or invoking either tensor's GPU sync callback. Object identity is
    /// always an alias; two distinct lazy GPU-only tensors are not, because backends cache their device storage by
    /// tensor identity. Reshapes and borrowed subranges already have host pointers, so their overlap is detected.
    /// </summary>
    /// <remarks>
    /// Callers must validate both byte counts before using this helper. Keeping the check here is important: no
    /// backend contract should read <see cref="DataPointer"/> merely to validate aliasing, since that would turn a
    /// device-resident in-place operation into an accidental D2H synchronization.
    /// </remarks>
    internal bool HasOverlappingHostStorageWithoutSync(Tensor other)
    {
        if (ReferenceEquals(this, other))
            return true;

        nint thisPointer = Volatile.Read(ref _dataPointer);
        nint otherPointer = Volatile.Read(ref other._dataPointer);
        if (thisPointer == 0 || otherPointer == 0 || _byteSize <= 0 || other._byteSize <= 0)
            return false;

        nuint thisStart = (nuint)thisPointer;
        nuint otherStart = (nuint)otherPointer;
        nuint thisLength = checked((nuint)_byteSize);
        nuint otherLength = checked((nuint)other._byteSize);
        return thisStart <= otherStart
            ? otherStart - thisStart < thisLength
            : thisStart - otherStart < otherLength;
    }

    /// <summary>Shape and strides of this tensor.</summary>
    public TensorShape Shape { get; }

    /// <summary>Data type of the tensor elements.</summary>
    public DType DType { get; }

    /// <summary>Device this tensor resides on.</summary>
    public DeviceKind Device { get; }

    /// <summary>Total number of elements across all dimensions.</summary>
    public long ElementCount => Shape.ElementCount;

    /// <summary>Whether this tensor owns its memory or borrows it from a mmap/view.</summary>
    /// <remarks>True even before the lazy host buffer is allocated, since it reflects how the tensor was constructed.</remarks>
    public bool OwnsMemory => _ownsLazy;

    /// <summary>Ensures the owned host buffer exists (zeroed on first call) and returns its pointer; borrowed tensors return as-is.</summary>
    /// <remarks>This is the single lazy-allocation chokepoint: GPU activations that are never read on the host
    /// skip it entirely and so never allocate host memory. The CUDA D2H sync callback also calls this to obtain a
    /// destination buffer at sync time.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void* EnsureHostBuffer()
    {
        nint ptr = _dataPointer;
        if (ptr != 0)
            return (void*)ptr;
        if (_disposed || !_ownsLazy)
            throw new ObjectDisposedException(nameof(Tensor));
        NativeBuffer buffer = new NativeBuffer((nuint)_byteSize);
        _ownedBuffer = buffer;
        ptr = (nint)buffer.Pointer;
        _dataPointer = ptr;
        return (void*)ptr;
    }

    /// <summary>Per-tensor scale for ComfyUI-style <c>fp8_scaled</c> quantization: real value = <c>fp8_byte_decoded * Fp8ScaleFactor</c>.</summary>
    /// <remarks>Default 1.0 means no extra scaling. Non-1 values are folded into cuBLAS' <c>alpha</c> at GEMM
    /// call sites so scaling happens for free during matmul.</remarks>
    public float Fp8ScaleFactor { get; set; } = 1.0f;

    /// <summary>Static activation dequant scale shipped alongside an fp8 weight (<c>.input_scale</c>); 0 = absent.</summary>
    /// <remarks>When present the backend quantizes this Linear's activation with a constant instead of computing a
    /// per-call absmax, which removes a full activation read and a grid-wide reduction barrier before every GEMM.
    /// Same units as <see cref="Fp8ScaleFactor"/>: real value = <c>fp8_byte_decoded * scale</c>.</remarks>
    public float Fp8InputScaleFactor { get; set; }

    /// <summary>Pointer to the raw tensor data. If GPU data is cached, triggers a lazy sync (D2H copy) first; otherwise
    /// the owned host buffer is allocated (zeroed) on first access.</summary>
    public void* DataPointer
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            EnsureCpuData();
            return EnsureHostBuffer();
        }
    }

    /// <summary>Returns a span over the tensor data interpreted as <typeparamref name="T"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> AsSpan<T>() where T : unmanaged
    {
        EnsureCpuData();
        void* ptr = EnsureHostBuffer();
        int count = (int)(DType.ComputeByteCount(Shape.ElementCount) / sizeof(T));
        return new Span<T>(ptr, count);
    }

    /// <summary>Returns a read-only span over the tensor data.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> AsReadOnlySpan<T>() where T : unmanaged
    {
        EnsureCpuData();
        void* ptr = EnsureHostBuffer();
        int count = (int)(DType.ComputeByteCount(Shape.ElementCount) / sizeof(T));
        return new ReadOnlySpan<T>(ptr, count);
    }

    /// <summary>Creates a zero-alloc TensorRef view of this tensor for use in kernel implementations.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TensorRef AsRef()
    {
        EnsureCpuData();
        nint ptr = (nint)EnsureHostBuffer();
        return new TensorRef(ptr, Shape, DType, Device);
    }

    /// <summary>Creates a view with a different shape but the same underlying data — no copy.</summary>
    /// <remarks>The view roots this tensor (see <see cref="_keepAlive"/>): without that, reshaping a temporary
    /// (e.g. an <c>EnsureF32</c> cast held by nothing else) left the view dangling once GC finalized the parent
    /// and freed its buffer — a GC-timing-dependent AccessViolation observed in the Qwen3-TTS vocoder after ~9 min
    /// of generation.</remarks>
    public Tensor Reshape(TensorShape newShape)
    {
        void* ptr = DataPointer;

        if (newShape.ElementCount != Shape.ElementCount)
            throw new HartsyInferenceException(
                $"Cannot reshape {Shape} ({Shape.ElementCount} elements) to {newShape} ({newShape.ElementCount} elements).");

        Tensor view = new(ptr, newShape, DType, Device);
        view.SetKeepAlive(this);
        return view;
    }

    /// <summary>Creates a contiguous copy on the specified device. Cross-device requires IBackend.CopyTo.</summary>
    public Tensor To(DeviceKind targetDevice)
    {
        void* ptr = DataPointer;

        if (targetDevice == Device)
        {
            Tensor copy = new Tensor(Shape, DType, Device);
            long byteSize = DType.ComputeByteCount(Shape.ElementCount);
            Buffer.MemoryCopy(ptr, copy.DataPointer, byteSize, byteSize);
            // A byte-identical copy of an fp8_scaled tensor is still scaled — dropping the factor here would
            // silently rescale the weight by 1/scale at the next GEMM.
            copy.Fp8ScaleFactor = Fp8ScaleFactor;
            return copy;
        }

        throw new HartsyInferenceException(
            $"Direct tensor copy from {Device} to {targetDevice} is not supported. Use the destination backend's " +
            "CopyFromPeer(dst, src, srcBackend) for cross-device transfers (direct peer copy when available, " +
            "host-staged otherwise), or CopyTo for host<->device moves within one backend.");
    }

    /// <summary>Element count at or above which elementwise dtype conversion is split across cores. Checkpoint load runs
    /// thousands of tiny casts (biases, norms, scalars) where the <see cref="Parallel.For"/> dispatch — measured at 22µs
    /// for 16 chunks on a 16-core box — costs more than the whole conversion. At 2^20 even the cheapest conversion here
    /// (BF16→F32) is ~500µs serial, so the dispatch is under 5% and the win is 4-6×; below the threshold the code path is
    /// byte-for-byte the pre-existing serial loop.</summary>
    private const long ParallelCastMinElements = 1L << 20;

    /// <summary>Minimum elements per chunk, so a barely-over-threshold cast fans out to a few fat slices rather than
    /// <see cref="Environment.ProcessorCount"/> slivers that cost more to dispatch than to run.</summary>
    private const long ParallelCastChunkElements = 1L << 18;

    /// <summary>Splits <paramref name="count"/> elements into per-core chunks and invokes <paramref name="body"/>(start, length)
    /// on each.</summary>
    /// <remarks>Every elementwise conversion below is a pure <c>dst[i] = f(src[i])</c> with no loop-carried state, so
    /// chunking produces bit-identical output to the serial loop. The <see cref="AggregateException"/> unwrap preserves the
    /// serial contract that an unsupported dtype pair surfaces as <c>HartsyInferenceException</c>.</remarks>
    private static void ParallelChunks(long count, Action<long, long> body)
    {
        int chunks = (int)Math.Min(Environment.ProcessorCount, Math.Max(1L, count / ParallelCastChunkElements));
        if (chunks <= 1)
        {
            body(0, count);
            return;
        }
        long perChunk = (count + chunks - 1) / chunks;
        try
        {
            Parallel.For(0, chunks, c =>
            {
                long start = c * perChunk;
                long length = Math.Min(perChunk, count - start);
                if (length > 0)
                    body(start, length);
            });
        }
        catch (AggregateException ex) when (ex.InnerException is not null)
        {
            ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
        }
    }

    /// <summary>Creates a copy cast to the specified dtype. Quantized types require GgufDequantizer.</summary>
    public Tensor CastTo(DType targetDtype)
    {
        void* ptr = DataPointer;

        if (targetDtype == DType)
            return To(Device);

        if (DType.IsQuantized || targetDtype.IsQuantized)
            throw new HartsyInferenceException(
                $"Quantized dtype conversion ({DType} → {targetDtype}) requires a dedicated dequantizer. Use GgufDequantizer instead.");

        Tensor result = new Tensor(Shape, targetDtype, Device);
        try
        {
            long count = Shape.ElementCount;
            // Dequantizing an fp8_scaled tensor must fold the per-tensor scale into the values: the fp8 bytes
            // alone are `real_value / scale`. The result carries factor 1.0 (already applied), so downstream
            // GEMM alpha logic sees an ordinary unscaled tensor. Scale is 1.0 for everything non-fp8_scaled,
            // making the multiply a no-op on plain checkpoints.
            float fp8Scale = Fp8ScaleFactor;
            void* dst = result.DataPointer;

            if (count >= ParallelCastMinElements)
            {
                // Both pointers are resolved above: DataPointer can lazily allocate the host buffer, which must not
                // race across chunk threads. Strides are element sizes — safe here because quantized pairs already threw.
                DType from = DType, to = targetDtype;
                nint srcBase = (nint)ptr, dstBase = (nint)dst;
                long srcStride = from.SizeInBytes, dstStride = to.SizeInBytes;
                ParallelChunks(count, (start, length) => ConvertRange(from, to,
                    (void*)(srcBase + (nint)(start * srcStride)), (void*)(dstBase + (nint)(start * dstStride)), length, fp8Scale));
            }
            else
            {
                ConvertRange(DType, targetDtype, ptr, dst, count, fp8Scale);
            }

            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
    }

    /// <summary>Converts <paramref name="count"/> elements from <paramref name="srcPtr"/> into <paramref name="dstPtr"/>.</summary>
    /// <param name="fp8Scale">Per-tensor fp8_scaled factor folded into the value when decoding an FP8 source.</param>
    private static void ConvertRange(DType from, DType to, void* srcPtr, void* dstPtr, long count, float fp8Scale)
    {
        if (from == DType.F32 && to == DType.F16)
        {
            ReadOnlySpan<float> src = new ReadOnlySpan<float>(srcPtr, (int)count);
            Span<Half> dst = new Span<Half>(dstPtr, (int)count);
            for (int i = 0; i < (int)count; i++)
                dst[i] = (Half)src[i];
        }
        else if (from == DType.F16 && to == DType.F32)
        {
            ReadOnlySpan<Half> src = new ReadOnlySpan<Half>(srcPtr, (int)count);
            Span<float> dst = new Span<float>(dstPtr, (int)count);
            for (int i = 0; i < (int)count; i++)
                dst[i] = (float)src[i];
        }
        else if (from == DType.F32 && to == DType.BF16)
        {
            ReadOnlySpan<float> src = new ReadOnlySpan<float>(srcPtr, (int)count);
            Span<ushort> dst = new Span<ushort>(dstPtr, (int)count);
            for (int i = 0; i < (int)count; i++)
            {
                // BF16: truncate lower 16 bits of F32
                uint bits = BitConverter.SingleToUInt32Bits(src[i]);
                dst[i] = (ushort)(bits >> 16);
            }
        }
        else if (from == DType.BF16 && to == DType.F32)
        {
            ReadOnlySpan<ushort> src = new ReadOnlySpan<ushort>(srcPtr, (int)count);
            Span<float> dst = new Span<float>(dstPtr, (int)count);
            for (int i = 0; i < (int)count; i++)
                dst[i] = BitConverter.UInt32BitsToSingle((uint)src[i] << 16);
        }
        else if (from == DType.F8E4M3 && to == DType.F32)
        {
            ReadOnlySpan<byte> src = new ReadOnlySpan<byte>(srcPtr, (int)count);
            Span<float> dst = new Span<float>(dstPtr, (int)count);
            for (int i = 0; i < (int)count; i++)
                dst[i] = Fp8E4M3ToFloat(src[i]) * fp8Scale;
        }
        else if (from == DType.F32 && to == DType.F8E4M3)
        {
            ReadOnlySpan<float> src = new ReadOnlySpan<float>(srcPtr, (int)count);
            Span<byte> dst = new Span<byte>(dstPtr, (int)count);
            for (int i = 0; i < (int)count; i++)
                dst[i] = FloatToFp8E4M3(src[i]);
        }
        else if (from == DType.F8E4M3 && to == DType.F16)
        {
            ReadOnlySpan<byte> src = new ReadOnlySpan<byte>(srcPtr, (int)count);
            Span<Half> dst = new Span<Half>(dstPtr, (int)count);
            for (int i = 0; i < (int)count; i++)
                dst[i] = (Half)(Fp8E4M3ToFloat(src[i]) * fp8Scale);
        }
        else if (from == DType.F16 && to == DType.F8E4M3)
        {
            ReadOnlySpan<Half> src = new ReadOnlySpan<Half>(srcPtr, (int)count);
            Span<byte> dst = new Span<byte>(dstPtr, (int)count);
            for (int i = 0; i < (int)count; i++)
                dst[i] = FloatToFp8E4M3((float)src[i]);
        }
        else if (from == DType.F8E5M2 && to == DType.F32)
        {
            ReadOnlySpan<byte> src = new ReadOnlySpan<byte>(srcPtr, (int)count);
            Span<float> dst = new Span<float>(dstPtr, (int)count);
            for (int i = 0; i < (int)count; i++)
                dst[i] = Fp8E5M2ToFloat(src[i]) * fp8Scale;
        }
        else if (from == DType.F32 && to == DType.F8E5M2)
        {
            ReadOnlySpan<float> src = new ReadOnlySpan<float>(srcPtr, (int)count);
            Span<byte> dst = new Span<byte>(dstPtr, (int)count);
            for (int i = 0; i < (int)count; i++)
                dst[i] = FloatToFp8E5M2(src[i]);
        }
        else if (from == DType.BF16 && to == DType.F8E4M3)
        {
            ReadOnlySpan<ushort> src = new ReadOnlySpan<ushort>(srcPtr, (int)count);
            Span<byte> dst = new Span<byte>(dstPtr, (int)count);
            for (int i = 0; i < (int)count; i++)
                dst[i] = FloatToFp8E4M3(BitConverter.UInt32BitsToSingle((uint)src[i] << 16));
        }
        else if (from == DType.BF16 && to == DType.F16)
        {
            // BF16 has F32-range exponent; values |x| > 65504 saturate to ±Inf in F16,
            // matching the standard half cast.
            ReadOnlySpan<ushort> src = new ReadOnlySpan<ushort>(srcPtr, (int)count);
            Span<Half> dst = new Span<Half>(dstPtr, (int)count);
            for (int i = 0; i < (int)count; i++)
                dst[i] = (Half)BitConverter.UInt32BitsToSingle((uint)src[i] << 16);
        }
        else if (from == DType.F16 && to == DType.BF16)
        {
            ReadOnlySpan<Half> src = new ReadOnlySpan<Half>(srcPtr, (int)count);
            Span<ushort> dst = new Span<ushort>(dstPtr, (int)count);
            for (int i = 0; i < (int)count; i++)
            {
                uint bits = BitConverter.SingleToUInt32Bits((float)src[i]);
                dst[i] = (ushort)(bits >> 16);
            }
        }
        else if (from == DType.F8E4M3 && to == DType.BF16)
        {
            ReadOnlySpan<byte> src = new ReadOnlySpan<byte>(srcPtr, (int)count);
            Span<ushort> dst = new Span<ushort>(dstPtr, (int)count);
            for (int i = 0; i < (int)count; i++)
            {
                float f = Fp8E4M3ToFloat(src[i]) * fp8Scale;
                dst[i] = (ushort)(BitConverter.SingleToUInt32Bits(f) >> 16);
            }
        }
        else if (from == DType.F64 && to == DType.F32)
        {
            // PyTorch pickle checkpoints occasionally store parameters as float64 (e.g. LDA heads).
            ReadOnlySpan<double> src = new ReadOnlySpan<double>(srcPtr, (int)count);
            Span<float> dst = new Span<float>(dstPtr, (int)count);
            for (int i = 0; i < (int)count; i++)
                dst[i] = (float)src[i];
        }
        else
        {
            throw new HartsyInferenceException(
                $"Unsupported dtype conversion: {from} → {to}.");
        }
    }

    /// <summary>Fused dequant: FP8-E4M3 → F16, multiplying each value by a per-tensor F32 scalar.</summary>
    /// <remarks>Used for ComfyUI-style fp8_scaled checkpoints (e.g. flux1-krea-dev_fp8_scaled) where
    /// real_weight = fp8_value * scale_weight. Single-pass to avoid an F32 intermediate, halving peak memory
    /// during checkpoint load.</remarks>
    public unsafe Tensor DequantFp8E4M3ScaledToF16(float scale)
    {
        if (DType != DType.F8E4M3)
            throw new HartsyInferenceException(
                $"DequantFp8E4M3ScaledToF16 requires FP8 E4M3 input, got {DType}");

        Tensor result = new Tensor(Shape, DType.F16, Device);
        try
        {
            long count = Shape.ElementCount;
            nint srcBase = (nint)DataPointer, dstBase = (nint)result.DataPointer;
            if (count >= ParallelCastMinElements)
            {
                ParallelChunks(count, (start, length) =>
                    DequantFp8E4M3ScaledToF16Range((byte*)(srcBase + (nint)start), (Half*)(dstBase + (nint)(start * 2)), length, scale));
            }
            else
            {
                DequantFp8E4M3ScaledToF16Range((byte*)srcBase, (Half*)dstBase, count, scale);
            }
            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
    }

    private static void DequantFp8E4M3ScaledToF16Range(byte* srcPtr, Half* dstPtr, long count, float scale)
    {
        ReadOnlySpan<byte> src = new ReadOnlySpan<byte>(srcPtr, (int)count);
        Span<Half> dst = new Span<Half>(dstPtr, (int)count);
        for (int i = 0; i < (int)count; i++)
            dst[i] = (Half)(Fp8E4M3ToFloat(src[i]) * scale);
    }

    /// <summary>Computes the total byte size for a tensor of the given shape and dtype.</summary>
    public static long ComputeByteSize(TensorShape shape, DType dtype) => dtype.ComputeByteCount(shape.ElementCount);

    /// <summary>If GPU data is cached on this tensor, syncs it to CPU and clears the callbacks — for EVERY owning
    /// backend: host access is about to read (or mutate) the host buffer, so each backend's device copy must sync
    /// or demote, not just the first owner's.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void EnsureCpuData()
    {
        Action? sync = ClaimPrimarySync();
        sync?.Invoke();
        GpuBinding[]? extras = TakeExtraBindings();
        if (extras is not null)
        {
            foreach (GpuBinding binding in extras)
            {
                binding.Sync();
            }
        }
    }

    // E4M3FN: sign(1) + exponent(4) + mantissa(3), bias=7, no NaN/Inf, max=448, min_subnormal=2^-9
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Fp8E4M3ToFloat(byte fp8)
    {
        uint sign = (uint)(fp8 >> 7) & 1;
        uint exp = (uint)(fp8 >> 3) & 0xF;
        uint mant = (uint)(fp8 & 0x7);

        if (exp == 0 && mant == 0)
            return sign != 0 ? -0.0f : 0.0f;

        float value;
        if (exp == 0)
        {
            // Subnormal: value = (-1)^sign * 2^(1-bias) * (mant / 8)
            value = MathF.Pow(2.0f, 1 - 7) * (mant / 8.0f);
        }
        else
        {
            // Normal: value = (-1)^sign * 2^(exp-bias) * (1 + mant/8)
            value = MathF.Pow(2.0f, (int)exp - 7) * (1.0f + mant / 8.0f);
        }

        return sign != 0 ? -value : value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte FloatToFp8E4M3(float f)
    {
        uint bits = BitConverter.SingleToUInt32Bits(f);
        uint sign = (bits >> 31) & 1;
        int f32Exp = (int)((bits >> 23) & 0xFF) - 127;
        uint f32Mant = bits & 0x7FFFFF;

        if (float.IsNaN(f))
            return (byte)((sign << 7) | 0x7F); // Max magnitude in E4M3FN (no NaN encoding)

        float absF = MathF.Abs(f);
        if (absF == 0.0f)
            return (byte)(sign << 7);

        // Saturate to max representable value (448)
        if (absF > 448.0f)
            return (byte)((sign << 7) | 0x7E);

        // Subnormal range: absF < 2^(1-7) = 2^-6 = 0.015625
        if (f32Exp < -6)
        {
            float scaled = absF / MathF.Pow(2.0f, -6);
            uint mant = (uint)MathF.Round(scaled * 8.0f);
            if (mant == 0) return (byte)(sign << 7);
            if (mant > 7) mant = 7;
            return (byte)((sign << 7) | mant);
        }

        int e4Exp = f32Exp + 7;
        if (e4Exp < 1) e4Exp = 1;
        if (e4Exp > 15) e4Exp = 15;

        uint mant3 = (f32Mant + (1u << 19)) >> 20; // round to nearest
        if (mant3 > 7)
        {
            mant3 = 0;
            e4Exp++;
            if (e4Exp > 15) return (byte)((sign << 7) | 0x7E);
        }

        return (byte)((sign << 7) | ((uint)e4Exp << 3) | mant3);
    }

    // E5M2: sign(1) + exponent(5) + mantissa(2), bias=15, has NaN/Inf, max=57344.
    // Maps directly onto the upper byte of FP16 (same exponent/mantissa layout).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Fp8E5M2ToFloat(byte fp8)
    {
        ushort f16Bits = (ushort)((uint)fp8 << 8);
        return (float)BitConverter.UInt16BitsToHalf(f16Bits);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte FloatToFp8E5M2(float f)
    {
        Half h = (Half)f;
        ushort f16Bits = BitConverter.HalfToUInt16Bits(h);
        byte upper = (byte)(f16Bits >> 8);
        // Round up if discarded bits' MSB is set and we're not already at max
        if ((f16Bits & 0x80) != 0 && (upper & 0x7F) < 0x7F)
            upper++;
        return upper;
    }

    /// <summary>Frees owned memory via atomic pointer exchange; frees cached GPU data (on every owning backend)
    /// without D2H copy. Borrowed tensors are no-ops. Every independent owner is attempted even if an earlier
    /// backend callback fails; the first cleanup error is rethrown only after the host buffer is released.</summary>
    public void Dispose()
    {
        Action? gpuDispose;
        GpuBinding[]? extras;
        lock (this)
        {
            if (_disposed)
                return;
            _disposed = true;
            gpuDispose = _gpuDisposeCallback;
            _gpuDisposeCallback = null;
            _gpuSyncCallback = null;
            _gpuCleanupContext = 0;
            if (_extraGpuBindings is { Count: > 0 } bindings)
            {
                extras = [.. bindings];
                bindings.Clear();
            }
            else
            {
                extras = null;
            }
        }
        Interlocked.Exchange(ref _dataPointer, 0);
        Exception? firstError = null;
        if (gpuDispose is not null)
        {
            try { gpuDispose(); }
            catch (Exception error) { firstError = error; }
        }
        if (extras is not null)
        {
            foreach (GpuBinding binding in extras)
            {
                try { binding.Dispose(); }
                catch (Exception error) { firstError ??= error; }
            }
        }
        NativeBuffer? buffer = Interlocked.Exchange(ref _ownedBuffer, null);
        if (buffer is not null)
        {
            try { buffer.Dispose(); }
            catch (Exception error) { firstError ??= error; }
        }
        _keepAlive = null;
        GC.SuppressFinalize(this);
        if (firstError is not null)
            ExceptionDispatchInfo.Capture(firstError).Throw();
    }

    ~Tensor()
    {
        _disposed = true;
        Interlocked.Exchange(ref _dataPointer, 0);
        Action? gpuDispose = Interlocked.Exchange(ref _gpuDisposeCallback, null);
        Interlocked.Exchange(ref _gpuSyncCallback, null);
        // Never invoke a CUDA-touching callback from the finalizer thread directly — queue it under the owning
        // context so only that backend's own thread drains it (see PendingFinalizerGpuCleanup). Each overflow
        // binding routes into ITS OWN key's bucket — that per-owner routing is the whole point of keyed bindings.
        if (gpuDispose is not null)
        {
            EnqueueFinalizerGpuCleanup(_gpuCleanupContext, gpuDispose);
        }
        GpuBinding[]? extras = TakeExtraBindings();
        if (extras is not null)
        {
            foreach (GpuBinding binding in extras)
            {
                EnqueueFinalizerGpuCleanup(binding.Key, binding.Dispose);
            }
        }
        NativeBuffer? buffer = Interlocked.Exchange(ref _ownedBuffer, null);
        if (buffer is not null)
        {
            buffer.Dispose();
        }
    }
}
