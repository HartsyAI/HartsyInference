using HartsyInference.Core.Numerics;

namespace HartsyInference.Core.Memory;

/// <summary>64-byte aligned unmanaged memory block. Backing store for tensor data.</summary>
public sealed unsafe class NativeBuffer : IDisposable
{
    /// <summary>Byte size at or above which the mandatory zero-fill is spread across cores.</summary>
    /// <remarks>Zeroing a fresh multi-hundred-MB weight buffer is a first-touch page-fault storm as much as a memset,
    /// and both scale with threads: on a 16-core box the 105 MB F32 intermediate that <c>Tensor.CastTo</c> allocates
    /// per fp8-quantized weight spent ~50 ms here versus ~14 ms doing the actual conversion. The buffer is still fully
    /// zeroed — only the work is split — so no caller can observe the difference.</remarks>
    private const nuint ParallelClearMinBytes = (nuint)(8UL << 20);

    /// <summary>Bytes per zero-fill chunk. The chunk count follows from the size alone, never from the core count.</summary>
    private const nuint ParallelClearChunkBytes = (nuint)(2UL << 20);

    private static readonly Action<nint, nuint> ClearRange = static (start, length) => NativeMemory.Clear((void*)start, length);

    private static readonly Action<int, ClearPlan> ClearChunk = static (chunk, plan) => plan.Run(chunk);

    private nint _pointer;

    /// <summary>Allocates an aligned block of unmanaged memory, zeroed.</summary>
    public NativeBuffer(nuint byteLength, nuint alignment = 64)
    {
        if (byteLength == 0)
            throw new ArgumentException("Cannot allocate zero bytes.", nameof(byteLength));
        ByteLength = byteLength;
        Alignment = alignment;
        _pointer = (nint)NativeMemory.AlignedAlloc(byteLength, alignment);
        Clear((void*)_pointer, byteLength, ClearRange);
        // This managed wrapper is a few dozen bytes — the GC sees no reason to collect it under
        // memory pressure even while it privately owns megabytes of unmanaged host RAM. Without this,
        // any Tensor/NativeBuffer that isn't explicitly Dispose()d (left to the finalizer) can pile up
        // unbounded host memory well past what a healthy managed heap would ever trigger a GC for —
        // only the OS OOM-killer eventually notices. AddMemoryPressure makes the GC feel the real cost.
        GC.AddMemoryPressure((long)byteLength);
    }

    /// <summary>Zeroes <paramref name="byteLength"/> bytes at <paramref name="pointer"/> through
    /// <paramref name="clearRange"/>: in one call below <see cref="ParallelClearMinBytes"/>, otherwise in
    /// <see cref="ParallelClearChunkBytes"/> chunks through <see cref="CpuParallel"/>, so the <c>numerics.cpuThreads</c>
    /// cap and a thread's <see cref="CpuParallel.InlineScope"/> hold for an allocation like for any kernel.</summary>
    internal static void Clear(void* pointer, nuint byteLength, Action<nint, nuint> clearRange)
    {
        if (byteLength < ParallelClearMinBytes)
        {
            clearRange((nint)pointer, byteLength);
            return;
        }
        ClearPlan plan = new((nint)pointer, byteLength, clearRange);
        CpuParallel.For(plan.Chunks, (long)byteLength, plan, ClearChunk);
    }

    /// <summary>Total size in bytes of the allocated buffer.</summary>
    public nuint ByteLength { get; }

    /// <summary>Alignment in bytes (64 for AVX-512 compatibility).</summary>
    public nuint Alignment { get; }

    /// <summary>Raw pointer to the allocated memory. Throws if disposed.</summary>
    public void* Pointer
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            nint ptr = _pointer;
            if (ptr == 0)
                throw new ObjectDisposedException(nameof(NativeBuffer));
            return (void*)ptr;
        }
    }

    /// <summary>Interprets the buffer as a span of <typeparamref name="T"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> AsSpan<T>() where T : unmanaged
    {
        nint ptr = _pointer;
        if (ptr == 0)
            throw new ObjectDisposedException(nameof(NativeBuffer));
        return new Span<T>((void*)ptr, (int)(ByteLength / (nuint)sizeof(T)));
    }

    /// <summary>Interprets the buffer as a read-only span of <typeparamref name="T"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> AsReadOnlySpan<T>() where T : unmanaged
    {
        nint ptr = _pointer;
        if (ptr == 0)
            throw new ObjectDisposedException(nameof(NativeBuffer));
        return new ReadOnlySpan<T>((void*)ptr, (int)(ByteLength / (nuint)sizeof(T)));
    }

    /// <summary>Frees the underlying unmanaged memory via atomic pointer exchange.</summary>
    public void Dispose()
    {
        nint ptr = Interlocked.Exchange(ref _pointer, 0);
        if (ptr != 0)
        {
            NativeMemory.AlignedFree((void*)ptr);
            GC.RemoveMemoryPressure((long)ByteLength);
        }
        GC.SuppressFinalize(this);
    }

    ~NativeBuffer()
    {
        nint ptr = Interlocked.Exchange(ref _pointer, 0);
        if (ptr != 0)
        {
            NativeMemory.AlignedFree((void*)ptr);
            GC.RemoveMemoryPressure((long)ByteLength);
        }
    }

    /// <summary>A zero-fill cut into <see cref="ParallelClearChunkBytes"/> chunks, the last one shorter; chunk
    /// <c>i</c> starts at <c>i * ParallelClearChunkBytes</c>.</summary>
    private readonly record struct ClearPlan(nint Start, nuint Length, Action<nint, nuint> ClearRange)
    {
        public int Chunks => checked((int)((Length + ParallelClearChunkBytes - 1) / ParallelClearChunkBytes));

        public void Run(int chunk)
        {
            nuint offset = (nuint)chunk * ParallelClearChunkBytes;
            ClearRange(Start + (nint)offset, Math.Min(ParallelClearChunkBytes, Length - offset));
        }
    }
}
