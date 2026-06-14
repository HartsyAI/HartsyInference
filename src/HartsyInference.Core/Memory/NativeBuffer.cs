namespace HartsyInference.Core.Memory;

/// <summary>64-byte aligned unmanaged memory block. Backing store for tensor data.</summary>
public sealed unsafe class NativeBuffer : IDisposable
{
    private nint _pointer;

    /// <summary>Allocates an aligned block of unmanaged memory, zeroed.</summary>
    public NativeBuffer(nuint byteLength, nuint alignment = 64)
    {
        if (byteLength == 0)
            throw new ArgumentException("Cannot allocate zero bytes.", nameof(byteLength));
        ByteLength = byteLength;
        Alignment = alignment;
        _pointer = (nint)NativeMemory.AlignedAlloc(byteLength, alignment);
        NativeMemory.Clear((void*)_pointer, byteLength);
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
            NativeMemory.AlignedFree((void*)ptr);
        GC.SuppressFinalize(this);
    }

    ~NativeBuffer()
    {
        nint ptr = Interlocked.Exchange(ref _pointer, 0);
        if (ptr != 0)
            NativeMemory.AlignedFree((void*)ptr);
    }
}
