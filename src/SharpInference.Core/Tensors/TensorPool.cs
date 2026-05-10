using System.Collections.Concurrent;
using SharpInference.Core.Memory;

namespace SharpInference.Core.Tensors;

/// <summary>Thread-safe pool of unmanaged memory buffers for temporary tensor allocations.</summary>
public sealed class TensorPool : IDisposable
{
    private readonly ConcurrentDictionary<nuint, ConcurrentBag<NativeBuffer>> _pools = new();
    private readonly ConcurrentDictionary<nint, NativeBuffer> _rentedBuffers = new();
    private int _disposed;

    /// <summary>Rents a buffer of at least the specified byte size. May return a larger buffer.</summary>
    public NativeBuffer Rent(nuint byteSize)
    {
        if (_disposed != 0)
            throw new ObjectDisposedException(nameof(TensorPool));

        nuint bucketSize = RoundUpToPowerOf2(byteSize);

        if (_pools.TryGetValue(bucketSize, out ConcurrentBag<NativeBuffer>? bag) && bag.TryTake(out NativeBuffer? buffer))
        {
            return buffer;
        }

        return new NativeBuffer(bucketSize);
    }

    /// <summary>Returns a buffer to the pool for future reuse.</summary>
    public void Return(NativeBuffer buffer)
    {
        if (_disposed != 0)
            throw new ObjectDisposedException(nameof(TensorPool));

        ConcurrentBag<NativeBuffer> bag = _pools.GetOrAdd(buffer.ByteLength, _ => new ConcurrentBag<NativeBuffer>());
        bag.Add(buffer);
    }

    /// <summary>Rents a tensor backed by a pooled buffer. Return via ReturnTensor when done.</summary>
    public unsafe Tensor RentTensor(TensorShape shape, DType dtype)
    {
        if (_disposed != 0)
            throw new ObjectDisposedException(nameof(TensorPool));

        long byteSize = Tensor.ComputeByteSize(shape, dtype);
        NativeBuffer buffer = Rent((nuint)byteSize);
        _rentedBuffers[(nint)buffer.Pointer] = buffer;
        return new Tensor(buffer.Pointer, shape, dtype);
    }

    /// <summary>Returns a rented tensor's underlying buffer to the pool.</summary>
    public unsafe void ReturnTensor(Tensor tensor)
    {
        if (_disposed != 0)
            throw new ObjectDisposedException(nameof(TensorPool));

        if (tensor.OwnsMemory)
            throw new InvalidOperationException("Cannot return an owned tensor to the pool. Only pooled (borrowed) tensors can be returned.");

        nint ptr = (nint)tensor.DataPointer;
        if (_rentedBuffers.TryRemove(ptr, out NativeBuffer? buffer))
        {
            Return(buffer);
        }
        else
        {
            throw new InvalidOperationException("Tensor was not rented from this pool.");
        }
    }

    private static nuint RoundUpToPowerOf2(nuint value) =>
        (nuint)BitOperations.RoundUpToPowerOf2((ulong)value);

    /// <summary>Disposes all pooled buffers. Thread-safe via Interlocked.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        foreach (KeyValuePair<nuint, ConcurrentBag<NativeBuffer>> kvp in _pools)
        {
            while (kvp.Value.TryTake(out NativeBuffer? buffer))
            {
                buffer.Dispose();
            }
        }

        foreach (KeyValuePair<nint, NativeBuffer> kvp in _rentedBuffers)
        {
            kvp.Value.Dispose();
        }

        _pools.Clear();
        _rentedBuffers.Clear();
    }
}
