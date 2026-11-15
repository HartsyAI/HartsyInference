using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using HartsyInference.Core.Memory.Host;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Gpu;

/// <summary>Reads a borrowed memory-mapped tensor into one reusable host buffer, a chunk at a time, before its device upload.</summary>
/// <remarks>A demand-paged mapping on a network volume faults one page at a time, far below sequential read speed. The
/// chunk range is advised as needed first, so the kernel fetches it in one request rather than per fault. A staged
/// chunk stays in use until the caller reports its copy complete with <see cref="MarkCopyComplete"/>; the next
/// <see cref="StageChunk"/> and <see cref="Dispose"/> refuse to run before that.</remarks>
public sealed unsafe class MappedUploadStaging : IDisposable
{
    /// <summary>Default upper bound on one staged chunk, so a large tensor never needs a buffer of its own size.</summary>
    public const long DefaultChunkBytes = 64L << 20;

    private readonly long _chunkBytes;
    private byte[] _buffer = [];
    private bool _disposed;
    private bool _copyPending;

    /// <summary>Creates a staging buffer that stages at most <paramref name="chunkBytes"/> per call.</summary>
    public MappedUploadStaging(long chunkBytes = DefaultChunkBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(chunkBytes);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(chunkBytes, (long)Array.MaxLength);
        _chunkBytes = chunkBytes;
    }

    /// <summary>Bytes currently held by the staging buffer.</summary>
    public long Capacity => _buffer.Length;

    /// <summary>Whether the tensor's bytes lie inside a memory-mapped file, directly or through a chain of borrowed views.</summary>
    public static bool IsMappedView(Tensor tensor) => !tensor.OwnsMemory && FindMapping(tensor) is not null;

    /// <summary>Copies the next chunk of <paramref name="source"/> into the staging buffer. The previous chunk must have
    /// been reported complete with <see cref="MarkCopyComplete"/>.</summary>
    /// <param name="source">A tensor for which <see cref="IsMappedView"/> is true.</param>
    /// <param name="offset">Byte offset into the tensor where this chunk starts.</param>
    /// <param name="remaining">Bytes from <paramref name="offset"/> to the end of the tensor.</param>
    /// <param name="host">The staged chunk, valid until the next call on this instance.</param>
    /// <returns>The number of bytes staged.</returns>
    public int StageChunk(Tensor source, long offset, long remaining, out void* host)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_copyPending)
            throw new InvalidOperationException("The staged chunk is still being read by a copy; call MarkCopyComplete first.");
        long total = source.DType.ComputeByteCount(source.ElementCount);
        if (offset < 0 || remaining < 0 || remaining > total - offset)
            throw new ArgumentOutOfRangeException(nameof(remaining), $"Chunk [{offset}, +{remaining}) is outside the {total}-byte tensor.");

        int count = (int)Math.Min(remaining, _chunkBytes);
        Grow(count);
        byte* src = (byte*)source.DataPointer + offset;
        AdviseWillNeed(source, src, count);
        fixed (byte* dst = _buffer)
            Buffer.MemoryCopy(src, dst, count, count);
        host = Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_buffer));
        _copyPending = true;
        return count;
    }

    /// <summary>Reports that the device copy reading the last staged chunk has finished, so the buffer may be reused.</summary>
    public void MarkCopyComplete() => _copyPending = false;

    /// <summary>Releases the staging buffer. Idempotent.</summary>
    /// <exception cref="InvalidOperationException">A staged chunk is still being read by a copy.</exception>
    public void Dispose()
    {
        if (_copyPending)
            throw new InvalidOperationException("The staged chunk is still being read by a copy; call MarkCopyComplete before disposing.");
        _disposed = true;
        _buffer = [];
    }

    private void Grow(int count)
    {
        if (_buffer.Length >= count)
            return;
        // Pinned on the object heap so the pointer StageChunk returns cannot move under the caller.
        long size = Math.Min(_chunkBytes, Math.Max(count, (long)_buffer.Length * 2));
        _buffer = GC.AllocateUninitializedArray<byte>((int)size, pinned: true);
    }

    private static void AdviseWillNeed(Tensor source, byte* src, int count)
    {
        MmapHandle? mapping = FindMapping(source);
        if (mapping is null)
            return;
        long start = (long)(src - mapping.BasePointer);
        if (start < 0 || start > mapping.ByteLength - count)
            return;
        mapping.Advise(start, count, MmapAdvice.WillNeed);
    }

    private static MmapHandle? FindMapping(Tensor tensor)
    {
        object? owner = tensor.KeepAliveOwner;
        // A view roots its parent tensor, which roots the mapping, so follow the chain to the mapping itself.
        for (int depth = 0; depth < 8 && owner is Tensor parent; depth++)
            owner = parent.KeepAliveOwner;
        return owner as MmapHandle;
    }
}
