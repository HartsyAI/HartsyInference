using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using HartsyInference.Core.Exceptions;
using Microsoft.Win32.SafeHandles;

namespace HartsyInference.Core.Memory.Host;

/// <summary>Manages the lifetime of a memory-mapped file for zero-copy model weight loading.</summary>
public sealed unsafe class MmapHandle : IDisposable
{
    private nint _basePointer;
    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _accessor;
    private readonly SafeFileHandle _fileHandle;
    private readonly object _gate = new();

    private MmapHandle(MemoryMappedFile mmf, MemoryMappedViewAccessor accessor, SafeFileHandle fileHandle, byte* basePointer,
        long byteLength, string filePath)
    {
        _mmf = mmf;
        _accessor = accessor;
        _fileHandle = fileHandle;
        _basePointer = (nint)basePointer;
        ByteLength = byteLength;
        FilePath = filePath;
    }

    /// <summary>Total byte length of the memory-mapped region.</summary>
    public long ByteLength { get; }

    /// <summary>File path that was mapped.</summary>
    public string FilePath { get; }

    /// <summary>Base pointer to the start of the mapped memory. Throws if disposed.</summary>
    public byte* BasePointer
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            nint ptr = _basePointer;
            if (ptr == 0)
                throw new ObjectDisposedException(nameof(MmapHandle));
            return (byte*)ptr;
        }
    }

    /// <summary>Opens a file as a read-only memory-mapped region.</summary>
    public static MmapHandle OpenRead(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Model file not found: {filePath}", filePath);

        // Open the file ourselves and measure the stream so the length follows symlinks to the
        // real target. FileInfo.Length reports the link's own size for a symlink, which yields a
        // truncated mapping and a bogus "invalid header length" on otherwise-valid weight files.
        FileStream fs = new(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        long fileLength = fs.Length;
        MemoryMappedFile mmf = MemoryMappedFile.CreateFromFile(
            fs, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.None, leaveOpen: false);
        MemoryMappedViewAccessor accessor = mmf.CreateViewAccessor(0, fileLength, MemoryMappedFileAccess.Read);

        byte* pointer = null;
        accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);

        // The mapping owns fs and closes it with mmf, so this handle serves positional reads of the same inode.
        return new MmapHandle(mmf, accessor, fs.SafeFileHandle, pointer, fileLength, filePath);
    }

    /// <summary>Returns a pointer offset from the base; allows an offset equal to <see cref="ByteLength"/> so trailing zero-byte marker tensors resolve to a valid past-the-end pointer.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public byte* PointerAt(long byteOffset)
    {
        nint ptr = _basePointer;
        if (ptr == 0)
            throw new ObjectDisposedException(nameof(MmapHandle));

        // Allow byteOffset == ByteLength so 0-byte marker tensors at the end of a file
        // (e.g. ComfyUI fp8_scaled `*.scaled_fp8` markers with shape=[0]) return a
        // past-the-end pointer that callers must not read from.
        if ((ulong)byteOffset > (ulong)ByteLength)
            throw new ArgumentOutOfRangeException(nameof(byteOffset));

        return (byte*)ptr + byteOffset;
    }

    /// <summary>Reads bytes of the mapped file by positional read through the handle that created the mapping, bypassing the mapping itself.</summary>
    /// <remarks>Safe from several threads at once, and the reads overlap: a network volume serves one reader far slower than several.</remarks>
    /// <exception cref="ObjectDisposedException">The handle was disposed before the read started.</exception>
    public void ReadAt(long offset, Span<byte> destination)
    {
        // A fast path only: Dispose can run right after this check, and the handle reference below is what guards the read.
        if (Volatile.Read(ref _basePointer) == 0)
            throw new ObjectDisposedException(nameof(MmapHandle));
        if (offset < 0 || offset > ByteLength - destination.Length)
            throw new ArgumentOutOfRangeException(nameof(offset), $"Read of {destination.Length} bytes at {offset} is outside the {ByteLength}-byte mapping.");
        // A reference on the file handle, not a lock, keeps it open for this read: Dispose's close takes effect once the last reader
        // releases it, and a read that starts after the close throws ObjectDisposedException here.
        bool referenced = false;
        try
        {
            _fileHandle.DangerousAddRef(ref referenced);
            int done = 0;
            while (done < destination.Length)
            {
                int read = RandomAccess.Read(_fileHandle, destination[done..], offset + done);
                if (read == 0)
                    throw new HartsyInferenceException($"'{FilePath}' ended after {done} of {destination.Length} bytes at offset {offset}; the file changed while it was mapped.");
                done += read;
            }
        }
        finally
        {
            if (referenced) _fileHandle.DangerousRelease();
        }
    }

    /// <summary>Hints the kernel about access to a byte range; false where madvise is unavailable or refuses, and never load-bearing.</summary>
    public bool Advise(long offset, long length, MmapAdvice advice)
    {
        nint ptr = _basePointer;
        if (ptr == 0)
            throw new ObjectDisposedException(nameof(MmapHandle));
        if (offset < 0 || length < 0 || offset > ByteLength - length)
            throw new ArgumentOutOfRangeException(nameof(length), $"Range [{offset}, {offset + length}) is outside the {ByteLength}-byte mapping.");
        if (length == 0 || !OperatingSystem.IsLinux())
            return false;

        // madvise wants a page-aligned start; widening downward keeps the requested range covered.
        long page = Environment.SystemPageSize;
        long start = offset / page * page;
        long span = offset + length - start;
        try
        {
            return Madvise(ptr + (nint)start, (nuint)span, (int)advice) == 0;
        }
        catch (EntryPointNotFoundException) { return false; }
        catch (DllNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
    }

    [DllImport("libc", EntryPoint = "madvise")]
    private static extern int Madvise(nint address, nuint length, int advice);

    /// <summary>Releases the memory-mapped view and file via atomic pointer exchange.</summary>
    public void Dispose()
    {
        // In-flight ReadAt calls hold a reference on the file handle, so its close here takes effect when the last of them releases it; the
        // gate only orders concurrent Dispose calls.
        lock (_gate)
        {
            nint ptr = Interlocked.Exchange(ref _basePointer, 0);
            if (ptr != 0)
            {
                _accessor?.SafeMemoryMappedViewHandle.ReleasePointer();
                _accessor?.Dispose();
                _accessor = null;
                _mmf?.Dispose();
                _mmf = null;
            }
        }
        GC.SuppressFinalize(this);
    }

    ~MmapHandle()
    {
        nint ptr = Interlocked.Exchange(ref _basePointer, 0);
        if (ptr != 0)
        {
            _accessor?.SafeMemoryMappedViewHandle.ReleasePointer();
            _accessor?.Dispose();
            _mmf?.Dispose();
        }
    }
}
