using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;

namespace HartsyInference.Core.Memory;

/// <summary>Manages the lifetime of a memory-mapped file for zero-copy model weight loading.</summary>
public sealed unsafe class MmapHandle : IDisposable
{
    private nint _basePointer;
    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _accessor;

    private MmapHandle(MemoryMappedFile mmf, MemoryMappedViewAccessor accessor, byte* basePointer, long byteLength, string filePath)
    {
        _mmf = mmf;
        _accessor = accessor;
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

        return new MmapHandle(mmf, accessor, pointer, fileLength, filePath);
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
    }

    [DllImport("libc", EntryPoint = "madvise")]
    private static extern int Madvise(nint address, nuint length, int advice);

    /// <summary>Releases the memory-mapped view and file via atomic pointer exchange.</summary>
    public void Dispose()
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
