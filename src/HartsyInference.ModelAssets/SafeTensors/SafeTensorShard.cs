using HartsyInference.Core.IO;
using HartsyInference.Core.Memory;

namespace HartsyInference.ModelAssets.SafeTensors;

/// <summary>One file of a <see cref="ShardedSafeTensorSet"/>: its validated header, plus a map or pread handle created only on first use.</summary>
public sealed class SafeTensorShard
{
    private readonly object _gate = new object();
    private MmapHandle? _map;
    private PreadByteSource? _source;
    private bool _released;

    internal SafeTensorShard(int index, string path, SafeTensorHeader header, bool preadOnly)
    {
        Index = index;
        Path = path;
        FileName = System.IO.Path.GetFileName(path);
        Header = header;
        PreadOnly = preadOnly;
    }

    /// <summary>Position of this shard within the set.</summary>
    public int Index { get; }

    /// <summary>Full path of the shard file.</summary>
    public string Path { get; }

    /// <summary>File name without its directory, as the index's <c>weight_map</c> spells it.</summary>
    public string FileName { get; }

    /// <summary>The validated header of this file.</summary>
    public SafeTensorHeader Header { get; }

    /// <summary>True when the shard is read only through <see cref="IWeightByteSource"/> and never mapped.</summary>
    public bool PreadOnly { get; }

    /// <summary>Whether the shard's file is currently memory-mapped.</summary>
    public bool IsMapped
    {
        get
        {
            lock (_gate) return _map is not null;
        }
    }

    internal MmapHandle Map(bool adviseRandom)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_released, this);
            if (_map is not null)
                return _map;
            MmapHandle handle = MmapHandle.OpenRead(Path);
            if (handle.ByteLength != Header.FileLength)
            {
                long actual = handle.ByteLength;
                handle.Dispose();
                throw new InvalidOperationException(
                    $"'{Path}' is {actual} bytes now but was {Header.FileLength} when its header was read; it changed while the set was open.");
            }
            if (adviseRandom)
                handle.Advise(0, handle.ByteLength, MmapAdvice.Random);
            _map = handle;
            return handle;
        }
    }

    internal PreadByteSource Source()
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_released, this);
            return _source ??= new PreadByteSource(Path);
        }
    }

    internal void Release()
    {
        lock (_gate)
        {
            _released = true;
            _map?.Dispose();
            _map = null;
            _source?.Dispose();
            _source = null;
        }
    }
}
