using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Memory;

namespace HartsyInference.Audio.Phonemizer.MeCab;

/// <summary>MeCab's connection-cost matrix (<c>matrix.bin</c>), memory-mapped: two uint16 sizes then
/// <c>lsize * rsize</c> int16 costs, indexed <c>left.rcAttr + lsize * right.lcAttr</c>.</summary>
internal sealed unsafe class MeCabConnector : IDisposable
{
    private readonly MmapHandle _map;
    private readonly long _cells;

    /// <summary>Maps <paramref name="path"/> and checks its size against the header.</summary>
    /// <exception cref="HartsyInferenceException">The file's size does not match its dimensions.</exception>
    public MeCabConnector(string path)
    {
        _map = MmapHandle.OpenRead(path);
        try
        {
            if (_map.ByteLength < 4)
                throw new HartsyInferenceException($"MeCab matrix '{path}' is truncated.");
            ushort* header = (ushort*)_map.BasePointer;
            LeftSize = header[0];
            RightSize = header[1];
            _cells = (long)LeftSize * RightSize;
            if (_map.ByteLength != 4 + 2 * _cells)
                throw new HartsyInferenceException(
                    $"MeCab matrix '{path}' is {_map.ByteLength} bytes, expected {4 + 2 * _cells} for {LeftSize}x{RightSize}.");
        }
        catch
        {
            _map.Dispose();
            throw;
        }
    }

    /// <summary>Number of left-context ids.</summary>
    public int LeftSize { get; }

    /// <summary>Number of right-context ids.</summary>
    public int RightSize { get; }

    /// <summary>Transition cost from a node whose right id is <paramref name="rightIdOfLeft"/> to one whose left id is
    /// <paramref name="leftIdOfRight"/>.</summary>
    public int Cost(int rightIdOfLeft, int leftIdOfRight)
    {
        long index = rightIdOfLeft + (long)LeftSize * leftIdOfRight;
        if ((ulong)index >= (ulong)_cells)
            throw new HartsyInferenceException($"MeCab context ids ({rightIdOfLeft}, {leftIdOfRight}) exceed the matrix.");
        return ((short*)(_map.BasePointer + 4))[index];
    }

    /// <summary>Unmaps the file.</summary>
    public void Dispose() => _map.Dispose();
}
