using System.Text;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Memory;

namespace HartsyInference.Audio.Phonemizer.MeCab;

/// <summary>A compiled MeCab dictionary (<c>sys.dic</c> or <c>unk.dic</c>), memory-mapped. Layout: ten uint32 header
/// fields (magic xor file size, version 102, type, lexicon size, left/right context sizes, double-array, token and
/// feature byte sizes, reserved), a 32-byte charset name, the Darts double-array (int32 base, uint32 check per unit),
/// 16-byte tokens (lcAttr, rcAttr, posid, wcost, feature offset, compound) and NUL-terminated feature strings.</summary>
internal sealed unsafe class MeCabDictionary : IDisposable
{
    private const uint MagicId = 0xef718f77u;
    private const uint SupportedVersion = 102;
    private const int HeaderBytes = 40 + 32;
    private const int UnitBytes = 8;
    private const int TokenBytes = 16;

    private readonly MmapHandle _map;
    private readonly long _unitCount;
    private readonly long _tokenOffset;
    private readonly long _tokenCount;
    private readonly long _featureOffset;
    private readonly long _featureBytes;

    /// <summary>Maps <paramref name="path"/> and validates its header.</summary>
    /// <exception cref="HartsyInferenceException">The file is not a version-102 MeCab dictionary.</exception>
    public MeCabDictionary(string path)
    {
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("MeCab dictionaries are read as little-endian.");
        _map = MmapHandle.OpenRead(path);
        try
        {
            if (_map.ByteLength < HeaderBytes)
                throw new HartsyInferenceException($"MeCab dictionary '{path}' is truncated.");
            uint* header = (uint*)_map.BasePointer;
            if ((header[0] ^ MagicId) != (ulong)_map.ByteLength)
                throw new HartsyInferenceException($"MeCab dictionary '{path}' is broken (magic/size mismatch).");
            if (header[1] != SupportedVersion)
                throw new HartsyInferenceException($"MeCab dictionary '{path}' has version {header[1]}, expected {SupportedVersion}.");
            Type = (int)header[2];
            LeftSize = (int)header[4];
            RightSize = (int)header[5];
            long daBytes = header[6], tokenBytes = header[7];
            _featureBytes = header[8];
            byte* charset = _map.BasePointer + 40;
            Charset = Encoding.ASCII.GetString(charset, new ReadOnlySpan<byte>(charset, 32).IndexOf((byte)0) is int n && n >= 0 ? n : 32);
            _unitCount = daBytes / UnitBytes;
            _tokenOffset = HeaderBytes + daBytes;
            _tokenCount = tokenBytes / TokenBytes;
            _featureOffset = _tokenOffset + tokenBytes;
            if (_featureOffset + _featureBytes != _map.ByteLength)
                throw new HartsyInferenceException($"MeCab dictionary '{path}' section sizes do not add up to its length.");
            if (!Charset.Equals("utf8", StringComparison.OrdinalIgnoreCase)
                && !Charset.Equals("utf-8", StringComparison.OrdinalIgnoreCase))
                throw new HartsyInferenceException($"MeCab dictionary '{path}' is '{Charset}'; only UTF-8 is supported.");
        }
        catch
        {
            _map.Dispose();
            throw;
        }
    }

    /// <summary>0 for a system dictionary, 1 for a user dictionary, 2 for the unknown-word dictionary.</summary>
    public int Type { get; }

    /// <summary>Number of left-context ids the entries were compiled against.</summary>
    public int LeftSize { get; }

    /// <summary>Number of right-context ids the entries were compiled against.</summary>
    public int RightSize { get; }

    /// <summary>The dictionary's declared charset.</summary>
    public string Charset { get; }

    /// <summary>Darts <c>commonPrefixSearch</c>: every key that is a prefix of <paramref name="key"/>, shortest
    /// first, as (value, byte length). Returns the full match count, of which at most <paramref name="results"/>'s
    /// length are written.</summary>
    public int CommonPrefixSearch(ReadOnlySpan<byte> key, Span<(int Value, int Length)> results)
    {
        if (key.Length == 0) return 0;
        byte* units = _map.BasePointer + HeaderBytes;
        int b = Base(units, 0);
        int num = 0;
        for (int i = 0; i < key.Length; i++)
        {
            uint p = (uint)b;
            if (p < _unitCount)
            {
                int n = Base(units, p);
                if ((uint)b == Check(units, p) && n < 0)
                {
                    if (num < results.Length) results[num] = (-n - 1, i);
                    num++;
                }
            }
            p = (uint)b + key[i] + 1u;
            if (p >= _unitCount || (uint)b != Check(units, p)) return num;
            b = Base(units, p);
        }
        uint last = (uint)b;
        if (last < _unitCount)
        {
            int n = Base(units, last);
            if ((uint)b == Check(units, last) && n < 0)
            {
                if (num < results.Length) results[num] = (-n - 1, key.Length);
                num++;
            }
        }
        return num;
    }

    /// <summary>Darts <c>exactMatchSearch</c>: the value stored for exactly <paramref name="key"/>, or -1.</summary>
    public int ExactMatchSearch(ReadOnlySpan<byte> key)
    {
        byte* units = _map.BasePointer + HeaderBytes;
        int b = Base(units, 0);
        for (int i = 0; i < key.Length; i++)
        {
            uint p = (uint)b + key[i] + 1u;
            if (p >= _unitCount || (uint)b != Check(units, p)) return -1;
            b = Base(units, p);
        }
        uint last = (uint)b;
        if (last >= _unitCount) return -1;
        int n = Base(units, last);
        return (uint)b == Check(units, last) && n < 0 ? -n - 1 : -1;
    }

    /// <summary>The token at <paramref name="index"/>.</summary>
    public MeCabEntry Entry(long index)
    {
        if ((ulong)index >= (ulong)_tokenCount)
            throw new HartsyInferenceException($"MeCab token index {index} is out of range.");
        byte* t = _map.BasePointer + _tokenOffset + index * TokenBytes;
        return new MeCabEntry(*(ushort*)t, *(ushort*)(t + 2), *(short*)(t + 6), *(uint*)(t + 8));
    }

    /// <summary>The feature CSV at <paramref name="offset"/> into the feature section.</summary>
    public string Feature(uint offset)
    {
        if (offset >= _featureBytes)
            throw new HartsyInferenceException($"MeCab feature offset {offset} is out of range.");
        byte* start = _map.BasePointer + _featureOffset + offset;
        int max = (int)Math.Min(int.MaxValue, _featureBytes - offset);
        int len = new ReadOnlySpan<byte>(start, max).IndexOf((byte)0);
        return Encoding.UTF8.GetString(start, len < 0 ? max : len);
    }

    /// <summary>Unmaps the file.</summary>
    public void Dispose() => _map.Dispose();

    private static int Base(byte* units, uint index) => *(int*)(units + (long)index * UnitBytes);

    private static uint Check(byte* units, uint index) => *(uint*)(units + (long)index * UnitBytes + 4);
}
