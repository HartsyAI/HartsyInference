using System.Buffers.Binary;
using System.Text;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.Audio.Phonemizer.MeCab;

/// <summary>MeCab's compiled character table (<c>char.bin</c>): a uint32 category count, that many 32-byte
/// NUL-padded category names, then one <see cref="MeCabCharInfo"/> per UCS-2 code point 0..0xFFFE.</summary>
internal sealed class MeCabCharProperty
{
    private const int NameBytes = 32;
    private const int CodePoints = 0xFFFF;

    private readonly uint[] _map;

    private MeCabCharProperty(string[] names, uint[] map)
    {
        Names = names;
        _map = map;
    }

    /// <summary>Category names in id order (<c>DEFAULT</c>, <c>SPACE</c>, ...).</summary>
    public IReadOnlyList<string> Names { get; }

    /// <summary>Reads <c>char.bin</c>.</summary>
    /// <exception cref="HartsyInferenceException">The file's size does not match its category count.</exception>
    public static MeCabCharProperty Load(string path)
    {
        byte[] data = File.ReadAllBytes(path);
        if (data.Length < sizeof(uint))
            throw new HartsyInferenceException($"MeCab char.bin '{path}' is truncated.");
        uint count = BinaryPrimitives.ReadUInt32LittleEndian(data);
        long expected = sizeof(uint) + NameBytes * (long)count + sizeof(uint) * (long)CodePoints;
        if (count > 256 || data.Length != expected)
            throw new HartsyInferenceException($"MeCab char.bin '{path}' is {data.Length} bytes, expected {expected}.");
        string[] names = new string[count];
        for (int i = 0; i < names.Length; i++)
        {
            ReadOnlySpan<byte> raw = data.AsSpan(sizeof(uint) + NameBytes * i, NameBytes);
            int nul = raw.IndexOf((byte)0);
            names[i] = Encoding.UTF8.GetString(nul < 0 ? raw : raw[..nul]);
        }
        uint[] map = new uint[CodePoints];
        int offset = sizeof(uint) + NameBytes * names.Length;
        for (int i = 0; i < map.Length; i++)
            map[i] = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset + sizeof(uint) * i));
        return new MeCabCharProperty(names, map);
    }

    /// <summary>The entry for UCS-2 code <paramref name="code"/>; codes the table does not cover read as code 0.</summary>
    public MeCabCharInfo Get(int code) => new((uint)code < (uint)_map.Length ? _map[code] : _map[0]);

    /// <summary>MeCab's <c>getCharInfo</c> over UTF-8: decodes the character at <paramref name="pos"/> to UCS-2 the
    /// way <c>utf8_to_ucs2</c> does (characters beyond the BMP become code 0) and returns its entry and byte length.
    /// Reading at the sentence end sees the terminating NUL.</summary>
    public MeCabCharInfo Get(ReadOnlySpan<byte> sentence, int pos, int end, out int byteLength)
    {
        int b0 = pos < sentence.Length ? sentence[pos] : 0;
        int len = end - pos;
        int code;
        if (b0 < 0x80)
        {
            byteLength = 1;
            code = b0;
        }
        else if (len >= 2 && (b0 & 0xE0) == 0xC0)
        {
            byteLength = 2;
            code = ((b0 & 0x1F) << 6) | (sentence[pos + 1] & 0x3F);
        }
        else if (len >= 3 && (b0 & 0xF0) == 0xE0)
        {
            byteLength = 3;
            code = ((b0 & 0x0F) << 12) | ((sentence[pos + 1] & 0x3F) << 6) | (sentence[pos + 2] & 0x3F);
        }
        else if (len >= 4 && (b0 & 0xF8) == 0xF0)
        {
            byteLength = 4;
            code = 0;
        }
        else
        {
            byteLength = 1;
            code = 0;
        }
        return Get(code);
    }
}
