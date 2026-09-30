using System.Buffers.Binary;

namespace HartsyInference.PhoneLink;

/// <summary>The fixed 16-byte little-endian frame header: <c>u32 payloadLength | u8 type | u8 flags | u16 reserved | u32 callId | u32 sequence</c>.
/// The reserved halfword is written as zero and ignored on read.</summary>
public readonly record struct LinkFrameHeader(uint PayloadLength, LinkMessageType Type, LinkFrameFlags Flags, uint CallId, uint Sequence)
{
    /// <summary>Header size in bytes; the payload starts at this offset.</summary>
    public const int Size = 16;

    /// <summary>Writes the header into the first <see cref="Size"/> bytes of <paramref name="destination"/>.</summary>
    public void Write(Span<byte> destination)
    {
        if (destination.Length < Size)
            throw new ArgumentException($"Header needs {Size} bytes, destination has {destination.Length}.", nameof(destination));
        BinaryPrimitives.WriteUInt32LittleEndian(destination, PayloadLength);
        destination[4] = (byte)Type;
        destination[5] = (byte)Flags;
        BinaryPrimitives.WriteUInt16LittleEndian(destination.Slice(6), 0);
        BinaryPrimitives.WriteUInt32LittleEndian(destination.Slice(8), CallId);
        BinaryPrimitives.WriteUInt32LittleEndian(destination.Slice(12), Sequence);
    }

    /// <summary>Parses a header from the first <see cref="Size"/> bytes of <paramref name="source"/>; false when fewer bytes are available. Does not validate the payload length against <see cref="LinkProtocol.MaxPayloadBytes"/>.</summary>
    public static bool TryRead(ReadOnlySpan<byte> source, out LinkFrameHeader header)
    {
        if (source.Length < Size)
        {
            header = default;
            return false;
        }
        header = new LinkFrameHeader(
            BinaryPrimitives.ReadUInt32LittleEndian(source),
            (LinkMessageType)source[4],
            (LinkFrameFlags)source[5],
            BinaryPrimitives.ReadUInt32LittleEndian(source.Slice(8)),
            BinaryPrimitives.ReadUInt32LittleEndian(source.Slice(12)));
        return true;
    }
}
