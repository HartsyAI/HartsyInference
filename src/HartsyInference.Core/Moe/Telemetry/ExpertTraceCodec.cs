using System.Buffers.Binary;

namespace HartsyInference.Core.Moe.Telemetry;

/// <summary>
/// Versioned binary trace format, little-endian. Header (16 bytes): magic "HMTR", version u16 (1), record size u16 (8),
/// record count u64, reserved u32 (must be zero). Body: one 8-byte record per access: layer u16, expert u16, bytes u32.
/// </summary>
public static class ExpertTraceCodec
{
    /// <summary>Format version written and accepted.</summary>
    public const ushort Version = 1;

    /// <summary>Bytes in the fixed header.</summary>
    public const int HeaderSize = 16;

    /// <summary>Bytes per record.</summary>
    public const int RecordSize = 8;

    private const int MaxRecords = int.MaxValue / RecordSize;

    private static ReadOnlySpan<byte> Magic => "HMTR"u8;

    /// <summary>Writes the header and every record to <paramref name="stream"/>.</summary>
    public static void Write(Stream stream, ReadOnlySpan<ExpertTraceRecord> records)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (records.Length > MaxRecords) throw new ArgumentOutOfRangeException(nameof(records), "Trace is too large for one file.");

        byte[] buffer = new byte[HeaderSize + (records.Length * RecordSize)];
        Magic.CopyTo(buffer);
        BinaryPrimitives.WriteUInt16LittleEndian(buffer.AsSpan(4), Version);
        BinaryPrimitives.WriteUInt16LittleEndian(buffer.AsSpan(6), RecordSize);
        BinaryPrimitives.WriteUInt64LittleEndian(buffer.AsSpan(8), (ulong)records.Length);

        int offset = HeaderSize;
        foreach (ExpertTraceRecord record in records)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(buffer.AsSpan(offset), record.Layer);
            BinaryPrimitives.WriteUInt16LittleEndian(buffer.AsSpan(offset + 2), record.Expert);
            BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(offset + 4), record.Bytes);
            offset += RecordSize;
        }

        stream.Write(buffer, 0, buffer.Length);
    }

    /// <summary>Reads a whole trace. Throws <see cref="ExpertTraceFormatException"/> on a bad magic, version, size or length.</summary>
    public static ExpertTraceRecord[] Read(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);

        Span<byte> header = stackalloc byte[HeaderSize];
        ReadFully(stream, header, "header");

        if (!header[..4].SequenceEqual(Magic)) throw new ExpertTraceFormatException("Bad magic: not an expert trace.");

        ushort version = BinaryPrimitives.ReadUInt16LittleEndian(header[4..]);
        if (version != Version) throw new ExpertTraceFormatException($"Unsupported trace version {version}.");

        ushort recordSize = BinaryPrimitives.ReadUInt16LittleEndian(header[6..]);
        if (recordSize != RecordSize) throw new ExpertTraceFormatException($"Unexpected record size {recordSize}.");

        ulong count = BinaryPrimitives.ReadUInt64LittleEndian(header[8..]);
        if (count > MaxRecords) throw new ExpertTraceFormatException("Record count exceeds the supported size.");

        if (BinaryPrimitives.ReadUInt32LittleEndian(header[12..]) != 0)
        {
            throw new ExpertTraceFormatException("Reserved header bytes are not zero.");
        }

        byte[] body = new byte[(int)count * RecordSize];
        ReadFully(stream, body, "body");
        if (stream.ReadByte() != -1) throw new ExpertTraceFormatException("Trailing bytes after the last record.");

        ExpertTraceRecord[] records = new ExpertTraceRecord[(int)count];
        for (int i = 0; i < records.Length; i++)
        {
            ReadOnlySpan<byte> at = body.AsSpan(i * RecordSize);
            records[i] = new ExpertTraceRecord(
                BinaryPrimitives.ReadUInt16LittleEndian(at),
                BinaryPrimitives.ReadUInt16LittleEndian(at[2..]),
                BinaryPrimitives.ReadUInt32LittleEndian(at[4..]));
        }

        return records;
    }

    private static void ReadFully(Stream stream, Span<byte> target, string part)
    {
        int total = 0;
        while (total < target.Length)
        {
            int read = stream.Read(target[total..]);
            if (read == 0) throw new ExpertTraceFormatException($"Trace ends inside the {part}.");
            total += read;
        }
    }
}
