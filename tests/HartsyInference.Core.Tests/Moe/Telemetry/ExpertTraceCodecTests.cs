using HartsyInference.Core.Moe.Telemetry;
using Xunit;

namespace HartsyInference.Core.Tests.Moe.Telemetry;

/// <summary>Binary trace format: round trip, corrupt or truncated input rejected, and reproducible synthetic traces.</summary>
public sealed class ExpertTraceCodecTests
{
    [Fact]
    public void RoundTrip_ReadsBackIdenticalRecords()
    {
        ExpertTraceRecord[] trace = SyntheticTraceGenerator.Zipf(7, 3, 32, 50, 2, 1.1, 4096);
        using MemoryStream stream = new();

        ExpertTraceCodec.Write(stream, trace);
        stream.Position = 0;
        ExpertTraceRecord[] read = ExpertTraceCodec.Read(stream);

        Assert.Equal(trace.Length, read.Length);
        Assert.Equal(trace, read);
        Assert.Equal(ExpertTraceCodec.HeaderSize + (trace.Length * ExpertTraceCodec.RecordSize), stream.Length);
    }

    [Fact]
    public void Read_RejectsBadMagic()
    {
        byte[] bytes = Encode(new ExpertTraceRecord(1, 2, 3));
        bytes[0] ^= 0xFF;

        Assert.Throws<ExpertTraceFormatException>(() => Decode(bytes));
    }

    [Fact]
    public void Read_RejectsUnsupportedVersion()
    {
        byte[] bytes = Encode(new ExpertTraceRecord(1, 2, 3));
        bytes[4] = 2;

        Assert.Throws<ExpertTraceFormatException>(() => Decode(bytes));
    }

    [Fact]
    public void Read_RejectsWrongRecordSizeAndNonZeroReserved()
    {
        byte[] wrongSize = Encode(new ExpertTraceRecord(1, 2, 3));
        wrongSize[6] = 16;
        Assert.Throws<ExpertTraceFormatException>(() => Decode(wrongSize));

        byte[] reserved = Encode(new ExpertTraceRecord(1, 2, 3));
        reserved[12] = 1;
        Assert.Throws<ExpertTraceFormatException>(() => Decode(reserved));
    }

    [Fact]
    public void Read_RejectsTruncatedAndTrailingData()
    {
        byte[] bytes = Encode(new ExpertTraceRecord(1, 2, 3), new ExpertTraceRecord(4, 5, 6));

        Assert.Throws<ExpertTraceFormatException>(() => Decode(bytes[..^1]));
        Assert.Throws<ExpertTraceFormatException>(() => Decode(bytes[..10]));
        Assert.Throws<ExpertTraceFormatException>(() => Decode([.. bytes, 0]));
    }

    [Fact]
    public void Zipf_SameSeedGivesSameTraceAndDifferentSeedDiffers()
    {
        ExpertTraceRecord[] first = SyntheticTraceGenerator.Zipf(99, 4, 64, 200, 4, 1.0, 1 << 20);
        ExpertTraceRecord[] second = SyntheticTraceGenerator.Zipf(99, 4, 64, 200, 4, 1.0, 1 << 20);
        ExpertTraceRecord[] other = SyntheticTraceGenerator.Zipf(100, 4, 64, 200, 4, 1.0, 1 << 20);

        Assert.Equal(first, second);
        Assert.NotEqual(first, other);
        Assert.Equal(200 * 4 * 4, first.Length);
    }

    private static byte[] Encode(params ExpertTraceRecord[] records)
    {
        using MemoryStream stream = new();
        ExpertTraceCodec.Write(stream, records);
        return stream.ToArray();
    }

    private static ExpertTraceRecord[] Decode(byte[] bytes)
    {
        using MemoryStream stream = new(bytes);
        return ExpertTraceCodec.Read(stream);
    }
}
