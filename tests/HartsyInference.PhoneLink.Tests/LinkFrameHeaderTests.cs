using Xunit;

namespace HartsyInference.PhoneLink.Tests;

public sealed class LinkFrameHeaderTests
{
    [Fact]
    public void Write_MatchesGoldenLittleEndianLayout()
    {
        LinkFrameHeader header = new(0x00000005, LinkMessageType.OutboundAudio, LinkFrameFlags.Concealed, 0x11223344, 0xAABBCCDD);
        byte[] bytes = new byte[LinkFrameHeader.Size];

        header.Write(bytes);

        byte[] expected =
        [
            0x05, 0x00, 0x00, 0x00, // payload length
            0x21,                   // type
            0x01,                   // flags
            0x00, 0x00,             // reserved
            0x44, 0x33, 0x22, 0x11, // call id
            0xDD, 0xCC, 0xBB, 0xAA, // sequence
        ];
        Assert.Equal(16, LinkFrameHeader.Size);
        Assert.Equal(expected, bytes);
    }

    [Fact]
    public void TryRead_RoundTripsAndIgnoresReserved()
    {
        LinkFrameHeader header = new(LinkProtocol.MaxPayloadBytes, LinkMessageType.Error, LinkFrameFlags.None, uint.MaxValue, 7);
        byte[] bytes = new byte[LinkFrameHeader.Size + 3];
        header.Write(bytes);
        bytes[6] = 0xFF;
        bytes[7] = 0xFF;

        Assert.True(LinkFrameHeader.TryRead(bytes, out LinkFrameHeader parsed));
        Assert.Equal(header, parsed);
    }

    [Fact]
    public void TryRead_NeedsTheWholeHeader()
    {
        byte[] bytes = new byte[LinkFrameHeader.Size - 1];

        Assert.False(LinkFrameHeader.TryRead(bytes, out LinkFrameHeader parsed));
        Assert.Equal(default, parsed);
    }

    [Fact]
    public void Write_RejectsShortDestination()
    {
        LinkFrameHeader header = new(0, LinkMessageType.Ping, LinkFrameFlags.None, 0, 0);
        byte[] bytes = new byte[LinkFrameHeader.Size - 1];

        Assert.Throws<ArgumentException>(() => header.Write(bytes));
    }
}
