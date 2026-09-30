using Xunit;

namespace HartsyInference.PhoneLink.Tests;

public sealed class LinkFrameWriterTests
{
    private static readonly CancellationToken None = CancellationToken.None;

    [Fact]
    public async Task SequenceCountsEveryFrameFromZero()
    {
        using MemoryStream stream = new();
        using LinkFrameWriter writer = new(stream);

        Assert.Equal(0u, writer.NextSequence);
        await writer.WritePingAsync(1, None);
        await writer.WriteFlushAsync(3, 1, None);
        await writer.WriteCallEndAsync(3, LinkCallEndReason.Completed, None);
        Assert.Equal(3u, writer.NextSequence);

        List<LinkFrame> frames = await LinkRoundTrip.DecodeAsync(new MemoryStream(stream.ToArray()));
        Assert.Equal(new uint[] { 0, 1, 2 }, frames.Select(f => f.Header.Sequence));
        Assert.Equal(new uint[] { 0, 3, 3 }, frames.Select(f => f.Header.CallId));
    }

    [Fact]
    public async Task OversizeRawPayload_IsRejectedBeforeAnythingIsWritten()
    {
        using MemoryStream stream = new();
        using LinkFrameWriter writer = new(stream);

        byte[] oversize = new byte[LinkProtocol.MaxPayloadBytes + 1];
        Assert.Throws<ArgumentOutOfRangeException>(() => writer.WriteAsync(LinkMessageType.Event, LinkFrameFlags.None, 1, oversize, None));

        Assert.Equal(0, stream.Length);
        await writer.WritePingAsync(1, None);
        Assert.Equal(LinkFrameHeader.Size + 8, stream.Length);
    }

    [Fact]
    public void InboundAudio_MustBeExactlyOneFrame()
    {
        using LinkFrameWriter writer = new(Stream.Null);

        Assert.Throws<ArgumentException>(() => writer.WriteInboundAudioAsync(1, new short[LinkProtocol.InboundFrameSamples - 1], false, None));
        Assert.Throws<ArgumentException>(() => writer.WriteInboundAudioAsync(1, new short[LinkProtocol.InboundFrameSamples + 1], false, None));
    }

    [Fact]
    public void OutboundAudio_RejectsEmptyAndOversize()
    {
        using LinkFrameWriter writer = new(Stream.Null);

        Assert.Throws<ArgumentException>(() => writer.WriteOutboundAudioAsync(1, 1, Array.Empty<short>(), None));
        Assert.Throws<ArgumentException>(() => writer.WriteOutboundAudioAsync(1, 1, new short[LinkProtocol.MaxPayloadBytes / 2], None));
    }

    [Fact]
    public void InvalidHandshakeAndDtmfValues_AreRejected()
    {
        using LinkFrameWriter writer = new(Stream.Null);

        Assert.Throws<ArgumentException>(() => writer.WriteHelloAckAsync(new LinkHelloAck(44100, 20), None));
        Assert.Throws<ArgumentException>(() => writer.WriteHelloAckAsync(new LinkHelloAck(16000, 0), None));
        Assert.Throws<ArgumentException>(() => writer.WriteHelloAsync(default, None));
        Assert.Throws<ArgumentException>(() => writer.WriteHelloAsync(new LinkHello(1, 16000, new string('t', 70_000)), None));
        Assert.Throws<ArgumentException>(() => writer.WriteDtmfAsync(1, new LinkDtmf('E', 10), None));
    }

    [Fact]
    public async Task ConcurrentWrite_ThrowsInsteadOfInterleaving()
    {
        GatedWriteStream stream = new();
        using LinkFrameWriter writer = new(stream);

        ValueTask first = writer.WritePingAsync(1, None);
        Assert.False(first.IsCompleted);
        Assert.Throws<InvalidOperationException>(() => writer.WritePongAsync(2, None));

        stream.Release();
        await first;
        await writer.WritePongAsync(2, None);
        Assert.Equal(2, stream.Writes);
    }

    [Fact]
    public async Task AfterAFailedFrame_TheWriterIsUsableAgain()
    {
        using LinkFrameWriter writer = new(Stream.Null);
        Assert.Throws<ArgumentException>(() => writer.WriteDtmfAsync(1, new LinkDtmf('x', 10), None));

        await writer.WritePingAsync(1, None);
        Assert.Equal(1u, writer.NextSequence);
    }

    [Fact]
    public async Task WriteAfterDispose_Throws()
    {
        LinkFrameWriter writer = new(Stream.Null);
        await writer.WritePingAsync(1, None);
        writer.Dispose();
        writer.Dispose();

        Assert.Throws<ObjectDisposedException>(() => writer.WritePingAsync(2, None));
    }

    [Fact]
    public async Task AudioPath_DoesNotAllocate()
    {
        using LinkFrameWriter writer = new(Stream.Null);
        short[] inbound = new short[LinkProtocol.InboundFrameSamples];
        short[] outbound = new short[480];
        for (int i = 0; i < 16; i++)
        {
            await writer.WriteInboundAudioAsync(1, inbound, i % 2 == 0, None);
            await writer.WriteOutboundAudioAsync(1, 1, outbound, None);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++)
        {
            await writer.WriteInboundAudioAsync(1, inbound, false, None);
            await writer.WriteOutboundAudioAsync(1, 1, outbound, None);
        }
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(0, allocated);
    }

    [Fact]
    public async Task ReadingAudio_DoesNotAllocate()
    {
        short[] pcm = new short[LinkProtocol.InboundFrameSamples];
        byte[] bytes = await LinkRoundTrip.EncodeAsync(async w =>
        {
            for (int i = 0; i < 1016; i++) await w.WriteInboundAudioAsync(1, pcm, false, None);
        });
        using LinkFrameReader reader = new(new MemoryStream(bytes));
        short[] decoded = new short[LinkProtocol.InboundFrameSamples];
        for (int i = 0; i < 16; i++)
            (await reader.ReadAsync(None))!.Value.ReadPcm(decoded);

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++)
        {
            LinkFrame? frame = await reader.ReadAsync(None);
            frame!.Value.ReadPcm(decoded);
        }
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(0, allocated);
        Assert.Null(await reader.ReadAsync(None));
    }
}
