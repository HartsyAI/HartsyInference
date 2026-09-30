using Xunit;

namespace HartsyInference.PhoneLink.Tests;

public sealed class LinkFrameReaderTests
{
    private static readonly CancellationToken None = CancellationToken.None;

    [Fact]
    public async Task PartialReads_AtEveryChunkSize_ReassembleBothFrames()
    {
        (byte[] bytes, List<LinkFrame> expected) = await TwoFrameStreamAsync();

        for (int chunk = 1; chunk <= bytes.Length; chunk++)
        {
            List<LinkFrame> frames = await LinkRoundTrip.DecodeAsync(new ChunkedReadStream(bytes, chunk, chunk));
            AssertSame(expected, frames, $"chunk={chunk}");
        }
    }

    [Fact]
    public async Task PartialReads_SplitAtEveryByteBoundary_ReassembleBothFrames()
    {
        (byte[] bytes, List<LinkFrame> expected) = await TwoFrameStreamAsync();

        for (int split = 1; split < bytes.Length; split++)
        {
            List<LinkFrame> frames = await LinkRoundTrip.DecodeAsync(new ChunkedReadStream(bytes, split, bytes.Length));
            AssertSame(expected, frames, $"split={split}");
        }
    }

    [Fact]
    public async Task OversizePayload_IsRejectedFromTheHeaderAlone()
    {
        byte[] header = new byte[LinkFrameHeader.Size];
        new LinkFrameHeader(LinkProtocol.MaxPayloadBytes + 1, LinkMessageType.Event, LinkFrameFlags.None, 1, 0).Write(header);
        using LinkFrameReader reader = new(new MemoryStream(header));

        LinkProtocolException ex = await Assert.ThrowsAsync<LinkProtocolException>(async () => await reader.ReadAsync(None));

        Assert.Contains("1048577", ex.Message);
        Assert.Contains("1048576", ex.Message);
    }

    [Fact]
    public async Task PayloadAtTheCap_IsAccepted()
    {
        byte[] payload = new byte[LinkProtocol.MaxPayloadBytes];
        new Random(3).NextBytes(payload);

        LinkFrame frame = await LinkRoundTrip.OneAsync(w => w.WriteAsync(LinkMessageType.Event, LinkFrameFlags.None, 1, payload, None));

        Assert.Equal((uint)LinkProtocol.MaxPayloadBytes, frame.Header.PayloadLength);
        Assert.Equal(payload, frame.Payload.ToArray());
    }

    [Fact]
    public async Task StreamEndingInsideAFrame_Throws()
    {
        byte[] bytes = await LinkRoundTrip.EncodeAsync(w => w.WritePingAsync(1, None));
        using LinkFrameReader headerOnly = new(new MemoryStream(bytes, 0, LinkFrameHeader.Size + 3));
        using LinkFrameReader partialHeader = new(new MemoryStream(bytes, 0, 5));

        await Assert.ThrowsAsync<LinkProtocolException>(async () => await headerOnly.ReadAsync(None));
        await Assert.ThrowsAsync<LinkProtocolException>(async () => await partialHeader.ReadAsync(None));
    }

    [Fact]
    public async Task CleanEndOfStream_ReturnsNull()
    {
        byte[] bytes = await LinkRoundTrip.EncodeAsync(w => w.WritePingAsync(1, None));
        using LinkFrameReader empty = new(new MemoryStream());
        using LinkFrameReader one = new(new MemoryStream(bytes));

        Assert.Null(await empty.ReadAsync(None));
        Assert.NotNull(await one.ReadAsync(None));
        Assert.Null(await one.ReadAsync(None));
        Assert.Null(await one.ReadAsync(None));
    }

    [Fact]
    public async Task BufferGrowsForALargeFrameAndKeepsTheBytesAlreadyRead()
    {
        byte[] big = new byte[200_000];
        new Random(5).NextBytes(big);
        byte[] bytes = await LinkRoundTrip.EncodeAsync(async w =>
        {
            await w.WritePingAsync(9, None);
            await w.WriteAsync(LinkMessageType.Event, LinkFrameFlags.None, 1, big, None);
            await w.WritePongAsync(10, None);
        });

        List<LinkFrame> frames = await LinkRoundTrip.DecodeAsync(new ChunkedReadStream(bytes, 7000, 7000));

        Assert.Equal(3, frames.Count);
        Assert.Equal(9UL, frames[0].ReadTimestampNs());
        Assert.Equal(big, frames[1].Payload.ToArray());
        Assert.Equal(10UL, frames[2].ReadTimestampNs());
    }

    [Fact]
    public async Task DisposeWhileAReadIsInFlight_LeavesTheBufferToTheGcAndEndsTheRead()
    {
        byte[] bytes = await LinkRoundTrip.EncodeAsync(w => w.WritePingAsync(1, None));
        GatedReadStream stream = new(bytes);
        LinkFrameReader reader = new(stream);
        ValueTask<LinkFrame?> pending = reader.ReadAsync(None);
        Assert.False(pending.IsCompleted);
        byte[] buffer = PoolProbe.ArrayOf(stream.LastRead);

        reader.Dispose();

        Assert.False(PoolProbe.IsInPool(buffer));
        stream.Release();
        Assert.Null(await pending);
        Assert.False(PoolProbe.IsInPool(buffer));
        await Assert.ThrowsAsync<ObjectDisposedException>(async () => await reader.ReadAsync(None));
    }

    [Fact]
    public async Task DisposeAfterTheLastRead_ReturnsTheBufferToThePool()
    {
        byte[] bytes = await LinkRoundTrip.EncodeAsync(w => w.WritePingAsync(1, None));
        GatedReadStream stream = new(bytes, gated: false);
        LinkFrameReader reader = new(stream);
        Assert.NotNull(await reader.ReadAsync(None));
        byte[] buffer = PoolProbe.ArrayOf(stream.LastRead);

        reader.Dispose();

        Assert.True(PoolProbe.IsInPool(buffer));
    }

    [Fact]
    public async Task ConcurrentRead_Throws()
    {
        GatedReadStream stream = new([]);
        using LinkFrameReader reader = new(stream);
        ValueTask<LinkFrame?> first = reader.ReadAsync(None);

        Assert.Throws<InvalidOperationException>(() => reader.ReadAsync(None));

        stream.Release();
        Assert.Null(await first);
        Assert.Null(await reader.ReadAsync(None));
    }

    [Fact]
    public async Task ReadAfterDispose_Throws()
    {
        LinkFrameReader reader = new(new MemoryStream());
        reader.Dispose();
        reader.Dispose();

        await Assert.ThrowsAsync<ObjectDisposedException>(async () => await reader.ReadAsync(None));
    }

    [Fact]
    public void Constructor_RejectsUnreadableStream()
    {
        Assert.Throws<ArgumentException>(() => new LinkFrameReader(new GatedWriteStream()));
    }

    private static async Task<(byte[] Bytes, List<LinkFrame> Expected)> TwoFrameStreamAsync()
    {
        short[] pcm = new short[160];
        for (int i = 0; i < pcm.Length; i++) pcm[i] = (short)(i * 31);
        byte[] bytes = await LinkRoundTrip.EncodeAsync(async w =>
        {
            await w.WriteOutboundAudioAsync(5, 2, pcm, None);
            await w.WriteEventAsync(5, new LinkEventMessage { Kind = LinkEventKind.TranscriptFinal, Text = "hello there" }, None);
        });
        List<LinkFrame> expected = await LinkRoundTrip.DecodeAsync(new MemoryStream(bytes));
        Assert.Equal(2, expected.Count);
        return (bytes, expected);
    }

    private static void AssertSame(List<LinkFrame> expected, List<LinkFrame> actual, string context)
    {
        Assert.True(expected.Count == actual.Count, $"{context}: expected {expected.Count} frames, got {actual.Count}");
        for (int i = 0; i < expected.Count; i++)
        {
            Assert.True(expected[i].Header == actual[i].Header, $"{context}: frame {i} header differs");
            Assert.True(expected[i].Payload.Span.SequenceEqual(actual[i].Payload.Span), $"{context}: frame {i} payload differs");
        }
    }
}
