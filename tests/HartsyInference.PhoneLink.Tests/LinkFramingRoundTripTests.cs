using System.Text.Json;
using Xunit;

namespace HartsyInference.PhoneLink.Tests;

/// <summary>Every message type through <see cref="LinkFrameWriter"/> and back through <see cref="LinkFrameReader"/> and
/// <see cref="LinkFrame"/>.</summary>
public sealed class LinkFramingRoundTripTests
{
    private static readonly CancellationToken None = CancellationToken.None;

    [Fact]
    public async Task Hello()
    {
        LinkHello hello = new(LinkProtocol.Version, 16000, "s3cret-tökén");
        LinkFrame frame = await LinkRoundTrip.OneAsync(w => w.WriteHelloAsync(hello, None));

        Assert.Equal(LinkMessageType.Hello, frame.Type);
        Assert.Equal(LinkProtocol.ConnectionCallId, frame.Header.CallId);
        Assert.Equal(hello, frame.ReadHello());
    }

    [Fact]
    public async Task HelloAck()
    {
        LinkHelloAck ack = new(24000, 60);
        LinkFrame frame = await LinkRoundTrip.OneAsync(w => w.WriteHelloAckAsync(ack, None));

        Assert.Equal(LinkMessageType.HelloAck, frame.Type);
        Assert.Equal(ack, frame.ReadHelloAck());
    }

    [Fact]
    public async Task CallStart()
    {
        CallStartMessage message = new()
        {
            Direction = LinkCallDirection.Outbound, CallerId = "+15550001111", Called = "+15552223333", SipCallId = "9f1c@gw", Resume = true,
        };
        LinkFrame frame = await LinkRoundTrip.OneAsync(w => w.WriteCallStartAsync(7, message, None));

        Assert.Equal(LinkMessageType.CallStart, frame.Type);
        Assert.Equal(7u, frame.Header.CallId);
        Assert.Equal(message, frame.ReadCallStart());
    }

    [Fact]
    public async Task CallEnd()
    {
        LinkFrame frame = await LinkRoundTrip.OneAsync(w => w.WriteCallEndAsync(7, LinkCallEndReason.RemoteHangup, None));

        Assert.Equal(LinkMessageType.CallEnd, frame.Type);
        Assert.Equal(LinkCallEndReason.RemoteHangup, frame.ReadCallEnd());
    }

    [Fact]
    public async Task InboundAudio_WithConcealedFlag()
    {
        short[] pcm = Ramp(LinkProtocol.InboundFrameSamples);
        LinkFrame frame = await LinkRoundTrip.OneAsync(w => w.WriteInboundAudioAsync(7, pcm, concealed: true, None));

        Assert.Equal(LinkMessageType.InboundAudio, frame.Type);
        Assert.True(frame.Concealed);
        Assert.Equal(LinkProtocol.InboundFrameSamples, frame.PcmSampleCount);
        short[] decoded = new short[LinkProtocol.InboundFrameSamples];
        Assert.Equal(LinkProtocol.InboundFrameSamples, frame.ReadPcm(decoded));
        Assert.Equal(pcm, decoded);
    }

    [Fact]
    public async Task OutboundAudio_CarriesTurnIdThenPcm()
    {
        short[] pcm = Ramp(480);
        LinkFrame frame = await LinkRoundTrip.OneAsync(w => w.WriteOutboundAudioAsync(7, 42, pcm, None));

        Assert.Equal(LinkMessageType.OutboundAudio, frame.Type);
        Assert.False(frame.Concealed);
        Assert.Equal(42u, frame.ReadTurnId());
        Assert.Equal(480, frame.PcmSampleCount);
        short[] decoded = new short[600];
        Assert.Equal(480, frame.ReadPcm(decoded));
        Assert.Equal(pcm, decoded.AsSpan(0, 480).ToArray());
    }

    [Fact]
    public async Task OutboundEnd_Flush_FlushAck()
    {
        LinkFrame end = await LinkRoundTrip.OneAsync(w => w.WriteOutboundEndAsync(7, 42, None));
        LinkFrame flush = await LinkRoundTrip.OneAsync(w => w.WriteFlushAsync(7, 43, None));
        LinkFrame ack = await LinkRoundTrip.OneAsync(w => w.WriteFlushAckAsync(7, new LinkFlushAck(43, 350), None));

        Assert.Equal(LinkMessageType.OutboundEnd, end.Type);
        Assert.Equal(42u, end.ReadTurnId());
        Assert.Equal(LinkMessageType.Flush, flush.Type);
        Assert.Equal(43u, flush.ReadTurnId());
        Assert.Equal(LinkMessageType.FlushAck, ack.Type);
        Assert.Equal(43u, ack.ReadTurnId());
        Assert.Equal(new LinkFlushAck(43, 350), ack.ReadFlushAck());
    }

    [Fact]
    public async Task Event()
    {
        LinkEventMessage message = new()
        {
            Kind = LinkEventKind.TurnLatency,
            TurnId = 3,
            Latency = new TurnLatency { SttMs = 120, LlmFirstTokenMs = 90, TtsFirstChunkMs = 200, TotalMs = 800 },
        };
        LinkFrame frame = await LinkRoundTrip.OneAsync(w => w.WriteEventAsync(7, message, None));

        Assert.Equal(LinkMessageType.Event, frame.Type);
        Assert.Equal(message, frame.ReadEvent());
    }

    [Fact]
    public async Task DtmfEvent()
    {
        LinkDtmf dtmf = new('#', 120);
        LinkFrame frame = await LinkRoundTrip.OneAsync(w => w.WriteDtmfAsync(7, dtmf, None));

        Assert.Equal(LinkMessageType.DtmfEvent, frame.Type);
        Assert.Equal(dtmf, frame.ReadDtmf());
    }

    [Fact]
    public async Task ToolRequest_WithArguments()
    {
        using JsonDocument args = JsonDocument.Parse("""{"digits":"123#","gapMs":80}""");
        ToolRequestMessage message = new() { Name = "send_dtmf", Arguments = args.RootElement.Clone() };
        LinkFrame frame = await LinkRoundTrip.OneAsync(w => w.WriteToolRequestAsync(7, 99, message, None));

        Assert.Equal(LinkMessageType.ToolRequest, frame.Type);
        ToolRequestMessage decoded = frame.ReadToolRequest(out uint requestId);
        Assert.Equal(99u, requestId);
        Assert.Equal("send_dtmf", decoded.Name);
        Assert.Equal("123#", decoded.Arguments!.Value.GetProperty("digits").GetString());
        Assert.Equal(80, decoded.Arguments!.Value.GetProperty("gapMs").GetInt32());
    }

    [Fact]
    public async Task ToolRequest_WithoutArguments()
    {
        ToolRequestMessage hangup = new() { Name = "hangup" };
        LinkFrame frame = await LinkRoundTrip.OneAsync(w => w.WriteToolRequestAsync(7, 100, hangup, None));

        ToolRequestMessage decoded = frame.ReadToolRequest(out uint requestId);
        Assert.Equal(100u, requestId);
        Assert.Equal("hangup", decoded.Name);
        Assert.Null(decoded.Arguments);
    }

    [Fact]
    public async Task ToolResult()
    {
        ToolResultMessage message = new() { Status = LinkToolStatus.Failed, Message = "no such target" };
        LinkFrame frame = await LinkRoundTrip.OneAsync(w => w.WriteToolResultAsync(7, 99, message, None));

        Assert.Equal(LinkMessageType.ToolResult, frame.Type);
        Assert.Equal(message, frame.ReadToolResult(out uint requestId));
        Assert.Equal(99u, requestId);
    }

    [Fact]
    public async Task Ping_Pong()
    {
        LinkFrame ping = await LinkRoundTrip.OneAsync(w => w.WritePingAsync(ulong.MaxValue - 5, None));
        LinkFrame pong = await LinkRoundTrip.OneAsync(w => w.WritePongAsync(123456789012345UL, None));

        Assert.Equal(LinkMessageType.Ping, ping.Type);
        Assert.Equal(ulong.MaxValue - 5, ping.ReadTimestampNs());
        Assert.Equal(LinkMessageType.Pong, pong.Type);
        Assert.Equal(123456789012345UL, pong.ReadTimestampNs());
        Assert.Equal(LinkProtocol.ConnectionCallId, ping.Header.CallId);
    }

    [Fact]
    public async Task Error()
    {
        LinkErrorMessage message = new() { Text = "unknown token" };
        LinkFrame frame = await LinkRoundTrip.OneAsync(w => w.WriteErrorAsync(0, message, None));

        Assert.Equal(LinkMessageType.Error, frame.Type);
        Assert.Equal(message, frame.ReadError());
    }

    [Fact]
    public async Task EveryMessageType_HasARoundTripAbove()
    {
        // Guards the suite itself: a type added to the enum without a typed writer/reader pair fails here.
        List<LinkMessageType> seen = new();
        byte[] bytes = await LinkRoundTrip.EncodeAsync(async w =>
        {
            await w.WriteHelloAsync(new LinkHello(1, 16000, ""), None);
            await w.WriteHelloAckAsync(new LinkHelloAck(16000, 20), None);
            await w.WriteCallStartAsync(1, new CallStartMessage { Direction = LinkCallDirection.Inbound, SipCallId = "x" }, None);
            await w.WriteCallEndAsync(1, LinkCallEndReason.Completed, None);
            await w.WriteInboundAudioAsync(1, new short[LinkProtocol.InboundFrameSamples], false, None);
            await w.WriteOutboundAudioAsync(1, 1, new short[10], None);
            await w.WriteOutboundEndAsync(1, 1, None);
            await w.WriteFlushAsync(1, 1, None);
            await w.WriteFlushAckAsync(1, new LinkFlushAck(1, 0), None);
            await w.WriteEventAsync(1, new LinkEventMessage { Kind = LinkEventKind.State, State = "Listening" }, None);
            await w.WriteDtmfAsync(1, new LinkDtmf('5', 80), None);
            await w.WriteToolRequestAsync(1, 1, new ToolRequestMessage { Name = "hold" }, None);
            await w.WriteToolResultAsync(1, 1, new ToolResultMessage { Status = LinkToolStatus.Ok }, None);
            await w.WritePingAsync(1, None);
            await w.WritePongAsync(1, None);
            await w.WriteErrorAsync(1, new LinkErrorMessage { Text = "x" }, None);
        });
        foreach (LinkFrame frame in await LinkRoundTrip.DecodeAsync(new MemoryStream(bytes)))
            seen.Add(frame.Type);

        LinkMessageType[] all = Enum.GetValues<LinkMessageType>();
        Assert.Equal(all.OrderBy(t => t), seen.OrderBy(t => t));
        Assert.Equal(all.Length, seen.Count);
    }

    [Fact]
    public async Task Decoders_RejectTheWrongFrameType()
    {
        LinkFrame ping = await LinkRoundTrip.OneAsync(w => w.WritePingAsync(1, None));

        Assert.Throws<LinkProtocolException>(() => ping.ReadHello());
        Assert.Throws<LinkProtocolException>(() => ping.ReadTurnId());
        Assert.Throws<LinkProtocolException>(() => ping.ReadPcm(new short[1]));
        Assert.Throws<LinkProtocolException>(() => ping.ReadEvent());
    }

    [Fact]
    public async Task Decoders_RejectTheWrongPayloadSize()
    {
        byte[] bytes = await LinkRoundTrip.EncodeAsync(async w =>
        {
            await w.WriteAsync(LinkMessageType.HelloAck, LinkFrameFlags.None, 0, new byte[5], None);
            await w.WriteAsync(LinkMessageType.OutboundAudio, LinkFrameFlags.None, 1, new byte[7], None);
            await w.WriteAsync(LinkMessageType.DtmfEvent, LinkFrameFlags.None, 1, new byte[] { (byte)'Z', 0, 0 }, None);
            byte[] shortToken = [1, 0, 0x80, 0x3E, 0, 0, 9, 0, (byte)'a'];
            await w.WriteAsync(LinkMessageType.Hello, LinkFrameFlags.None, 0, shortToken, None);
        });
        List<LinkFrame> frames = await LinkRoundTrip.DecodeAsync(new MemoryStream(bytes));

        Assert.Throws<LinkProtocolException>(() => frames[0].ReadHelloAck());
        Assert.Throws<LinkProtocolException>(() => frames[1].ReadPcm(new short[8]));
        Assert.Throws<LinkProtocolException>(() => frames[2].ReadDtmf());
        Assert.Throws<LinkProtocolException>(() => frames[3].ReadHello());
    }

    private static short[] Ramp(int count)
    {
        short[] pcm = new short[count];
        for (int i = 0; i < count; i++) pcm[i] = (short)(i * 97 - 16000);
        return pcm;
    }
}
