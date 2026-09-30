using System.Text;
using System.Text.Json;
using Xunit;

namespace HartsyInference.PhoneLink.Tests;

/// <summary>Pins the JSON shape documented in <c>docs/Research/PHONE_LINK_PROTOCOL.md</c>: camelCase keys, enums by name, nulls omitted.</summary>
public sealed class LinkControlJsonTests
{
    private static readonly CancellationToken None = CancellationToken.None;

    [Fact]
    public async Task CallStart_WireShape()
    {
        CallStartMessage message = new()
        {
            Direction = LinkCallDirection.Inbound, CallerId = "+15551234567", Called = "+15557654321", SipCallId = "abc@gw",
        };

        LinkFrame frame = await LinkRoundTrip.OneAsync(w => w.WriteCallStartAsync(1, message, None));

        Assert.Equal(
            """{"direction":"Inbound","callerId":"+15551234567","called":"+15557654321","sipCallId":"abc@gw","resume":false}""",
            Json(frame, 0));
        Assert.Equal(message, frame.ReadCallStart());
    }

    [Fact]
    public async Task CallStart_OmitsUnknownParties()
    {
        CallStartMessage message = new() { Direction = LinkCallDirection.Inbound, SipCallId = "k" };
        LinkFrame frame = await LinkRoundTrip.OneAsync(w => w.WriteCallStartAsync(1, message, None));

        Assert.Equal("""{"direction":"Inbound","sipCallId":"k","resume":false}""", Json(frame, 0));
    }

    [Fact]
    public async Task Event_WireShapes()
    {
        LinkEventMessage stateEvent = new() { Kind = LinkEventKind.State, State = "Speaking", TurnId = 4 };
        LinkEventMessage partialEvent = new() { Kind = LinkEventKind.TranscriptPartial, Text = "book a" };
        LinkEventMessage latencyEvent = new()
        {
            Kind = LinkEventKind.TurnLatency,
            TurnId = 3,
            Latency = new TurnLatency { SttMs = 120, LlmFirstTokenMs = 90, TotalMs = 800 },
        };
        LinkFrame state = await LinkRoundTrip.OneAsync(w => w.WriteEventAsync(1, stateEvent, None));
        LinkFrame partial = await LinkRoundTrip.OneAsync(w => w.WriteEventAsync(1, partialEvent, None));
        LinkFrame latency = await LinkRoundTrip.OneAsync(w => w.WriteEventAsync(1, latencyEvent, None));

        Assert.Equal("""{"kind":"State","turnId":4,"state":"Speaking"}""", Json(state, 0));
        Assert.Equal("""{"kind":"TranscriptPartial","text":"book a"}""", Json(partial, 0));
        Assert.Equal(
            """{"kind":"TurnLatency","turnId":3,"latency":{"sttMs":120,"llmFirstTokenMs":90,"totalMs":800}}""",
            Json(latency, 0));
    }

    [Fact]
    public async Task ToolRequestAndResult_WireShapes()
    {
        using JsonDocument args = JsonDocument.Parse("""{"target":"sip:ops@example.test"}""");
        ToolRequestMessage transfer = new() { Name = "transfer", Arguments = args.RootElement.Clone() };
        ToolRequestMessage hangup = new() { Name = "hangup" };
        ToolResultMessage unsupported = new() { Status = LinkToolStatus.Unsupported, Message = "no transfer" };
        LinkFrame request = await LinkRoundTrip.OneAsync(w => w.WriteToolRequestAsync(1, 7, transfer, None));
        LinkFrame bare = await LinkRoundTrip.OneAsync(w => w.WriteToolRequestAsync(1, 8, hangup, None));
        LinkFrame result = await LinkRoundTrip.OneAsync(w => w.WriteToolResultAsync(1, 7, unsupported, None));

        Assert.Equal("""{"name":"transfer","arguments":{"target":"sip:ops@example.test"}}""", Json(request, 4));
        Assert.Equal("""{"name":"hangup"}""", Json(bare, 4));
        Assert.Equal("""{"status":"Unsupported","message":"no transfer"}""", Json(result, 4));
    }

    [Fact]
    public async Task Error_WireShape()
    {
        LinkFrame frame = await LinkRoundTrip.OneAsync(w => w.WriteErrorAsync(0, new LinkErrorMessage { Text = "bad token" }, None));

        Assert.Equal("""{"text":"bad token"}""", Json(frame, 0));
    }

    [Fact]
    public async Task UnknownFields_AreIgnored_AndMissingRequiredFieldsAreRejected()
    {
        byte[] tolerant = Encoding.UTF8.GetBytes("""{"kind":"State","state":"Listening","future":{"x":1}}""");
        byte[] missing = Encoding.UTF8.GetBytes("""{"state":"Listening"}""");
        byte[] garbage = Encoding.UTF8.GetBytes("{not json");
        List<LinkFrame> frames = await LinkRoundTrip.DecodeAsync(new MemoryStream(await LinkRoundTrip.EncodeAsync(async w =>
        {
            await w.WriteAsync(LinkMessageType.Event, LinkFrameFlags.None, 1, tolerant, None);
            await w.WriteAsync(LinkMessageType.Event, LinkFrameFlags.None, 1, missing, None);
            await w.WriteAsync(LinkMessageType.Event, LinkFrameFlags.None, 1, garbage, None);
        })));

        Assert.Equal("Listening", frames[0].ReadEvent().State);
        Assert.Throws<LinkProtocolException>(() => frames[1].ReadEvent());
        Assert.Throws<LinkProtocolException>(() => frames[2].ReadEvent());
    }

    private static string Json(LinkFrame frame, int offset) => Encoding.UTF8.GetString(frame.Payload.Span.Slice(offset));
}
