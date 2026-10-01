using HartsyInference.PhoneLink;
using HartsyInference.VoiceHost.Tests.Support;
using Xunit;

namespace HartsyInference.VoiceHost.Tests;

/// <summary>The host's side of <c>Hello</c>: the right token gets <c>HelloAck</c> with the outbound rate and the 20 ms
/// frame bound; a wrong token, a rate other than 16 kHz, another protocol version, a first frame that is not
/// <c>Hello</c> or no <c>Hello</c> at all gets <c>Error</c> and a closed socket, never the token in a log line; at most
/// four connections wait in their handshake at once; a connection replaces the current one only once its own
/// <c>Hello</c> has passed.</summary>
public sealed class PhoneLinkHandshakeTests
{
    [Fact]
    public async Task TheRightTokenGetsHelloAckWithTheRates()
    {
        await using HostRig rig = HostRig.Start();
        using FakeGateway gateway = FakeGateway.Connect(rig.SocketPath);
        FakeGateway.RecordedFrame answer = gateway.Hello(HostRig.Token);

        Assert.Equal(LinkMessageType.HelloAck, answer.Header.Type);
        LinkHelloAck ack = answer.AsFrame().ReadHelloAck();
        Assert.Equal(16_000u, ack.OutboundRate);
        Assert.Equal((ushort)20, ack.MaxFrameMs);
        Assert.Equal(0u, answer.Header.Sequence);
        Assert.True(HostRig.Wait(() => rig.Server.Current is not null));
        Assert.Equal(0, rig.Server.Refused);
    }

    [Fact]
    public async Task AWrongTokenIsRefusedWithErrorAndNeverLogged()
    {
        using LogCapture log = new();
        await using HostRig rig = HostRig.Start();
        using FakeGateway gateway = FakeGateway.Connect(rig.SocketPath);
        FakeGateway.RecordedFrame answer = gateway.Hello("not-the-token");

        Assert.Equal(LinkMessageType.Error, answer.Header.Type);
        Assert.Equal(LinkProtocol.ConnectionCallId, answer.Header.CallId);
        Assert.Contains("token", answer.AsFrame().ReadError().Text, StringComparison.Ordinal);
        Assert.True(gateway.WaitForClose(), "the host kept a connection with a wrong token open.");
        Assert.Empty(gateway.FramesOf(LinkMessageType.HelloAck));
        Assert.Null(rig.Server.Current);
        Assert.Equal(1, rig.Server.Refused);
        Assert.DoesNotContain("not-the-token", log.All, StringComparison.Ordinal);
        Assert.DoesNotContain(HostRig.Token, log.All, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AnInboundRateOtherThan16kIsRefused()
    {
        await using HostRig rig = HostRig.Start();
        using FakeGateway gateway = FakeGateway.Connect(rig.SocketPath);
        FakeGateway.RecordedFrame answer = gateway.Hello(HostRig.Token, inboundRate: 8_000);

        Assert.Equal(LinkMessageType.Error, answer.Header.Type);
        Assert.Contains("16000", answer.AsFrame().ReadError().Text, StringComparison.Ordinal);
        Assert.True(gateway.WaitForClose());
        Assert.Null(rig.Server.Current);
    }

    [Fact]
    public async Task AnotherProtocolVersionIsRefused()
    {
        await using HostRig rig = HostRig.Start();
        using FakeGateway gateway = FakeGateway.Connect(rig.SocketPath);
        FakeGateway.RecordedFrame answer = gateway.Hello(HostRig.Token, version: 2);

        Assert.Equal(LinkMessageType.Error, answer.Header.Type);
        Assert.Contains("version 2", answer.AsFrame().ReadError().Text, StringComparison.Ordinal);
        Assert.True(gateway.WaitForClose());
    }

    [Fact]
    public async Task AFirstFrameThatIsNotHelloIsRefused()
    {
        await using HostRig rig = HostRig.Start();
        using FakeGateway gateway = FakeGateway.Connect(rig.SocketPath);
        gateway.SendCallStart(1);

        Assert.Equal(LinkMessageType.Error, gateway.WaitFor(LinkMessageType.Error)[0].Header.Type);
        Assert.True(gateway.WaitForClose());
        Assert.Empty(rig.Factory.Sessions);
    }

    [Fact]
    public async Task AClientThatNeverSaysHelloIsDropped()
    {
        await using HostRig rig = HostRig.Start(options => options with { HandshakeTimeoutMs = 200 });
        using FakeGateway gateway = FakeGateway.Connect(rig.SocketPath);

        Assert.Contains("Hello", gateway.WaitFor(LinkMessageType.Error)[0].AsFrame().ReadError().Text, StringComparison.Ordinal);
        Assert.True(gateway.WaitForClose());
    }

    [Fact]
    public async Task AtMostFourConnectionsWaitInTheirHandshakeAtOnce()
    {
        await using HostRig rig = HostRig.Start(options => options with { HandshakeTimeoutMs = 30_000 });
        List<FakeGateway> silent = [];
        try
        {
            for (int i = 0; i < Link.PhoneLinkServer.MaxPendingHandshakes; i++)
            {
                silent.Add(FakeGateway.Connect(rig.SocketPath));
            }
            Assert.True(HostRig.Wait(() => rig.Server.PendingHandshakes == Link.PhoneLinkServer.MaxPendingHandshakes));
            using FakeGateway extra = FakeGateway.Connect(rig.SocketPath);

            Assert.True(extra.WaitForClose(2_000), "a fifth connection still in its handshake was kept.");
            Assert.Equal(1, rig.Server.TurnedAway);
            silent[0].Dispose();
            Assert.True(HostRig.Wait(() => rig.Server.PendingHandshakes == Link.PhoneLinkServer.MaxPendingHandshakes - 1),
                "the closed connection's slot never came back.");
            FakeGateway gateway = rig.Connect();
            Assert.False(gateway.IsClosed);
        }
        finally
        {
            foreach (FakeGateway gateway in silent)
            {
                gateway.Dispose();
            }
        }
    }

    [Fact]
    public async Task ANewConnectionReplacesTheCurrentOneOnlyAfterItsHelloPasses()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway first, FakeCallSession session) = await rig.StartCallAsync();

        using FakeGateway impostor = FakeGateway.Connect(rig.SocketPath);
        Assert.Equal(LinkMessageType.Error, impostor.Hello("wrong").Header.Type);
        Assert.False(first.IsClosed, "a connection that failed its Hello displaced the gateway.");
        Assert.False(session.EndCalled);

        rig.Connect();
        Assert.True(first.WaitForClose(), "the replaced connection stayed open.");
        Assert.True(HostRig.Wait(() => session.EndCalled && session.Disposed), "the replaced connection's call was not ended.");
    }

    [Fact]
    public async Task APingIsAnsweredWithItsOwnTimestamp()
    {
        await using HostRig rig = HostRig.Start();
        FakeGateway gateway = rig.Connect();
        gateway.SendPing(123_456_789UL);

        FakeGateway.RecordedFrame pong = gateway.WaitFor(LinkMessageType.Pong)[0];
        Assert.Equal(123_456_789UL, pong.AsFrame().ReadTimestampNs());
    }
}
