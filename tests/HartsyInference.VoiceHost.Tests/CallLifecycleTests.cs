using System.Net.Sockets;
using HartsyInference.PhoneLink;
using HartsyInference.Voice;
using HartsyInference.VoiceHost.Link;
using HartsyInference.VoiceHost.Tests.Support;
using HartsyInference.VoiceHost.Tools;
using Xunit;

namespace HartsyInference.VoiceHost.Tests;

/// <summary>Calls over a live link: <c>CallStart</c> creates and starts a session with every tool bound, session events
/// reach the gateway as <c>Event</c> frames, <c>CallEnd</c> ends and disposes the session without echoing, a resumed call
/// gets a fresh session and the apology, a lost link ends its calls, a session that fails anywhere ends only its own call
/// with <c>CallEnd(Failed)</c>, and stopping the host ends calls with <c>LocalHangup</c> and removes the socket.</summary>
public sealed class CallLifecycleTests
{
    [Fact]
    public async Task CallStartCreatesAStartedSessionWithEveryToolAndItsEventsReachTheGateway()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync(callId: 7);

        Assert.Equal(7u, session.CallId);
        Assert.Equal(VoiceHostTools.Names, session.Tools.Names);
        session.Raise(VoiceAgentEventKind.StateChanged, turnId: 0, state: VoiceAgentState.Listening);
        session.Raise(VoiceAgentEventKind.UserTranscript, turnId: 1, text: "what time is it");

        List<FakeGateway.RecordedFrame> events = gateway.WaitFor(LinkMessageType.Event, 2);
        Assert.All(events, e => Assert.Equal(7u, e.Header.CallId));
        Assert.Equal(LinkEventKind.State, events[0].Event.Kind);
        Assert.Equal("Listening", events[0].Event.State);
        Assert.Null(events[0].Event.TurnId);
        Assert.Equal(LinkEventKind.TranscriptFinal, events[1].Event.Kind);
        Assert.Equal("what time is it", events[1].Event.Text);
        Assert.Equal(1u, events[1].Event.TurnId);
    }

    [Fact]
    public async Task CallEndFromTheGatewayEndsAndDisposesTheSessionWithoutEchoingIt()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        gateway.SendCallEnd(1, LinkCallEndReason.RemoteHangup);

        Assert.True(HostRig.Wait(() => session.EndCalled && session.Disposed), "the session outlived its call.");
        Assert.True(HostRig.Wait(() => rig.Call(1) is null));
        await Task.Delay(100);
        Assert.Empty(gateway.FramesOf(LinkMessageType.CallEnd));
        Assert.False(gateway.IsClosed);
    }

    [Fact]
    public async Task TheGreetingOpensANewCall()
    {
        await using HostRig rig = HostRig.Start(options => options with { Greeting = "Hello, how can I help?", ResumeApology = "Sorry about that." });
        (_, FakeCallSession session) = await rig.StartCallAsync();

        Assert.True(HostRig.Wait(() => session.Spoken.Length == 1));
        Assert.Equal("Hello, how can I help?", Assert.Single(session.Spoken));
    }

    [Fact]
    public async Task AResumedCallGetsAFreshSessionAndTheApology()
    {
        await using HostRig rig = HostRig.Start(options => options with { Greeting = "Hello.", ResumeApology = "Sorry, the line dropped." });
        FakeGateway gateway = rig.Connect();
        gateway.SendCallStart(3, resume: true);
        FakeCallSession session = await rig.Factory.WaitForSessionAsync();

        Assert.True(HostRig.Wait(() => session.Spoken.Length == 1));
        Assert.Equal("Sorry, the line dropped.", Assert.Single(session.Spoken));
        Assert.True(rig.Call(3)!.Resumed);
    }

    [Fact]
    public async Task ALostLinkEndsItsCallsAndTheReconnectResumesThemFresh()
    {
        await using HostRig rig = HostRig.Start(options => options with { ResumeApology = "Sorry, the line dropped." });
        (FakeGateway first, FakeCallSession lost) = await rig.StartCallAsync();
        first.Dispose();

        Assert.True(HostRig.Wait(() => lost.EndCalled && lost.Disposed), "a call on a dead link kept its session.");
        Assert.True(HostRig.Wait(() => rig.Server.Current is null));
        FakeGateway second = rig.Connect();
        second.SendCallStart(1, resume: true);
        FakeCallSession fresh = await rig.Factory.WaitForSessionAsync(1);

        Assert.NotSame(lost, fresh);
        Assert.True(HostRig.Wait(() => fresh.Spoken.Length == 1));
        Assert.Equal("Sorry, the line dropped.", fresh.Spoken[0]);
    }

    [Fact]
    public async Task ASessionThatFailsToStartEndsOnlyItsCallWithFailed()
    {
        await using HostRig rig = HostRig.Start();
        FakeGateway gateway = rig.Connect();
        rig.Factory.StartFailure = new InvalidOperationException("the model set is gone");
        gateway.SendCallStart(1);

        FakeGateway.RecordedFrame end = gateway.WaitFor(LinkMessageType.CallEnd)[0];
        Assert.Equal(1u, end.Header.CallId);
        Assert.Equal(LinkCallEndReason.Failed, end.AsFrame().ReadCallEnd());
        Assert.True(HostRig.Wait(() => rig.Factory.Sessions[0].Disposed));
        await AssertTheHostStillTakesCallsAsync(rig, gateway, callId: 2, sessionIndex: 1);
    }

    [Fact]
    public async Task ASessionThatCannotBeCreatedEndsOnlyItsCallWithFailed()
    {
        await using HostRig rig = HostRig.Start();
        FakeGateway gateway = rig.Connect();
        rig.Factory.CreateFailure = new InvalidOperationException("no VAD weights");
        gateway.SendCallStart(1);

        Assert.Equal(LinkCallEndReason.Failed, gateway.WaitFor(LinkMessageType.CallEnd)[0].AsFrame().ReadCallEnd());
        await AssertTheHostStillTakesCallsAsync(rig, gateway, callId: 2, sessionIndex: 0);
    }

    [Fact]
    public async Task ASessionThatEndsOnItsOwnEndsItsCallWithFailed()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        session.EndOnItsOwn();

        FakeGateway.RecordedFrame end = gateway.WaitFor(LinkMessageType.CallEnd)[0];
        Assert.Equal(LinkCallEndReason.Failed, end.AsFrame().ReadCallEnd());
        Assert.True(HostRig.Wait(() => session.Disposed && rig.Call(1) is null));
        await AssertTheHostStillTakesCallsAsync(rig, gateway, callId: 2, sessionIndex: 1);
    }

    [Fact]
    public async Task ASessionThatThrowsOnTheSenderEndsOnlyItsCallWithFailed()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        session.ReadFailure = new InvalidOperationException("the outbound queue broke");

        Assert.Equal(LinkCallEndReason.Failed, gateway.WaitFor(LinkMessageType.CallEnd)[0].AsFrame().ReadCallEnd());
        Assert.True(HostRig.Wait(() => session.Disposed));
        Assert.False(gateway.IsClosed, "a session fault took the whole link down.");
        await AssertTheHostStillTakesCallsAsync(rig, gateway, callId: 2, sessionIndex: 1);
    }

    [Fact]
    public async Task KeysReachTheSession()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        gateway.SendDtmf(1, '5');
        gateway.SendDtmf(1, '#');

        Assert.True(HostRig.Wait(() => session.Dtmf.Length == 2));
        Assert.Equal(['5', '#'], session.Dtmf);
    }

    [Fact]
    public async Task StoppingTheHostEndsCallsWithLocalHangupAndRemovesTheSocket()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        await rig.Server.StopAsync(LinkCallEndReason.LocalHangup);

        FakeGateway.RecordedFrame end = Assert.Single(gateway.FramesOf(LinkMessageType.CallEnd));
        Assert.Equal(LinkCallEndReason.LocalHangup, end.AsFrame().ReadCallEnd());
        Assert.True(gateway.WaitForClose());
        Assert.True(session.Disposed);
        Assert.False(File.Exists(rig.SocketPath));
    }

    [Fact]
    public async Task TheSocketFileGetsTheConfiguredMode()
    {
        if (OperatingSystem.IsWindows())
        {
            return;
        }
        UnixFileMode ownerOnly = UnixFileMode.UserRead | UnixFileMode.UserWrite;
        await using HostRig rig = HostRig.Start(options => options with { SocketMode = ownerOnly });
        Assert.Equal(ownerOnly, File.GetUnixFileMode(rig.SocketPath));
    }

    [Fact]
    public async Task AStaleSocketFileIsReplacedButALiveListenerIsNot()
    {
        string directory = HostRig.NewSocketDirectory();
        string path = Path.Combine(directory, "phone.sock");
        try
        {
            using (Socket dead = new(AddressFamily.Unix, SocketType.Stream, ProtocolType.Unspecified))
            {
                dead.Bind(new UnixDomainSocketEndPoint(path));
            }
            Assert.True(File.Exists(path), "the stand-in for a killed host left no socket file.");
            await using (PhoneLinkServer revived = new(new PhoneLinkServerOptions { SocketPath = path }, new FakeSessionFactory()))
            {
                revived.Start();
                using FakeGateway gateway = FakeGateway.Connect(path);
                Assert.Equal(LinkMessageType.HelloAck, gateway.Hello("").Header.Type);

                PhoneLinkServer rival = new(new PhoneLinkServerOptions { SocketPath = path }, new FakeSessionFactory());
                Assert.Throws<InvalidOperationException>(rival.Start);
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static async Task AssertTheHostStillTakesCallsAsync(HostRig rig, FakeGateway gateway, uint callId, int sessionIndex)
    {
        gateway.SendCallStart(callId);
        FakeCallSession next = await rig.Factory.WaitForSessionAsync(sessionIndex);
        rig.WaitUntilReady(callId);
        Assert.Equal(callId, next.CallId);
        Assert.False(gateway.IsClosed);
    }
}
