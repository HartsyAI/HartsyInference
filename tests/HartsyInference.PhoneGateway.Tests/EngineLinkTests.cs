using System.Collections.Concurrent;
using HartsyInference.PhoneGateway.Media;
using HartsyInference.PhoneGateway.Tests.Support;
using HartsyInference.PhoneGateway.Transport;
using HartsyInference.PhoneLink;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.PhoneGateway.Tests;

/// <summary>The link is the gateway's only path to the models, and every failure here is silent on a call: a lost
/// frame, a stale turn played after a barge-in, a reconnect that never re-announces the call, or an outage that
/// leaves the caller hanging. A fake host on a temporary Unix socket stands in for the voice host.</summary>
public sealed class EngineLinkTests
{
    private const int WaitMs = 5000;

    private readonly ITestOutputHelper _output;
    public EngineLinkTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Handshake_ConnectsAndReportsTheHostRates()
    {
        using FakeLinkHost host = new() { OutboundRate = 24000, MaxFrameMs = 40, ExpectedToken = "secret" };
        host.Start();
        using EngineLink link = new(Options(host, token: "secret"));
        LinkHelloAck? ack = null;
        using ManualResetEventSlim connected = new(false);
        link.Connected = a =>
        {
            ack = a;
            connected.Set();
        };
        link.Start();
        Assert.True(connected.Wait(WaitMs));
        Assert.True(link.IsConnected);
        Assert.Equal(24000u, ack!.Value.OutboundRate);
        Assert.Equal(40, ack.Value.MaxFrameMs);
        Assert.Equal(24000u, link.OutboundRate);
        Assert.True(host.WaitUntil(() => host.FramesOf(LinkMessageType.Hello).Count == 1, WaitMs));
        LinkHello hello = host.FramesOf(LinkMessageType.Hello)[0].AsFrame().ReadHello();
        Assert.Equal(LinkProtocol.Version, hello.Version);
        Assert.Equal((uint)LinkProtocol.InboundSampleRate, hello.InboundRate);
        Assert.Equal("secret", hello.Token);
    }

    [Fact]
    public void Audio_FlowsBothWaysWithContinuousSequence()
    {
        using FakeLinkHost host = new();
        host.Start();
        using EngineLink link = new(Options(host));
        ConcurrentQueue<(uint CallId, uint TurnId, short[] Pcm)> received = new();
        link.OutboundAudio = (callId, turnId, pcm) => received.Enqueue((callId, turnId, pcm.ToArray()));
        using ManualResetEventSlim connected = new(false);
        link.Connected = _ => connected.Set();
        link.Start();
        Assert.True(connected.Wait(WaitMs));
        link.SendCallStart(1, new CallStartMessage { Direction = LinkCallDirection.Inbound, SipCallId = "abc" });
        short[] frame = new short[LinkProtocol.InboundFrameSamples];
        // Each frame waits for the writer to take it: this test is about order and continuity on a busy box, and the
        // ten-deep drop-oldest lane is covered by AudioLane_DropsTheOldestFrameWhenFull.
        for (int i = 0; i < 25; i++)
        {
            Array.Fill(frame, (short)i);
            long sent = link.FramesSent;
            Assert.True(link.TryEnqueueInboundAudio(1, frame, concealed: i % 5 == 0));
            Assert.True(host.WaitUntil(() => link.FramesSent > sent, WaitMs), $"the writer never took frame {i}");
        }
        Assert.True(host.WaitUntil(() => host.AudioFramesReceived >= 25, WaitMs),
            $"host audio={host.AudioFramesReceived} link sent={link.FramesSent} laneDropped={link.AudioLaneDropped} connected={link.IsConnected}");
        for (int i = 0; i < 10; i++)
        {
            short[] pcm = new short[320];
            Array.Fill(pcm, (short)(100 + i));
            host.SendOutboundAudio(1, 1, pcm);
        }
        Assert.True(host.WaitUntil(() => received.Count >= 10, WaitMs), $"received={received.Count} link received={link.FramesReceived}");

        List<FakeLinkHost.RecordedFrame> all = host.AllFrames();
        for (int i = 1; i < all.Count; i++)
        {
            Assert.Equal(all[i - 1].Header.Sequence + 1, all[i].Header.Sequence);
        }
        List<FakeLinkHost.RecordedFrame> audio = host.FramesOf(LinkMessageType.InboundAudio);
        Assert.Equal(25, audio.Count);
        short[] decoded = new short[LinkProtocol.InboundFrameSamples];
        for (int i = 0; i < audio.Count; i++)
        {
            Assert.Equal(1u, audio[i].Header.CallId);
            Assert.Equal(i % 5 == 0, audio[i].AsFrame().Concealed);
            audio[i].AsFrame().ReadPcm(decoded);
            Assert.Equal((short)i, decoded[0]);
        }
        (uint CallId, uint TurnId, short[] Pcm)[] outbound = received.ToArray();
        Assert.Equal(10, outbound.Length);
        for (int i = 0; i < outbound.Length; i++)
        {
            Assert.Equal(1u, outbound[i].CallId);
            Assert.Equal(1u, outbound[i].TurnId);
            Assert.Equal(320, outbound[i].Pcm.Length);
            Assert.Equal((short)(100 + i), outbound[i].Pcm[0]);
        }
        Assert.Equal(0, link.AudioLaneDropped);
    }

    [Fact]
    public void Flush_DropsStaleTurnsAndAcksWithDiscardedMs()
    {
        using FakeLinkHost host = new();
        host.Start();
        using EngineLink link = new(Options(host));
        ConcurrentQueue<string> events = new();
        link.OutboundAudio = (_, turnId, _) => events.Enqueue($"audio:{turnId}");
        link.OutboundEnd = (_, turnId) => events.Enqueue($"end:{turnId}");
        link.Flush = (_, turnId) =>
        {
            events.Enqueue($"flush:{turnId}");
            return 123;
        };
        using ManualResetEventSlim connected = new(false);
        link.Connected = _ => connected.Set();
        link.Start();
        Assert.True(connected.Wait(WaitMs));
        link.SendCallStart(7, new CallStartMessage { Direction = LinkCallDirection.Outbound, SipCallId = "x" });
        short[] pcm = new short[160];
        host.SendOutboundAudio(7, 1, pcm);
        host.SendFlush(7, 1);
        host.SendOutboundAudio(7, 1, pcm);
        host.SendOutboundEnd(7, 1);
        host.SendOutboundAudio(7, 2, pcm);
        host.SendFlush(7, 1);
        Assert.True(host.WaitUntil(() => host.FramesOf(LinkMessageType.FlushAck).Count >= 2, WaitMs));
        Assert.Equal(["audio:1", "flush:1", "audio:2"], events.ToArray());
        Assert.Equal(2, link.StaleOutboundDropped);
        Assert.Equal(1u, link.FlushedTurn);
        List<FakeLinkHost.RecordedFrame> acks = host.FramesOf(LinkMessageType.FlushAck);
        Assert.Equal(new LinkFlushAck(1, 123), acks[0].AsFrame().ReadFlushAck());
        Assert.Equal(new LinkFlushAck(1, 0), acks[1].AsFrame().ReadFlushAck());
        Assert.Equal(7u, acks[0].Header.CallId);
    }

    [Fact]
    public void ToolRequest_IsAnsweredOnTheControlLane()
    {
        using FakeLinkHost host = new();
        host.Start();
        using EngineLink link = new(Options(host));
        link.ToolRequest = (callId, requestId, request) =>
            link.SendToolResult(callId, requestId, new ToolResultMessage { Status = LinkToolStatus.Ok, Message = request.Name });
        using ManualResetEventSlim connected = new(false);
        link.Connected = _ => connected.Set();
        link.Start();
        Assert.True(connected.Wait(WaitMs));
        link.SendCallStart(3, new CallStartMessage { Direction = LinkCallDirection.Inbound, SipCallId = "t" });
        host.SendToolRequest(3, 42, new ToolRequestMessage { Name = "hangup" });
        Assert.True(host.WaitUntil(() => host.FramesOf(LinkMessageType.ToolResult).Count == 1, WaitMs));
        ToolResultMessage result = host.FramesOf(LinkMessageType.ToolResult)[0].AsFrame().ReadToolResult(out uint requestId);
        Assert.Equal(42u, requestId);
        Assert.Equal(LinkToolStatus.Ok, result.Status);
        Assert.Equal("hangup", result.Message);
    }

    [Fact]
    public void Disconnect_ReconnectsWithBackoffAndResumesTheCall()
    {
        using FakeLinkHost host = new();
        host.Start();
        using EngineLink link = new(Options(host));
        int connects = 0;
        using ManualResetEventSlim second = new(false);
        using ManualResetEventSlim dropped = new(false);
        link.Connected = _ =>
        {
            if (Interlocked.Increment(ref connects) == 2)
            {
                link.SendCallStart(9, new CallStartMessage { Direction = LinkCallDirection.Inbound, SipCallId = "r", Resume = true });
                second.Set();
            }
        };
        link.Disconnected = _ => dropped.Set();
        link.Start();
        Assert.True(host.WaitForConnections(1, WaitMs));
        Assert.True(host.WaitUntil(() => link.IsConnected, WaitMs));
        long droppedAt = Environment.TickCount64;
        host.DropConnection();
        Assert.True(dropped.Wait(WaitMs));
        Assert.True(second.Wait(WaitMs));
        long reconnectMs = Environment.TickCount64 - droppedAt;
        _output.WriteLine($"reconnected after {reconnectMs} ms");
        Assert.True(host.WaitUntil(() => host.FramesOf(LinkMessageType.CallStart).Count == 1, WaitMs));
        CallStartMessage resume = host.FramesOf(LinkMessageType.CallStart)[0].AsFrame().ReadCallStart();
        Assert.True(resume.Resume);
        Assert.Equal(1, link.Reconnects);
        Assert.Equal(2, host.Connections);
        Assert.InRange(reconnectMs, 0, 2000);
    }

    [Fact]
    public void Liveness_ClosesASilentConnection()
    {
        using FakeLinkHost host = new() { AnswerPings = false };
        host.Start();
        using EngineLink link = new(Options(host) with { PingIntervalMs = 100, LivenessTimeoutMs = 300 });
        using ManualResetEventSlim dropped = new(false);
        string? reason = null;
        link.Disconnected = r =>
        {
            reason = r;
            dropped.Set();
        };
        link.Start();
        Assert.True(host.WaitForConnections(1, WaitMs));
        Assert.True(dropped.Wait(WaitMs));
        _output.WriteLine($"dropped: {reason}");
        Assert.True(host.FramesOf(LinkMessageType.Ping).Count >= 2);
        Assert.True(host.WaitForConnections(2, WaitMs));
    }

    [Fact]
    public async Task Outage_HangsUpAfterTheTimeoutWithGoodbye()
    {
        using FakeLinkHost host = new();
        host.Start();
        using EngineLink link = new(Options(host));
        using LinkOutageGuard guard = new(new LinkOutageGuardOptions { OutageHangupMs = 600, PromptRepeatMs = 200 }, () => link.IsConnected);
        ConcurrentQueue<PromptKind> prompts = new();
        using ManualResetEventSlim hungUp = new(false);
        int resumes = 0;
        guard.PlayPrompt = kind =>
        {
            prompts.Enqueue(kind);
            return 50;
        };
        guard.ResumeCall = () => Interlocked.Increment(ref resumes);
        guard.HangUp = () => hungUp.Set();
        link.Connected = _ => guard.LinkConnected();
        link.Disconnected = _ => guard.LinkDisconnected();
        link.Start();
        Assert.True(host.WaitUntil(() => link.IsConnected, WaitMs));
        guard.CallStarted();
        long start = Environment.TickCount64;
        host.Shutdown();
        Assert.True(hungUp.Wait(WaitMs));
        long elapsed = Environment.TickCount64 - start;
        await Task.Delay(50);
        _output.WriteLine($"hung up after {elapsed} ms; prompts={string.Join(",", prompts)}");
        Assert.InRange(elapsed, 600, 3000);
        PromptKind[] played = prompts.ToArray();
        Assert.True(played.Count(p => p == PromptKind.OneMoment) >= 2);
        Assert.Equal(PromptKind.Goodbye, played[^1]);
        Assert.Equal(0, resumes);
        Assert.Equal(1, guard.OutageHangups);
        Assert.False(guard.InOutage);
    }

    [Fact]
    public async Task Outage_ResumesWhenTheHostReturnsInTime()
    {
        using FakeLinkHost host = new();
        host.Start();
        using EngineLink link = new(Options(host));
        using LinkOutageGuard guard = new(new LinkOutageGuardOptions { OutageHangupMs = 3000, PromptRepeatMs = 200 }, () => link.IsConnected);
        ConcurrentQueue<PromptKind> prompts = new();
        using ManualResetEventSlim resumed = new(false);
        int hangups = 0;
        guard.PlayPrompt = kind =>
        {
            prompts.Enqueue(kind);
            return 50;
        };
        guard.ResumeCall = () => resumed.Set();
        guard.HangUp = () => Interlocked.Increment(ref hangups);
        link.Connected = _ => guard.LinkConnected();
        link.Disconnected = _ => guard.LinkDisconnected();
        link.Start();
        Assert.True(host.WaitUntil(() => link.IsConnected, WaitMs));
        guard.CallStarted();
        host.DropConnection();
        Assert.True(host.WaitUntil(() => guard.InOutage, WaitMs));
        Assert.True(resumed.Wait(WaitMs));
        await Task.Delay(300);
        Assert.False(guard.InOutage);
        Assert.Contains(PromptKind.OneMoment, prompts);
        Assert.DoesNotContain(PromptKind.Goodbye, prompts);
        Assert.Equal(0, hangups);
        Assert.Equal(1, guard.Outages);
        guard.CallEnded();
    }

    [Fact]
    public void OutageGuard_CallStartingInTheConnectGap_IsNotAnOutage()
    {
        // EngineLink reports IsConnected just before it raises Connected; a call activated in between used to start an
        // outage (a "one moment" prompt) and then "resume" with a second CallStart.
        bool linkUp = true;
        using LinkOutageGuard guard = new(new LinkOutageGuardOptions { OutageHangupMs = 1000, PromptRepeatMs = 200 }, () => linkUp);
        int prompts = 0;
        int resumes = 0;
        guard.PlayPrompt = _ =>
        {
            Interlocked.Increment(ref prompts);
            return 0;
        };
        guard.ResumeCall = () => Interlocked.Increment(ref resumes);
        guard.CallStarted();
        guard.LinkConnected();
        Assert.False(guard.InOutage);
        Assert.Equal(0, guard.Outages);
        Assert.Equal(0, prompts);
        Assert.Equal(0, resumes);
        linkUp = false;
        guard.LinkDisconnected();
        Assert.True(guard.InOutage);
        Assert.Equal(1, guard.Outages);
        guard.CallEnded();
        Assert.False(guard.InOutage);
    }

    [Fact]
    public void AudioLane_DropsTheOldestFrameWhenFull()
    {
        LinkSendQueue queue = new(audioDepth: 3, controlDepth: 4, controlTimeoutMs: 100);
        short[] frame = new short[LinkProtocol.InboundFrameSamples];
        for (int i = 1; i <= 5; i++)
        {
            Array.Fill(frame, (short)i);
            queue.EnqueueAudio(1, frame, concealed: false);
        }
        Assert.Equal(2, queue.AudioDropped);
        Assert.Equal(3, queue.AudioCount);
        queue.EnqueueControl(new LinkControlItem(LinkMessageType.CallEnd, 1, 0, 0, 0, null));
        short[] scratch = new short[LinkProtocol.InboundFrameSamples];
        Assert.True(queue.TryDequeue(10, out LinkControlItem control, scratch, out _, out _, out bool isAudio));
        Assert.False(isAudio);
        Assert.Equal(LinkMessageType.CallEnd, control.Type);
        Assert.True(queue.TryDequeue(10, out _, scratch, out uint callId, out _, out isAudio));
        Assert.True(isAudio);
        Assert.Equal(1u, callId);
        Assert.Equal(3, scratch[0]);
        Assert.True(queue.TryDequeue(10, out _, scratch, out _, out _, out _));
        Assert.Equal(4, scratch[0]);
        Assert.True(queue.TryDequeue(10, out _, scratch, out _, out _, out _));
        Assert.Equal(5, scratch[0]);
        Assert.False(queue.TryDequeue(10, out _, scratch, out _, out _, out _));
    }

    [Fact]
    public void ControlLane_TimesOutInsteadOfDroppingWhenWedged()
    {
        LinkSendQueue queue = new(audioDepth: 2, controlDepth: 2, controlTimeoutMs: 50);
        queue.EnqueueControl(new LinkControlItem(LinkMessageType.CallEnd, 1, 0, 0, 0, null));
        queue.EnqueueControl(new LinkControlItem(LinkMessageType.CallEnd, 1, 0, 0, 0, null));
        Assert.Throws<TimeoutException>(() => queue.EnqueueControl(new LinkControlItem(LinkMessageType.CallEnd, 1, 0, 0, 0, null)));
        Assert.Equal(2, queue.ControlCount);
    }

    private static EngineLinkOptions Options(FakeLinkHost host, string token = "") => new()
    {
        SocketPath = host.SocketPath,
        Token = token,
        ReconnectBaseMs = 10,
        ReconnectCapMs = 100,
    };
}
