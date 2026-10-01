using System.Collections.Concurrent;
using HartsyInference.PhoneGateway.Media;
using HartsyInference.PhoneGateway.Metrics;
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

    /// <summary>For waits on a reconnect, which starts threads and so can stall on a box running other test lanes; the
    /// waits are on events, so a healthy run never comes near it.</summary>
    private const int UnderLoadWaitMs = 30_000;

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

    /// <summary>The outage is held open by the test rather than raced. Left alone, it lasts only for the link's first
    /// redial (a 0-10 ms backoff plus a handshake), which a 5 ms poll of <see cref="LinkOutageGuard.InOutage"/> can miss
    /// entirely on a loaded box. Here the host stops reading before the drop, so the redial connects but its handshake
    /// waits until the test opens the gate; the guard's own hooks mark the transitions; and the hang-up timeout is far
    /// beyond any scheduling delay, so only a host that never returns could end the call.</summary>
    [Fact]
    public void Outage_ResumesWhenTheHostReturnsInTime()
    {
        using FakeLinkHost host = new();
        host.Start();
        using EngineLink link = new(Options(host));
        using LinkOutageGuard guard = new(new LinkOutageGuardOptions { OutageHangupMs = 120_000, PromptRepeatMs = 200 }, () => link.IsConnected);
        ConcurrentQueue<PromptKind> prompts = new();
        using ManualResetEventSlim outageStarted = new(false);
        using ManualResetEventSlim resumed = new(false);
        int hangups = 0;
        guard.PlayPrompt = kind =>
        {
            prompts.Enqueue(kind);
            return 50;
        };
        guard.OutageStarted = outageStarted.Set;
        guard.ResumeCall = resumed.Set;
        guard.HangUp = () => Interlocked.Increment(ref hangups);
        link.Connected = _ => guard.LinkConnected();
        link.Disconnected = _ => guard.LinkDisconnected();
        link.Start();
        Assert.True(host.WaitUntil(() => link.IsConnected, UnderLoadWaitMs));
        guard.CallStarted();
        Assert.False(guard.InOutage);

        host.ReadGate.Reset();
        host.DropConnection();

        Assert.True(outageStarted.Wait(UnderLoadWaitMs), "dropping the host never started an outage");
        // The first "one moment" plays as the outage starts, before the hook fires.
        Assert.True(guard.InOutage);
        Assert.Contains(PromptKind.OneMoment, prompts);
        Assert.False(resumed.IsSet);

        host.ReadGate.Set();

        Assert.True(resumed.Wait(UnderLoadWaitMs), "the call was never resumed after the host came back");
        // LinkConnected ends the wait before it resumes the call, so these hold the moment resumed is set.
        Assert.False(guard.InOutage);
        Assert.DoesNotContain(PromptKind.Goodbye, prompts);
        Assert.Equal(0, Volatile.Read(ref hangups));
        Assert.Equal(1, guard.Outages);
        Assert.Equal(0, guard.OutageHangups);
        guard.CallEnded();
    }

    [Fact]
    public void Liveness_ClosesALinkWhoseWriterIsBlockedOnAHostThatStoppedReading()
    {
        using FakeLinkHost host = new();
        host.Start();
        using EngineLink link = new(Options(host) with { PingIntervalMs = 100, LivenessTimeoutMs = 400 });
        using ManualResetEventSlim dropped = new(false);
        string? reason = null;
        link.Disconnected = r =>
        {
            reason = r;
            dropped.Set();
        };
        link.Start();
        Assert.True(host.WaitUntil(() => link.IsConnected, WaitMs));
        host.ReadGate.Reset();

        // Fill the socket fast so the writer blocks inside a write well before liveness expires; a check that lived
        // in the writer loop would then never run again.
        short[] frame = new short[LinkProtocol.InboundFrameSamples];
        long lastSent = -1;
        long unchangedSince = Environment.TickCount64;
        long deadline = Environment.TickCount64 + WaitMs;
        while (!dropped.IsSet && Environment.TickCount64 < deadline)
        {
            link.TryEnqueueInboundAudio(1, frame, concealed: false);
            long sent = link.FramesSent;
            if (sent != lastSent)
            {
                lastSent = sent;
                unchangedSince = Environment.TickCount64;
            }
            else if (Environment.TickCount64 - unchangedSince > 50)
            {
                break;
            }
        }
        _output.WriteLine($"writer stopped after {link.FramesSent} frames");

        Assert.True(dropped.Wait(WaitMs), "a link whose writer is blocked was never closed");
        _output.WriteLine($"dropped: {reason}");
        host.ReadGate.Set();
    }

    [Fact]
    public void OutageGuard_AThrowingPromptStillEndsInAHangUp()
    {
        bool linkUp = true;
        using LinkOutageGuard guard = new(new LinkOutageGuardOptions { OutageHangupMs = 300, PromptRepeatMs = 100 }, () => linkUp);
        guard.PlayPrompt = _ => throw new IOException("injected prompt failure");
        using ManualResetEventSlim hungUp = new(false);
        guard.HangUp = () => hungUp.Set();
        guard.CallStarted();
        linkUp = false;
        guard.LinkDisconnected();

        Assert.True(hungUp.Wait(WaitMs), "the outage never ended in a hang-up");
        Assert.False(guard.InOutage);
        Assert.Equal(1, guard.OutageHangups);
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
        Assert.True(queue.TryEnqueueControl(new LinkControlItem(LinkMessageType.CallEnd, 1, 0, 0, 0, null)));
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
    public void ControlLane_ReportsAWedgedLaneInsteadOfThrowingOrDropping()
    {
        LinkSendQueue queue = new(audioDepth: 2, controlDepth: 2, controlTimeoutMs: 50);
        Assert.True(queue.TryEnqueueControl(new LinkControlItem(LinkMessageType.CallEnd, 1, 0, 0, 0, null)));
        Assert.True(queue.TryEnqueueControl(new LinkControlItem(LinkMessageType.CallEnd, 1, 0, 0, 0, null)));
        Assert.False(queue.TryEnqueueControl(new LinkControlItem(LinkMessageType.CallEnd, 1, 0, 0, 0, null)));
        Assert.Equal(2, queue.ControlCount);
    }

    [Fact]
    public void ControlLane_AZeroTimeoutNeverWaits()
    {
        LinkSendQueue queue = new(audioDepth: 2, controlDepth: 1, controlTimeoutMs: 30_000);
        LinkControlItem item = new(LinkMessageType.CallEnd, 1, 0, 0, 0, null);
        Assert.True(queue.TryEnqueueControl(item, 0));
        long start = Environment.TickCount64;
        Assert.False(queue.TryEnqueueControl(item, 0));
        Assert.InRange(Environment.TickCount64 - start, 0, 1000);
    }

    [Fact]
    public void ControlLane_AWedgedLaneDropsTheFrameCountsItAndReconnects()
    {
        using FakeLinkHost host = new();
        host.Start();
        // Liveness far beyond the test, so only the control lane can be what restarts the connection.
        using EngineLink link = new(Options(host) with { LivenessTimeoutMs = 60_000, ControlLaneDepth = 2, ControlEnqueueTimeoutMs = 100 });
        using ManualResetEventSlim dropped = new(false);
        link.Disconnected = _ => dropped.Set();
        link.Start();
        Assert.True(host.WaitUntil(() => link.IsConnected, WaitMs));
        host.ReadGate.Reset();
        Assert.True(BlockTheWriter(link), "the writer never blocked on the host that stopped reading");

        link.SendDtmf(1, new LinkDtmf('1', 100));
        link.SendDtmf(1, new LinkDtmf('2', 100));
        Assert.Equal(0, link.ControlLaneDropped);
        Assert.False(dropped.IsSet);
        link.SendDtmf(1, new LinkDtmf('3', 100));
        Assert.Equal(1, link.ControlLaneDropped);
        Assert.True(dropped.Wait(WaitMs), "a wedged control lane never restarted the connection");
        GatewayMetrics metrics = new()
        {
            LinkProbe = () => new LinkSnapshot(link.IsConnected, link.OutboundRate, link.Reconnects, 0, link.AudioLaneDropped,
                link.ControlLaneDropped, link.InboundDroppedWhileDown, link.StaleOutboundDropped, link.FramesSent, link.FramesReceived),
        };
        Assert.Contains("hartsy_phone_link_control_lane_dropped_total 1\n", PrometheusTextWriter.Render(metrics), StringComparison.Ordinal);

        host.ReadGate.Set();
        Assert.True(host.WaitForConnections(2, WaitMs));
        Assert.True(host.WaitUntil(() => link.IsConnected, WaitMs));
        Assert.Equal(1, link.Reconnects);
    }

    [Fact]
    public void ControlLane_TheReaderThreadNeverWaitsForRoom()
    {
        using FakeLinkHost host = new();
        host.Start();
        // A reader that waited for room would sit out the whole minute; liveness is longer still, so the watchdog
        // cannot be what ends the connection either.
        using EngineLink link = new(Options(host) with { LivenessTimeoutMs = 120_000, ControlLaneDepth = 2, ControlEnqueueTimeoutMs = 60_000 });
        using ManualResetEventSlim dropped = new(false);
        link.Disconnected = _ => dropped.Set();
        link.Start();
        Assert.True(host.WaitUntil(() => link.IsConnected, WaitMs));
        host.ReadGate.Reset();
        Assert.True(BlockTheWriter(link), "the writer never blocked on the host that stopped reading");
        link.SendDtmf(1, new LinkDtmf('1', 100));
        link.SendDtmf(1, new LinkDtmf('2', 100));

        long pingAt = Environment.TickCount64;
        host.Send(w => w.WritePingAsync(42, CancellationToken.None));
        Assert.True(dropped.Wait(WaitMs), "the reader waited for room instead of dropping its Pong");
        _output.WriteLine($"the reader dropped its Pong; link restarted after {Environment.TickCount64 - pingAt} ms");
        Assert.Equal(1, link.ControlLaneDropped);

        host.ReadGate.Set();
        Assert.True(host.WaitForConnections(2, WaitMs));
    }

    [Fact]
    public void Connected_ACallbackThatThrowsLeavesTheLinkUp()
    {
        using FakeLinkHost host = new();
        host.Start();
        using EngineLink link = new(Options(host));
        int connectedCalls = 0;
        int disconnects = 0;
        link.Connected = _ =>
        {
            Interlocked.Increment(ref connectedCalls);
            throw new InvalidOperationException("injected Connected failure");
        };
        link.Disconnected = _ => Interlocked.Increment(ref disconnects);
        link.ToolRequest = (callId, requestId, request) =>
            link.SendToolResult(callId, requestId, new ToolResultMessage { Status = LinkToolStatus.Ok, Message = request.Name });
        link.Start();
        Assert.True(host.WaitUntil(() => Volatile.Read(ref connectedCalls) == 1, WaitMs));

        // The connection whose Connected callback threw still carries frames both ways.
        host.SendToolRequest(3, 7, new ToolRequestMessage { Name = "still-up" });
        Assert.True(host.WaitUntil(() => host.FramesOf(LinkMessageType.ToolResult).Count == 1, WaitMs), "the link stopped working after Connected threw");
        Assert.Equal("still-up", host.FramesOf(LinkMessageType.ToolResult)[0].AsFrame().ReadToolResult(out _).Message);
        Assert.True(link.IsConnected);
        Assert.Equal(1, host.Connections);
        Assert.Equal(0, link.Reconnects);
        Assert.Equal(0, Volatile.Read(ref disconnects));
    }

    /// <summary>Floods the audio lane until the writer has finished no frame for 500 ms while frames keep coming: with
    /// the host no longer reading, the socket is full and the writer is parked inside a write.</summary>
    private static bool BlockTheWriter(EngineLink link)
    {
        short[] frame = new short[LinkProtocol.InboundFrameSamples];
        long lastSent = -1;
        long unchangedSince = Environment.TickCount64;
        long deadline = Environment.TickCount64 + WaitMs;
        while (Environment.TickCount64 < deadline)
        {
            link.TryEnqueueInboundAudio(1, frame, concealed: false);
            long sent = link.FramesSent;
            if (sent != lastSent)
            {
                lastSent = sent;
                unchangedSince = Environment.TickCount64;
            }
            else if (Environment.TickCount64 - unchangedSince > 500)
            {
                return true;
            }
        }
        return false;
    }

    private static EngineLinkOptions Options(FakeLinkHost host, string token = "") => new()
    {
        SocketPath = host.SocketPath,
        Token = token,
        ReconnectBaseMs = 10,
        ReconnectCapMs = 100,
    };
}
