using HartsyInference.Core.Runtime;
using HartsyInference.PhoneGateway.Sip;
using HartsyInference.PhoneLink;
using HartsyInference.Voice;
using HartsyInference.VoiceHost.Calls;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.VoiceHost.Tests.Loopback;

/// <summary>Whole calls across every seam: a sipsorcery softphone speaking real speech (the JFK clip) to the real gateway,
/// over PhoneLink to the real host running Kokoro and Whisper small.en on the RTX 3060, with a scripted language model.
/// Asserted here and nowhere else: audio and transcripts cross both ways; a barge-in stops the reply on the wire within
/// 100 ms of the VAD decision, measured to the last frame the gateway's RTP tick actually sent; no frame of the cancelled
/// turn reaches the gateway after its <c>Flush</c>; both hang-ups end the call on both sides; and the agent's BYE reaches
/// the phone only after its goodbye has played out, with silence on the line in between. Run alone, after the
/// quiet window:
/// <c>CUDA_VISIBLE_DEVICES=1 HARTSY_REQUIRE_REAL_WEIGHTS=1 HARTSYINFERENCE_MODELS_DIR=/mnt/model-storage/Models dotnet test tests/HartsyInference.VoiceHost.Tests -c Release --filter "FullyQualifiedName~LoopbackSipCallWithHostTests"</c>.</summary>
[Trait("Category", "Slow")]
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class LoopbackSipCallWithHostTests : IClassFixture<LoopbackHostFixture>
{
    private const int Audible = LoopbackPhone.AudiblePeak;
    private const double BargeInGateMs = 100;
    private const string LongReply = "Let me tell you about our opening hours. We open at nine in the morning on weekdays. "
        + "On Saturdays we open a little later, at ten. Sundays we are closed all day. Holidays follow the Sunday hours. "
        + "You can also reach us online at any time of the day or night.";

    private readonly LoopbackHostFixture _host;
    private readonly ITestOutputHelper _output;

    public LoopbackSipCallWithHostTests(LoopbackHostFixture host, ITestOutputHelper output)
    {
        _host = host;
        _output = output;
        foreach (string line in host.Log)
        {
            output.WriteLine(line);
        }
    }

    /// <summary>"And so, my fellow Americans": the question the caller asks.</summary>
    private static byte[] Question => Slice(0.0, 2.6);

    /// <summary>"Ask not what your country can do for you": what the caller says over the reply.</summary>
    private static byte[] Interruption => Slice(5.3, 8.4);

    [Fact]
    public async Task AudioAndTranscriptsCrossBothWaysAndTheCallerHangsUp()
    {
        if (Skipped())
        {
            return;
        }
        LoopbackGateway gateway = _host.Gateway!;
        _host.Text.Reply("Thank you for calling. Your order has shipped and will arrive on Friday.");
        using LoopbackPhone phone = new();
        try
        {
            long start = MonotonicClock.NowNs();
            Assert.True(await phone.CallAsync(gateway.Port), $"call failed: {phone.LastFailure}");
            Assert.True(gateway.WaitUntil(() => phone.AudibleSince(start, Audible) >= 25), "the greeting never reached the phone.");
            WaitForQuiet(phone);

            long asked = MonotonicClock.NowNs();
            await phone.SpeakAsync(Question);
            LinkEventMessage? heard = _host.WaitForEvent(asked, e => e.Kind == LinkEventKind.TranscriptFinal, 30_000);
            Assert.NotNull(heard);
            _output.WriteLine($"host heard: \"{heard.Text}\" (turn {heard.TurnId})");
            Assert.True(heard.Text!.Contains("fellow", StringComparison.OrdinalIgnoreCase) || heard.Text.Contains("americans", StringComparison.OrdinalIgnoreCase),
                $"the host heard \"{heard.Text}\" for \"And so, my fellow Americans\".");
            Assert.True(gateway.WaitUntil(() => phone.AudibleSince(asked, Audible) >= 50), "the reply never reached the phone.");
            VoiceCall call = _host.Call!;
            Assert.True(call.InboundFrames > 100, $"the host took only {call.InboundFrames} caller frames.");
            Assert.True(call.OutboundSamples > 16_000, $"the host sent only {call.OutboundSamples} reply samples.");
            LogLatency(gateway, asked);

            phone.Agent.Hangup();
            Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Idle && _host.Call is null, 10_000),
                "the caller's hang-up did not end the call on both sides.");
        }
        finally
        {
            _host.EndCall(phone);
        }
    }

    [Fact]
    public async Task ABargeInStopsTheReplyOnTheWireWithin100MsAndNothingStaleFollows()
    {
        if (Skipped())
        {
            return;
        }
        LoopbackGateway gateway = _host.Gateway!;
        _host.Text.Reply(LongReply);
        using LoopbackPhone phone = new();
        try
        {
            long start = MonotonicClock.NowNs();
            Assert.True(await phone.CallAsync(gateway.Port), $"call failed: {phone.LastFailure}");
            Assert.True(gateway.WaitUntil(() => phone.AudibleSince(start, Audible) >= 25), "the greeting never reached the phone.");
            WaitForQuiet(phone);
            VoiceCall call = _host.Call!;
            (long Ns, int Turn) bargeIn = default;
            call.Session!.EventRaised += e =>
            {
                if (e.Kind == VoiceAgentEventKind.BargeIn)
                {
                    bargeIn = (e.TimestampNs, e.TurnId);
                }
            };

            long staleBefore = gateway.Link.StaleOutboundDropped;
            long asked = MonotonicClock.NowNs();
            await phone.SpeakAsync(Question);
            Assert.NotNull(_host.WaitForEvent(asked, e => e.Kind == LinkEventKind.TranscriptFinal, 30_000));
            Assert.True(gateway.WaitUntil(() => phone.AudibleSince(asked, Audible) >= 75), "the long reply never reached the phone.");
            _ = phone.SpeakAsync(Interruption);
            Assert.True(gateway.WaitUntil(() => bargeIn.Ns != 0, 10_000), "the caller spoke over the reply and no barge-in followed.");
            await Task.Delay(1_500);

            (long Ns, int Peak)[] sentAfter = gateway.Ticks.Audible(bargeIn.Ns, bargeIn.Ns + 1_500_000_000L, Audible);
            double lastSentMs = sentAfter.Length == 0 ? 0 : (sentAfter[^1].Ns - bargeIn.Ns) / 1e6;
            (long Ns, uint Turn) flush = Assert.Single(gateway.Flushes, f => f.Ns >= asked);
            double flushMs = (flush.Ns - bargeIn.Ns) / 1e6;
            _output.WriteLine($"barge-in on turn {bargeIn.Turn}: Flush({flush.Turn}) reached the gateway after {flushMs:F1} ms; "
                + $"{sentAfter.Length} audible frames sent after the decision, the last at {lastSentMs:F1} ms; "
                + $"gateway stale drops {gateway.Link.StaleOutboundDropped - staleBefore}, host stale samples {call.StaleSamples}");

            Assert.Equal((uint)bargeIn.Turn, flush.Turn);
            Assert.True(lastSentMs <= BargeInGateMs, $"the cancelled reply was still on the wire {lastSentMs:F1} ms after the barge-in.");
            // The gateway counts every frame at or below a flushed turn that arrives after the Flush: the host sent none.
            Assert.Equal(staleBefore, gateway.Link.StaleOutboundDropped);
        }
        finally
        {
            _host.EndCall(phone);
        }
    }

    [Fact]
    public async Task TheAgentsHangupEndsTheCallAtThePhoneAfterItsGoodbye()
    {
        if (Skipped())
        {
            return;
        }
        LoopbackGateway gateway = _host.Gateway!;
        _host.Text.ReplyThenCall("Goodbye, and have a wonderful day.", "hangup").Reply("");
        using LoopbackPhone phone = new();
        try
        {
            long start = MonotonicClock.NowNs();
            Assert.True(await phone.CallAsync(gateway.Port), $"call failed: {phone.LastFailure}");
            Assert.True(gateway.WaitUntil(() => phone.AudibleSince(start, Audible) >= 25), "the greeting never reached the phone.");
            WaitForQuiet(phone);

            long asked = MonotonicClock.NowNs();
            await phone.SpeakAsync(Question);
            Assert.True(phone.HungUp.Wait(45_000), "the agent's hangup never reached the phone.");
            (long Ns, int Peak)[] heard = phone.Frames().Where(f => f.Ns >= asked && f.Ns <= phone.HungUpNs).ToArray();
            int goodbyeFrames = heard.Count(f => f.Peak >= Audible);
            _output.WriteLine($"{goodbyeFrames} audible frames of goodbye before the BYE");
            Assert.True(goodbyeFrames >= 50, "the call ended before the goodbye played.");
            // The BYE waited for the goodbye to drain: its last audible frame arrived, then the gateway's tick was
            // sending silence again before the BYE.
            int lastAudible = Array.FindLastIndex(heard, f => f.Peak >= Audible);
            int quietTail = heard.Length - 1 - lastAudible;
            double tailMs = (phone.HungUpNs - heard[lastAudible].Ns) / 1e6;
            _output.WriteLine($"the BYE came {tailMs:F0} ms after the goodbye's last audible frame, {quietTail} quiet frames later");
            Assert.True(quietTail >= 2, $"only {quietTail} quiet frames between the goodbye's last audible frame and the BYE: the goodbye was cut.");
            Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Idle && _host.Call is null, 10_000));
        }
        finally
        {
            _host.EndCall(phone);
        }
    }

    private bool Skipped()
    {
        if (_host.SkipReason is null)
        {
            return false;
        }
        _output.WriteLine(_host.SkipReason);
        return true;
    }

    /// <summary>Waits until the phone has heard 400 ms with no audible packet: the greeting is over.</summary>
    private static void WaitForQuiet(LoopbackPhone phone)
    {
        long deadline = Environment.TickCount64 + 30_000;
        while (Environment.TickCount64 < deadline)
        {
            long now = MonotonicClock.NowNs();
            if (phone.AudibleSince(now - 400_000_000L, Audible) == 0)
            {
                return;
            }
            Thread.Sleep(50);
        }
        throw new TimeoutException("The phone never went quiet after the greeting.");
    }

    private void LogLatency(LoopbackGateway gateway, long sinceNs)
    {
        foreach ((long _, LinkEventMessage item) in gateway.Events.Where(e => e.Ns >= sinceNs && e.Event.Kind == LinkEventKind.TurnLatency))
        {
            TurnLatency l = item.Latency!;
            _output.WriteLine($"turn {item.TurnId}: stt {l.SttMs} ms, llm first token {l.LlmFirstTokenMs} ms, tts first chunk {l.TtsFirstChunkMs} ms, "
                + $"transport {l.TransportMs} ms, total {l.TotalMs} ms");
        }
    }

    private static byte[] Slice(double fromSeconds, double toSeconds)
    {
        byte[] all = LoopbackAssets.JfkPhonePcm();
        int from = (int)(fromSeconds * LoopbackAssets.PhoneRate) * 2;
        int to = Math.Min(all.Length, (int)(toSeconds * LoopbackAssets.PhoneRate) * 2);
        return all[from..to];
    }
}
