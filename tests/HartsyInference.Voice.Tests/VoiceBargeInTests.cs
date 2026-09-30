using HartsyInference.Voice.Audio;
using HartsyInference.Voice.Tests.Fakes;
using HartsyInference.Voice.Turns;
using Xunit;

namespace HartsyInference.Voice.Tests;

/// <summary>Barge-in on the audio thread's own path (front-end, worker, turn signals, outbound queue), driven on the test
/// thread: speech over the reply past the hold-off stops it, cancels the turn and flushes its queued audio; the hold-off,
/// the minimum duration and the disabled switch hold; the interrupting speech is still answered, while speech that ends
/// over the reply without barging in is not.</summary>
public sealed class VoiceBargeInTests
{
    private const int Rate = VoiceAudioFrontend.SampleRate;
    private const int TurnId = 7;

    private static VoiceAgentOptions Options(bool enabled = true) =>
        new() { BargeInEnabled = enabled, BargeInHoldoffMs = 300, BargeInMinMs = 200, BargeInProbability = 0.6f };

    [Fact]
    public async Task SpeechOverTheReplyCancelsTheTurnAndResetsTheOutboundQueue()
    {
        using Rig rig = new(Options());
        CancellationTokenSource turn = rig.Speak();
        float[] reply = new float[2_000];
        Array.Fill(reply, 0.25f);
        long epoch = await rig.Outbound.WaitFlushesAppliedAsync(CancellationToken.None);
        Assert.True(await rig.Outbound.WriteAsync(reply, 0, reply.Length, epoch, CancellationToken.None));

        rig.Push(0.35, 0f);
        Assert.Empty(rig.Sink.BargeIns);
        rig.Push(0.3, VoiceHarness.SpeechLevel);

        (int bargedTurn, long detectNs) = Assert.Single(rig.Sink.BargeIns);
        Assert.Equal(TurnId, bargedTurn);
        Assert.True(detectNs > 0);
        Assert.True(await Cancelled(turn.Token), "the turn's token was not cancelled.");
        Assert.Equal(0, rig.Signals.SpeakingTurn);
        Assert.Equal(TurnId, rig.Signals.BargedInTurn);
        Assert.False(await rig.Outbound.WriteAsync(reply, 0, reply.Length, epoch, CancellationToken.None));

        float[] played = new float[4_096];
        Assert.Equal(0, rig.Outbound.Read(played));
        Assert.All(played, sample => Assert.Equal(0f, sample));
        Assert.Equal(reply.Length, rig.Outbound.DiscardedSamples);
        Assert.True(rig.Outbound.LastDiscardNs >= detectNs);
    }

    [Fact]
    public void TheHoldOffIgnoresSpeechAtTheStartOfTheReply()
    {
        using Rig rig = new(Options());
        rig.Speak();
        long speakingFrom = rig.Frontend.ClockSamples;
        // Speech from the first frame of the reply: only what follows the 300 ms hold-off counts toward the 200 ms run.
        rig.Push(0.28, VoiceHarness.SpeechLevel);
        Assert.Empty(rig.Sink.BargeIns);
        rig.Push(0.4, VoiceHarness.SpeechLevel);

        Assert.Single(rig.Sink.BargeIns);
        long decidedAt = rig.BargeInClock;
        Assert.InRange(decidedAt - speakingFrom, (300 + 200) * Rate / 1000, (300 + 200) * Rate / 1000 + 2 * 512);
    }

    [Fact]
    public void SpeechShorterThanTheMinimumDoesNotBargeIn()
    {
        using Rig rig = new(Options());
        rig.Speak();
        rig.Push(0.4, 0f);
        rig.Push(0.14, VoiceHarness.SpeechLevel);
        rig.Push(0.3, 0f);
        rig.Push(0.14, VoiceHarness.SpeechLevel);
        rig.Push(1.0, 0f);

        Assert.Empty(rig.Sink.BargeIns);
        Assert.Equal(TurnId, rig.Signals.SpeakingTurn);
        Assert.Equal(0, rig.Outbound.DiscardedSamples);
    }

    [Fact]
    public void DisabledBargeInIgnoresSpeechOverTheReply()
    {
        using Rig rig = new(Options(enabled: false));
        CancellationTokenSource turn = rig.Speak();
        rig.Push(0.4, 0f);
        rig.Push(1.0, VoiceHarness.SpeechLevel);
        rig.Push(1.0, 0f);

        Assert.Empty(rig.Sink.BargeIns);
        Assert.False(turn.IsCancellationRequested);
        Assert.Equal(TurnId, rig.Signals.SpeakingTurn);
        Assert.Empty(rig.Sink.Utterances);
        Assert.Single(rig.Sink.Discarded);
    }

    [Fact]
    public void TheInterruptingSpeechIsStillAnswered()
    {
        using Rig rig = new(Options());
        rig.Speak();
        rig.Push(0.4, 0f);
        rig.Push(1.0, VoiceHarness.SpeechLevel);
        rig.Push(1.0, 0f);

        Assert.Single(rig.Sink.BargeIns);
        VoiceTurnInput utterance = Assert.Single(rig.Sink.Utterances);
        // The whole second of speech, including the part before the barge-in was decided.
        Assert.InRange(utterance.Audio!.Length, Rate, Rate + 2 * 480 + 512);
        Assert.Empty(rig.Sink.Discarded);
    }

    [Fact]
    public void SpeechThatEndsOverTheReplyWithoutBargingInIsNotAnswered()
    {
        using Rig rig = new(Options());
        rig.Speak();
        // All of it inside the hold-off, so it never counts toward a barge-in, and long enough to be an utterance.
        rig.Push(0.28, VoiceHarness.SpeechLevel);
        rig.Push(1.0, 0f);

        Assert.Empty(rig.Sink.BargeIns);
        Assert.Empty(rig.Sink.Utterances);
        Assert.Single(rig.Sink.Discarded);
    }

    [Fact]
    public void SpeechInsideTheReplyIsNotAnsweredEvenWhenTheReplyEndsDuringTheHangover()
    {
        using Rig rig = new(Options());
        rig.Speak();
        // Inside the hold-off, so no barge-in; the reply then finishes 100 ms after the speech, long before the
        // 700 ms endpoint, which is decided with nothing audible any more.
        rig.Push(0.28, VoiceHarness.SpeechLevel);
        rig.Push(0.1, 0f);
        rig.StopSpeaking();
        rig.Push(1.0, 0f);

        Assert.Empty(rig.Sink.BargeIns);
        Assert.Empty(rig.Sink.Utterances);
        Assert.Single(rig.Sink.Discarded);
    }

    [Fact]
    public void SpeechThatBeganBeforeTheReplyIsAnswered()
    {
        using Rig rig = new(Options());
        // The caller was already talking when a prompt started over them, and stopped inside its hold-off.
        rig.Push(0.4, VoiceHarness.SpeechLevel);
        rig.Speak();
        rig.Push(0.2, VoiceHarness.SpeechLevel);
        rig.Push(1.0, 0f);

        Assert.Empty(rig.Sink.BargeIns);
        Assert.Single(rig.Sink.Utterances);
        Assert.Empty(rig.Sink.Discarded);
    }

    [Fact]
    public void SpeechThatOutlastsTheReplyIsAnswered()
    {
        using Rig rig = new(Options(enabled: false));
        rig.Speak();
        rig.Push(0.3, 0f);
        rig.Push(0.3, VoiceHarness.SpeechLevel);
        rig.StopSpeaking();
        // Well past the reply and its echo tail before the caller stops.
        rig.Push(0.8, VoiceHarness.SpeechLevel);
        rig.Push(1.0, 0f);

        Assert.Single(rig.Sink.Utterances);
        Assert.Empty(rig.Sink.Discarded);
    }

    private static async Task<bool> Cancelled(CancellationToken token)
    {
        try
        {
            await Task.Delay(Timeout.Infinite, token).WaitAsync(TimeSpan.FromSeconds(5));
        }
        catch (OperationCanceledException)
        {
            return true;
        }
        catch (TimeoutException)
        {
            return false;
        }
        return false;
    }

    /// <summary>Front-end, worker and outbound queue on shared turn signals, with the reply of <see cref="TurnId"/> audible.</summary>
    private sealed class Rig : IDisposable
    {
        private readonly FrontendDriver _driver;
        private readonly VoiceAudioWorker _worker;

        public Rig(VoiceAgentOptions options)
        {
            Signals = new VoiceTurnSignals();
            _driver = new FrontendDriver(options, Signals);
            Sink = new VoiceEndpointingTests.RecordingSink();
            _worker = new VoiceAudioWorker(_driver.Frontend, Signals, Sink, inboundCapacity: 1 << 17, backlogLimitSamples: 1 << 17);
            Outbound = new VoiceOutbound(1 << 14, Signals);
            // Settle the VAD on silence before the reply starts, as a live call would be.
            Push(0.2, 0f);
        }

        public VoiceTurnSignals Signals { get; }

        public VoiceEndpointingTests.RecordingSink Sink { get; }

        public VoiceOutbound Outbound { get; }

        public VoiceAudioFrontend Frontend => _driver.Frontend;

        /// <summary>The front-end clock when the (single) barge-in was decided.</summary>
        public long BargeInClock { get; private set; }

        /// <summary>Publishes turn <see cref="TurnId"/> and marks its reply audible.</summary>
        public CancellationTokenSource Speak()
        {
            CancellationTokenSource cancellation = new();
            Signals.BeginTurn(new VoiceTurnHandle(TurnId, cancellation));
            Signals.BeginSpeaking(TurnId);
            return cancellation;
        }

        /// <summary>The reply finished playing.</summary>
        public void StopSpeaking() => Signals.EndSpeaking(TurnId);

        public void Push(double seconds, float level)
        {
            float[] frame = new float[VoiceAudioFrontend.FrameSamples];
            Array.Fill(frame, level);
            int frames = (int)Math.Round(seconds * Rate / frame.Length);
            for (int i = 0; i < frames; i++)
            {
                int before = Sink.BargeIns.Count;
                _worker.Push(frame);
                _worker.ProcessAvailable();
                if (Sink.BargeIns.Count > before)
                {
                    BargeInClock = Frontend.ClockSamples;
                }
            }
        }

        public void Dispose()
        {
            _worker.Dispose();
            _driver.Dispose();
        }
    }
}
