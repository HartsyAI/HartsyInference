using HartsyInference.Core.Numerics;
using HartsyInference.Voice.Audio;
using HartsyInference.Voice.Tests.Fakes;
using HartsyInference.Voice.Turns;
using Xunit;

namespace HartsyInference.Voice.Tests;

/// <summary>Endpointing through the real <see cref="HartsyInference.Audio.Models.Wake.SileroVadStream"/> with a VAD scripted
/// by the audio level: the turn ends only after the configured silence, a long utterance is cut at the maximum, the
/// decision is counted in samples (the same whether audio arrives in real time or in one burst), and the captured
/// utterance is the caller's audio, sample for sample, across the capture ring's wrap.</summary>
public sealed class VoiceEndpointingTests
{
    private const int Rate = FrontendDriver.Rate;
    private const int Window = 512;
    private const int PadSamples = VoiceAudioFrontend.SpeechPadMs * Rate / 1000;

    [Fact]
    public void TheTurnEndsAfterTheConfiguredSilenceAndNotBefore()
    {
        using FrontendDriver driver = new();
        driver.Feed(0.5, 0f);
        driver.Feed(1.0, VoiceHarness.SpeechLevel);
        long speechEnd = driver.Fed;
        driver.Feed(0.68, 0f);
        Assert.Empty(driver.With(VoiceFrameEvents.Endpoint));
        driver.Feed(0.5, 0f);

        FrontendDriver.Decision endpoint = Assert.Single(driver.With(VoiceFrameEvents.Endpoint));
        long silenceAtDecision = endpoint.Clock - speechEnd;
        Assert.InRange(silenceAtDecision, 700 * Rate / 1000, 700 * Rate / 1000 + 2 * Window);
        Assert.InRange(endpoint.HangoverSamples, 700 * Rate / 1000, 700 * Rate / 1000 + 2 * Window);
        Assert.InRange(endpoint.UtteranceSamples, Rate + 2 * PadSamples - Window, Rate + 2 * PadSamples + Window);
    }

    [Fact]
    public void APauseShorterThanTheSilenceKeepsOneTurn()
    {
        using FrontendDriver driver = new();
        driver.Feed(0.3, 0f);
        driver.Feed(1.0, VoiceHarness.SpeechLevel);
        driver.Feed(0.6, 0f);
        driver.Feed(1.0, VoiceHarness.SpeechLevel);
        driver.Feed(1.0, 0f);

        FrontendDriver.Decision endpoint = Assert.Single(driver.With(VoiceFrameEvents.Endpoint));
        Assert.InRange(endpoint.UtteranceSamples, (int)(2.6 * Rate), (int)(2.6 * Rate) + 2 * PadSamples + Window);
    }

    [Fact]
    public void ContinuousSpeechIsCutAtTheMaximumUtterance()
    {
        using FrontendDriver driver = new(new VoiceAgentOptions { MaxUtteranceMs = 2_000 });
        driver.Feed(0.2, 0f);
        driver.Feed(5.0, VoiceHarness.SpeechLevel);
        List<FrontendDriver.Decision> cuts = [.. driver.With(VoiceFrameEvents.Endpoint)];
        Assert.Equal(2, cuts.Count);
        foreach (FrontendDriver.Decision cut in cuts)
        {
            Assert.Equal(0, cut.HangoverSamples);
            Assert.InRange(cut.UtteranceSamples, 2 * Rate, 2 * Rate + PadSamples + Window);
        }

        driver.Feed(1.0, 0f);
        Assert.Equal(3, driver.With(VoiceFrameEvents.Endpoint).Count());
    }

    [Fact]
    public void DecisionsFollowTheSampleClockNotTheArrivalRate()
    {
        (double Seconds, float Level)[] script = [(0.4, 0f), (0.8, 0.9f), (0.3, 0f), (0.5, 0.9f), (1.2, 0f), (0.6, 0.9f), (1.0, 0f)];

        using FrontendDriver paced = new();
        foreach ((double seconds, float level) in script)
        {
            paced.Feed(seconds, level);
        }

        RecordingSink sink = new();
        VoiceTurnSignals signals = new();
        using FrontendDriver burstFrontend = new(signals: signals);
        using VoiceAudioWorker worker = new(burstFrontend.Frontend, signals, sink, inboundCapacity: 1 << 17, backlogLimitSamples: 1 << 17);
        foreach ((double seconds, float level) in script)
        {
            float[] block = new float[(int)Math.Round(seconds * Rate)];
            Array.Fill(block, level);
            worker.Push(block);
        }
        worker.ProcessAvailable();

        List<FrontendDriver.Decision> expected = [.. paced.With(VoiceFrameEvents.Endpoint)];
        Assert.Equal(2, expected.Count);
        Assert.Equal(expected.Select(d => d.UtteranceSamples), sink.Utterances.Select(u => u.Audio!.Length));
        Assert.Equal(expected.Select(d => d.HangoverSamples), sink.Utterances.Select(u => u.HangoverSamples));
    }

    [Fact]
    public void TheCapturedUtteranceIsTheCallersAudioAcrossTheRingWrap()
    {
        // A 1 s maximum makes the capture ring 2.1 s, so three seconds of lead-in wrap it before the speech.
        using FrontendDriver driver = new(new VoiceAgentOptions { MaxUtteranceMs = 1_000, EndOfTurnSilenceMs = 100 });
        long speechStart = 3 * Rate;
        long speechEnd = speechStart + (long)(0.6 * Rate);
        float Level(long i) => i >= speechStart && i < speechEnd ? 0.6f + 0.3f * MathF.Sin(i / 37f) : 0.001f * MathF.Sin(i / 11f);
        driver.Feed((int)speechEnd + Rate / 2, Level);

        FrontendDriver.Decision endpoint = Assert.Single(driver.With(VoiceFrameEvents.Endpoint));
        float[] captured = new float[endpoint.UtteranceSamples];
        driver.Frontend.CopyUtterance(captured);
        long start = endpoint.UtteranceStart;
        Assert.Equal(endpoint.Clock - endpoint.HangoverSamples + PadSamples, start + endpoint.UtteranceSamples);
        for (int i = 0; i < captured.Length; i++)
        {
            Assert.Equal(Level(start + i), captured[i]);
        }
        Assert.InRange(start, speechStart - PadSamples - Window, speechStart - PadSamples + Window);
    }

    /// <summary>Records what the audio thread hands off.</summary>
    internal sealed class RecordingSink : IVoiceAudioSink
    {
        public List<VoiceTurnInput> Utterances { get; } = [];

        public List<int> Discarded { get; } = [];

        public List<(int Turn, long Ns)> BargeIns { get; } = [];

        public List<Exception> Faults { get; } = [];

        public void OnUtterance(VoiceTurnInput utterance) => Utterances.Add(utterance);

        public void OnUtteranceDiscarded(int samples) => Discarded.Add(samples);

        public void OnBargeIn(int turnId, long detectNs) => BargeIns.Add((turnId, detectNs));

        public void OnAudioFault(Exception error) => Faults.Add(error);
    }
}
