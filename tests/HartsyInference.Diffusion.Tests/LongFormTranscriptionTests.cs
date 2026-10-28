using HartsyInference.Audio.Dsp.Windowing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Long-form transcription with a fake runner: window order, silence skipping, timeline offsets, fallbacks and cancellation.</summary>
public sealed class LongFormTranscriptionTests
{
    private const int SampleRate = 16_000;

    [Fact]
    public void Plan_WithoutAWindow_IsOneWindow()
    {
        IReadOnlyList<SpeechWindow> windows = LongFormTranscription.Plan(new float[SampleRate * 90], SampleRate, maxInputSeconds: null);

        SpeechWindow only = Assert.Single(windows);
        Assert.Equal(SampleRate * 90, only.Length);
        Assert.False(only.Silent);
    }

    [Fact]
    public void Plan_LongClipUnderAWindowLimit_IsSplit()
    {
        IReadOnlyList<SpeechWindow> windows = LongFormTranscription.Plan(new float[SampleRate * 65], SampleRate, maxInputSeconds: 30);

        Assert.True(windows.Count > 1);
    }

    [Fact]
    public void Plan_ClipThatFitsAWindowBelowThePlannerFloor_IsOneWindow()
    {
        IReadOnlyList<SpeechWindow> windows = LongFormTranscription.Plan(new float[SampleRate], SampleRate, maxInputSeconds: 1.5);

        Assert.Single(windows);
    }

    [Fact]
    public void RequireWithinLimit_RefusesAWindowedModelPastTheKnobButNotAnUnwindowedOne()
    {
        double past = EngineKnobs.TranscribeMaxSeconds.Value + 1;

        Assert.Throws<ArgumentException>(() => LongFormTranscription.RequireWithinLimit(30, past, "pcm"));
        LongFormTranscription.RequireWithinLimit(null, past, "pcm");
        LongFormTranscription.RequireWithinLimit(30, 3600, "pcm");
    }

    [Fact]
    public void Transcribe_TimedSpansStartingPastTheWindowEnd_AreDropped()
    {
        FakeRunner runner = new FakeRunner
        {
            SpansFor = _ =>
            [
                new SttSegment { Text = "real", Start = 0.2, End = 0.8 },
                new SttSegment { Text = "Thank you.", Start = 1.5, End = 3.0 },
            ],
        };
        SpeechWindow[] windows = [new(0, SampleRate, false, false), new(SampleRate, SampleRate, false, false)];

        LongFormTranscript result = Run(runner, new float[SampleRate * 2], windows, timed: true);

        Assert.NotNull(result.Segments);
        Assert.Equal(2, result.Segments.Count);
        Assert.Equal("real real", result.Text);
    }

    [Fact]
    public void Transcribe_TimedWindowWithNoTextAnywhere_AddsNoSegment()
    {
        FakeRunner runner = new FakeRunner
        {
            SpansFor = _ => [],
            TextFor = _ => string.Empty,
        };
        SpeechWindow[] windows = [new(0, SampleRate, false, false), new(SampleRate, SampleRate, false, false)];

        LongFormTranscript result = Run(runner, new float[SampleRate * 2], windows, timed: true);

        Assert.NotNull(result.Segments);
        Assert.Empty(result.Segments);
        Assert.Equal(string.Empty, result.Text);
        Assert.Equal(2, result.EmptyWindowCount);
    }

    [Fact]
    public void Transcribe_TextMode_JoinsWindowTextsInOrder()
    {
        FakeRunner runner = new FakeRunner { TextFor = piece => $"w{piece.Length}" };
        SpeechWindow[] windows = [new(0, 100, false, false), new(100, 50, false, false)];

        LongFormTranscript result = Run(runner, new float[150], windows, timed: false);

        Assert.Equal("w100 w50", result.Text);
        Assert.Null(result.Segments);
        Assert.Equal(2, result.WindowCount);
    }

    [Fact]
    public void Transcribe_SkipsSilentWindowsButCountsThem()
    {
        FakeRunner runner = new FakeRunner { TextFor = _ => "speech" };
        SpeechWindow[] windows = [new(0, 100, false, false), new(100, 100, true, false), new(200, 100, false, true)];

        LongFormTranscript result = Run(runner, new float[300], windows, timed: false);

        Assert.Equal("speech speech", result.Text);
        Assert.Equal(2, runner.TextCalls);
        Assert.Equal(3, result.WindowCount);
        Assert.Equal(1, result.SilentWindowCount);
        Assert.Equal(1, result.HardCutCount);
        Assert.Equal(0, result.EmptyWindowCount);
    }

    [Fact]
    public void Transcribe_NonSilentWindowThatDecodesToNothing_IsCountedAndItsAudioIsMissing()
    {
        FakeRunner runner = new FakeRunner { TextFor = piece => piece.Length == SampleRate ? string.Empty : "words" };
        SpeechWindow[] windows = [new(0, SampleRate, false, false), new(SampleRate, SampleRate / 2, false, false)];

        LongFormTranscript result = Run(runner, new float[SampleRate + SampleRate / 2], windows, timed: false);

        Assert.Equal(1, result.EmptyWindowCount);
        Assert.Equal("words", result.Text);
    }

    [Fact]
    public void Transcribe_TimedSegments_AreOffsetOntoTheClipTimeline()
    {
        FakeRunner runner = new FakeRunner
        {
            SpansFor = _ => [new SttSegment { Text = "a", Start = 0.5, End = 1.0 }, new SttSegment { Text = "b", Start = 1.0, End = 2.0 }],
        };
        SpeechWindow[] windows = [new(0, SampleRate * 2, false, false), new(SampleRate * 2, SampleRate * 2, false, false)];

        LongFormTranscript result = Run(runner, new float[SampleRate * 4], windows, timed: true);

        Assert.NotNull(result.Segments);
        Assert.Equal(4, result.Segments.Count);
        Assert.Equal(2.5, result.Segments[2].Start, precision: 9);
        Assert.Equal(3.0, result.Segments[2].End, precision: 9);
        Assert.Equal("a b a b", result.Text);
    }

    [Fact]
    public void Transcribe_TimedSpansPastTheWindowEnd_AreClamped()
    {
        FakeRunner runner = new FakeRunner
        {
            SpansFor = _ => [new SttSegment { Text = "x", Start = 0.0, End = 3.0 }],
        };
        SpeechWindow[] windows = [new(0, SampleRate, false, false), new(SampleRate, SampleRate, false, false)];

        LongFormTranscript result = Run(runner, new float[SampleRate * 2], windows, timed: true);

        Assert.NotNull(result.Segments);
        Assert.Equal(1.0, result.Segments[0].End, precision: 9);
    }

    [Fact]
    public void Transcribe_TimedWindowWithoutSegments_FallsBackToOneSpanningSegment()
    {
        FakeRunner runner = new FakeRunner
        {
            SpansFor = _ => [],
            TextFor = _ => "plain words",
        };
        SpeechWindow[] windows = [new(0, SampleRate, false, false), new(SampleRate, SampleRate, false, false)];

        LongFormTranscript result = Run(runner, new float[SampleRate * 2], windows, timed: true);

        Assert.NotNull(result.Segments);
        Assert.Equal(2, result.Segments.Count);
        Assert.Equal("plain words", result.Segments[1].Text);
        Assert.Equal(1.0, result.Segments[1].Start, precision: 9);
        Assert.Equal(2.0, result.Segments[1].End, precision: 9);
    }

    [Fact]
    public void Transcribe_ModelWithoutTimestamps_ReturnsTextAndNoSegments()
    {
        FakeRunner runner = new FakeRunner
        {
            SpansFor = null,
            TextFor = _ => "text",
        };
        SpeechWindow[] windows = [new(0, SampleRate, false, false), new(SampleRate, SampleRate, false, false)];

        LongFormTranscript result = Run(runner, new float[SampleRate * 2], windows, timed: true);

        Assert.Null(result.Segments);
        Assert.Equal("text text", result.Text);
    }

    [Fact]
    public void Transcribe_CancelledBetweenWindows_Throws()
    {
        using CancellationTokenSource cts = new CancellationTokenSource();
        FakeRunner runner = new FakeRunner { TextFor = _ => "x", OnTextCall = () => cts.Cancel() };
        SpeechWindow[] windows = [new(0, 100, false, false), new(100, 100, false, false)];

        Assert.Throws<OperationCanceledException>(() => LongFormTranscription.Transcribe(runner, null!, new float[200], SampleRate,
            windows, new AudioRequest { Audio = new AudioClip { Data = [] } }, timed: false, cts.Token));
        Assert.Equal(1, runner.TextCalls);
    }

    private static LongFormTranscript Run(FakeRunner runner, float[] audio, IReadOnlyList<SpeechWindow> windows, bool timed) =>
        LongFormTranscription.Transcribe(runner, null!, audio, SampleRate, windows,
            new AudioRequest { Audio = new AudioClip { Data = [] } }, timed, CancellationToken.None);

    private sealed class FakeRunner : ISttRunner
    {
        public Func<float[], string> TextFor { get; init; } = _ => string.Empty;

        public Func<float[], IReadOnlyList<SttSegment>>? SpansFor { get; init; }

        public Action? OnTextCall { get; init; }

        public int TextCalls { get; private set; }

        public double? MaxInputSeconds => 30;

        public string Transcribe(IBackend backend, float[] audioMono, AudioRequest request)
        {
            TextCalls++;
            OnTextCall?.Invoke();
            return TextFor(audioMono);
        }

        public IReadOnlyList<SttSegment>? TranscribeTimed(IBackend backend, float[] audioMono, AudioRequest request)
            => SpansFor?.Invoke(audioMono);

        public ScoreTranscriptResult? TranscribeScore(IBackend backend, float[] audioMono, AudioRequest request) => null;

        public void Dispose()
        {
        }
    }
}
