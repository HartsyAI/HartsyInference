using HartsyInference.Audio.Dsp.Windowing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Tests.Audio;

/// <summary>How word-timed spans move through the engine's transcript assembly: joining, long-form offsets and diarization spans.</summary>
public sealed class WordTimedTranscriptTests
{
    private sealed class FakeRunner(Queue<IReadOnlyList<SttSegment>> timed, string plain) : ISttRunner
    {
        public double? MaxInputSeconds => 10d;

        public string Transcribe(IBackend backend, float[] audioMono, AudioRequest request) => plain;

        public IReadOnlyList<SttSegment>? TranscribeTimed(IBackend backend, float[] audioMono, AudioRequest request) => timed.Dequeue();

        public ScoreTranscriptResult? TranscribeScore(IBackend backend, float[] audioMono, AudioRequest request) => null;

        public void Dispose()
        {
        }
    }

    private static SttSegment Word(string text, double start, double end) => new() { Text = text, Start = start, End = end, IsWord = true };

    private static AudioRequest Request(bool words) => new() { Audio = new AudioClip { Data = [] }, WordTimestamps = words, Diarization = !words };

    [Fact]
    public void Join_ConcatenatesWordsAsTheyAre_AndSpacesPhrases()
    {
        Assert.Equal("Hello, world", SttSegment.Join([Word(" Hello,", 0, 1), Word(" world", 1, 2)]));
        Assert.Equal("你好", SttSegment.Join([Word("你", 0, 1), Word("好", 1, 2)]));
        Assert.Equal("one two", SttSegment.Join([
            new SttSegment { Text = " one ", Start = 0, End = 1 },
            new SttSegment { Text = "two", Start = 1, End = 2 },
        ]));
    }

    [Fact]
    public void LongForm_MovesWordsOntoTheClipTimeline_AndJoinsTheWindowsText()
    {
        Queue<IReadOnlyList<SttSegment>> timed = new();
        timed.Enqueue([Word(" Hello", 0.0, 0.5), Word(" world", 0.5, 1.0)]);
        timed.Enqueue([Word(" again", 0.2, 0.9)]);
        SpeechWindow[] windows = [new(0, 100, false, false), new(100, 100, false, false)];

        LongFormTranscript result = LongFormTranscription.Transcribe(new FakeRunner(timed, "unused"), new CpuBackend(),
            new float[200], sampleRate: 10, windows, Request(words: true), timed: true, CancellationToken.None);

        Assert.Equal("Hello world again", result.Text);
        Assert.NotNull(result.Segments);
        Assert.Equal([0.0, 0.5, 10.2], result.Segments.Select(span => span.Start).ToArray());
        Assert.All(result.Segments, span => Assert.True(span.IsWord));
    }

    [Fact]
    public void LongForm_LeavesAWholeWindowPhraseOutOfAWordRequest_ButKeepsItForPhraseTimestamps()
    {
        SpeechWindow[] windows = [new(0, 100, false, false)];

        LongFormTranscript forWords = LongFormTranscription.Transcribe(new FakeRunner(new Queue<IReadOnlyList<SttSegment>>([[]]), "spoken"),
            new CpuBackend(), new float[100], sampleRate: 10, windows, Request(words: true), timed: true, CancellationToken.None);
        LongFormTranscript forPhrases = LongFormTranscription.Transcribe(new FakeRunner(new Queue<IReadOnlyList<SttSegment>>([[]]), "spoken"),
            new CpuBackend(), new float[100], sampleRate: 10, windows, Request(words: false), timed: true, CancellationToken.None);

        Assert.Equal("spoken", forWords.Text);
        Assert.Empty(forWords.Segments!);
        Assert.Equal("spoken", forPhrases.Text);
        Assert.Single(forPhrases.Segments!);
    }

    [Fact]
    public void Diarization_EmbedsRunsOfWords_NotWordsTooShortToEmbed()
    {
        SttSegment[] words =
        [
            Word(" a", 0.0, 0.3), Word(" b", 0.3, 0.6), Word(" c", 0.6, 0.9),
            Word(" d", 3.0, 3.3), Word(" e", 3.3, 3.6),
        ];

        List<SttSegment> spans = SpeakerDiarizer.BuildSpans(words, totalSeconds: 4.0);

        Assert.Equal(2, spans.Count);
        Assert.Equal((0.0, 0.9, "a b c"), (spans[0].Start, spans[0].End, spans[0].Text));
        Assert.Equal((3.0, 3.6, "d e"), (spans[1].Start, spans[1].End, spans[1].Text));
    }
}
