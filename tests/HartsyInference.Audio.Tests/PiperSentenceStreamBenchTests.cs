using System.Diagnostics;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>How long a listener waits for the first sound, whole-passage against sentence-at-a-time.
///
/// <para>Piper is a whole-utterance model: nothing comes out until every phoneme of the text has been through
/// the decoder. Splitting the text and synthesizing sentence by sentence does not make the total any faster —
/// it is the same work plus a little per-call overhead — but it changes when the first sample exists, from
/// "after all of it" to "after the first sentence". On a voice assistant that is the whole difference between a
/// pause and an answer.</para>
///
/// <para>This measures both, on the same text and the same pipeline, and asserts the relationship rather than
/// an absolute number, because the absolute depends entirely on the machine. Gated on the cached voice and
/// <c>ESPEAK_DATA_DIR</c>, like the other Piper bench.</para></summary>
public sealed class PiperSentenceStreamBenchTests
{
    private readonly ITestOutputHelper _out;
    public PiperSentenceStreamBenchTests(ITestOutputHelper o) => _out = o;

    /// <summary>The audit's long TTS text: four sentences, about 11 s of speech.</summary>
    private const string LongText =
        "Good morning. The forecast today calls for scattered clouds with a high of seventy two degrees and a "
        + "light breeze from the northwest. You have three meetings on your calendar, the first at nine thirty. "
        + "Traffic on your usual route is moving normally, so leaving at nine should be plenty of time to "
        + "arrive without rushing.";

    [Fact]
    public void FirstSentenceArrivesLongBeforeTheWholePassage()
    {
        string onnx = Path.Combine(AudioModelCache.GetRepoDirectory("rhasspy/piper-voices", "tts"),
            "en", "en_US", "lessac", "medium", "en_US-lessac-medium.onnx");
        if (!File.Exists(onnx) || string.IsNullOrEmpty(Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR")))
        {
            _out.WriteLine("Piper onnx or ESPEAK_DATA_DIR missing — skip.");
            return;
        }

        using PiperPipeline pipeline = PiperPipeline.LoadFromFiles(onnx, onnx + ".json");
        using IBackend backend = new CpuBackend();
        pipeline.SynthesizeText(backend, "Warm up.", seed: 1234);   // first call pays for lazy init

        Stopwatch clock = Stopwatch.StartNew();
        float[] whole = pipeline.SynthesizeText(backend, LongText, seed: 1234);
        double wholeSeconds = clock.Elapsed.TotalSeconds;
        double audioSeconds = whole.Length / (double)pipeline.SampleRate;

        IReadOnlyList<string> sentences = SentenceSplitter.Split(LongText);
        Assert.True(sentences.Count >= 3, $"expected the passage to split; got {sentences.Count} piece(s)");

        clock.Restart();
        double firstSeconds = -1;
        int total = 0;
        foreach (string sentence in sentences)
        {
            float[] part = pipeline.SynthesizeText(backend, sentence, seed: 1234);
            if (firstSeconds < 0)
            {
                firstSeconds = clock.Elapsed.TotalSeconds;
            }
            total += part.Length;
        }
        double splitSeconds = clock.Elapsed.TotalSeconds;

        _out.WriteLine($"whole passage : {wholeSeconds:F3}s for {audioSeconds:F2}s of audio (RTF {wholeSeconds / audioSeconds:F3})");
        _out.WriteLine($"first sentence: {firstSeconds:F3}s  ({firstSeconds / wholeSeconds:P0} of the whole)");
        _out.WriteLine($"all {sentences.Count} sentences: {splitSeconds:F3}s for {total / (double)pipeline.SampleRate:F2}s of audio");

        // The point of the exercise. Half is a deliberately loose bound — the real figure on a four-sentence
        // passage is nearer a fifth — so the test fails on a broken split rather than on a slow machine.
        Assert.True(firstSeconds < wholeSeconds / 2,
            $"first sentence took {firstSeconds:F3}s against {wholeSeconds:F3}s for the whole passage, so "
            + "splitting bought nothing");

        // And the split must not cost the total much: this is a latency change, not a throughput one.
        Assert.True(splitSeconds < wholeSeconds * 1.5,
            $"synthesizing sentence by sentence took {splitSeconds:F3}s against {wholeSeconds:F3}s whole");
    }
}
