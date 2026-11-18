using HartsyInference.Audio.Io;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Whisper;

/// <summary>Skips unless the clip, the voiced-burst times of each word and the cached whisper-base weights are all present.</summary>
public sealed class WhisperAlignmentFactAttribute : FactAttribute
{
    internal const string WavVariable = "HARTSY_ALIGN_WAV";
    internal const string TruthVariable = "HARTSY_ALIGN_TRUTH";

    public WhisperAlignmentFactAttribute()
    {
        foreach (string name in new[] { WavVariable, TruthVariable })
        {
            string? path = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            {
                Skip = $"SKIPPED: {name} is unset or not an existing file; real-weight word alignment did not run.";
                return;
            }
        }
    }
}

/// <summary>Word alignment on real speech: a clip built from separately spoken words, with the voiced burst of each measured from its energy. Needs whisper-base in the model cache.</summary>
[Trait("Category", "Integration")]
public sealed class WordAlignmentRealWeightsTests(ITestOutputHelper output)
{
    // The truth file lists where each word is actually voiced (measured from the clip's energy), not its padded piece.
    private const double MinBurstOverlap = 0.5;

    private static (string Word, double Start, double End)[] ReadTruth()
    {
        return File.ReadAllLines(Environment.GetEnvironmentVariable(WhisperAlignmentFactAttribute.TruthVariable)!)
            .Where(line => line.Length > 0)
            .Select(line => line.Split('\t'))
            .Select(parts => (parts[0], double.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture),
                double.Parse(parts[2], System.Globalization.CultureInfo.InvariantCulture)))
            .ToArray();
    }

    private static void AssertEachWordCoversItsVoicedBurst(IReadOnlyList<WhisperWord> words, (string Word, double Start, double End)[] truth)
    {
        Assert.Equal(truth.Length, words.Count);
        for (int i = 0; i < words.Count; i++)
        {
            Assert.True(words[i].End >= words[i].Start, $"word {i} runs backwards");
            double overlap = Math.Min(words[i].End, truth[i].End) - Math.Max(words[i].Start, truth[i].Start);
            Assert.True(overlap >= MinBurstOverlap * (truth[i].End - truth[i].Start),
                $"word {i} '{truth[i].Word}' {words[i].Start:0.00}-{words[i].End:0.00} misses its burst {truth[i].Start:0.00}-{truth[i].End:0.00}");
            if (i > 0)
            {
                Assert.True(words[i].Start >= words[i - 1].End - 1e-6, $"word {i} starts before word {i - 1} ends");
            }
        }
    }

    [WhisperAlignmentFact]
    public async Task KnownText_IsTimedInsideEachSpokenPiece()
    {
        (string Word, double Start, double End)[] truth = ReadTruth();
        WavFile.DecodedAudio clip = WavFile.Read(Environment.GetEnvironmentVariable(WhisperAlignmentFactAttribute.WavVariable)!);
        using WhisperPipeline pipeline = await WhisperPipeline.LoadAsync("openai/whisper-base");
        IBackend backend = new CpuBackend();

        IReadOnlyList<WhisperWord> words = pipeline.AlignWords(backend, clip.ToMono(), clip.SampleRate,
            string.Join(" ", truth.Select(piece => piece.Word)));

        foreach (WhisperWord word in words)
        {
            output.WriteLine($"{word.Start:0.00}-{word.End:0.00} {word.Text}");
        }
        AssertEachWordCoversItsVoicedBurst(words, truth);
    }

    [WhisperAlignmentFact]
    public async Task Transcript_IsTimedWordByWord()
    {
        (string Word, double Start, double End)[] truth = ReadTruth();
        WavFile.DecodedAudio clip = WavFile.Read(Environment.GetEnvironmentVariable(WhisperAlignmentFactAttribute.WavVariable)!);
        using WhisperPipeline pipeline = await WhisperPipeline.LoadAsync("openai/whisper-base");
        IBackend backend = new CpuBackend();

        IReadOnlyList<WhisperWord> words = pipeline.AlignWords(backend, clip.ToMono(), clip.SampleRate);

        foreach (WhisperWord word in words)
        {
            output.WriteLine($"{word.Start:0.00}-{word.End:0.00} {word.Text}");
        }
        Assert.NotEmpty(words);
        Assert.True(words[^1].End <= clip.ToMono().Length / (double)clip.SampleRate + 0.1);
        for (int i = 1; i < words.Count; i++)
        {
            Assert.True(words[i].Start >= words[i - 1].End - 1e-6);
        }
    }
}
