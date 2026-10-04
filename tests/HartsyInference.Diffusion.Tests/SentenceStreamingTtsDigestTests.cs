using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Engine.Audio;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Gates for the shared sentence-streaming refactor, on real weights and the CPU backend.
///
/// <para>Piper: the streamed concatenation must be byte-identical to what the descriptor's previous private loop
/// produced. That loop was <see cref="SentenceSplitter.Split"/> followed by one <c>SynthesizeText(sentence, seed)</c>
/// per sentence, which is exactly the runner's non-streaming <c>Synthesize</c> called per sentence, so that is the
/// reference computed here. Kokoro: the whole-text <c>Synthesize</c> is unchanged and must match a direct pipeline
/// call; the new sentence stream is Whisper-verified for intelligibility; a job whose token is cancelled stops in the
/// pipeline. Every test skips through <see cref="RealWeightGate"/> when its files are not on the box.</para></summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class SentenceStreamingTtsDigestTests
{
    private const string PiperVoice = "en_US-ryan-medium";
    private const int Seed = 1234;

    private const string Passage =
        "Good morning. The forecast today calls for scattered clouds with a high of seventy two degrees and a "
        + "light breeze from the northwest. You have three meetings on your calendar, the first at nine thirty. "
        + "Traffic on your usual route is moving normally, so leaving at nine should be plenty of time.";

    private const string TwoSentences =
        "The weather tomorrow is clear and mild with a gentle breeze. Rain arrives on Thursday evening, so bring a coat.";
    private static readonly string[] TwoSentenceContentWords =
        ["weather", "tomorrow", "clear", "mild", "gentle", "breeze", "rain", "thursday", "evening", "coat"];

    private readonly ITestOutputHelper _output;
    public SentenceStreamingTtsDigestTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task Piper_StreamedConcatenation_IsByteIdenticalToThePerSentenceReferenceLoop()
    {
        string voiceDir = Path.Combine(AudioModelCache.GetRepoDirectory("rhasspy/piper-voices", "tts"), "en", "en_US", "ryan", "medium");
        string onnx = Path.Combine(voiceDir, PiperVoice + ".onnx");
        if (!RealWeightGate.Require(_output.WriteLine, onnx, onnx + ".json")) return;
        if (!EspeakDataPresent()) { _output.WriteLine("SKIPPED: no espeak-ng-data directory (ESPEAK_DATA_DIR or the distro path)"); return; }

        using CpuBackend backend = new();
        using ITtsRunner runner = await PiperModel.Descriptor.LoadAsync(new TtsLoadContext { Backend = backend }, PiperVoice, CancellationToken.None);
        IStreamingTtsRunner streaming = Assert.IsAssignableFrom<IStreamingTtsRunner>(runner);
        TtsJob job = new() { Text = Passage, Seed = Seed };

        IReadOnlyList<string> sentences = SentenceSplitter.Split(Passage);
        Assert.True(sentences.Count >= 3, $"the passage should split into several sentences, got {sentences.Count}");
        List<float[]> reference = [];
        foreach (string sentence in sentences)
        {
            reference.Add(runner.Synthesize(backend, job with { Text = sentence }));
        }
        // The reference must itself be reproducible, or a digest comparison proves nothing.
        List<float[]> again = [];
        foreach (string sentence in sentences)
        {
            again.Add(runner.Synthesize(backend, job with { Text = sentence }));
        }
        Assert.Equal(PcmDigest.Of(reference), PcmDigest.Of(again));

        List<AudioChunk> chunks = [];
        await foreach (AudioChunk chunk in streaming.SynthesizeStream(backend, job, CancellationToken.None))
        {
            chunks.Add(chunk);
        }

        Assert.Equal(reference.Count, chunks.Count);
        long offset = 0;
        for (int i = 0; i < chunks.Count; i++)
        {
            Assert.Equal(runner.SampleRate, chunks[i].SampleRate);
            Assert.Equal(offset, chunks[i].StartSampleOffset);
            offset += chunks[i].Samples.Length;
        }
        string referenceDigest = PcmDigest.Of(reference);
        string streamedDigest = PcmDigest.Of(chunks.Select(c => c.Samples));
        _output.WriteLine($"Piper {PiperVoice}: {chunks.Count} sentence chunks, {offset} samples, digest {streamedDigest}");
        Assert.Equal(referenceDigest, streamedDigest);
    }

    [Fact]
    public async Task Kokoro_WholeTextSynthesize_MatchesADirectPipelineCall()
    {
        if (!KokoroPresent(out string[] g2pFiles)) return;

        using CpuBackend backend = new();
        using ITtsRunner runner = await TtsCatalog.Kokoro.LoadAsync(new TtsLoadContext { Backend = backend }, "", CancellationToken.None);
        float[] viaRunner = runner.Synthesize(backend, new TtsJob { Text = TwoSentences });

        using KokoroPipeline direct = await KokoroPipeline.LoadAsync();
        EnglishG2P g2p = new(MisakiLexicon.FromFiles(g2pFiles[0], g2pFiles[1]), g2pFiles[2]);
        float[] viaPipeline = direct.Synthesize(backend, g2p.ToIpa(TwoSentences), voiceName: "af_heart", speed: 1f);

        _output.WriteLine($"Kokoro whole-text: {viaRunner.Length} samples, digest {PcmDigest.Of(viaRunner)}");
        Assert.Equal(PcmDigest.Of(viaPipeline), PcmDigest.Of(viaRunner));
    }

    [Fact]
    public async Task Kokoro_ACancelledJob_StopsInThePipeline()
    {
        if (!KokoroPresent(out _)) return;

        using CpuBackend backend = new();
        using ITtsRunner runner = await TtsCatalog.Kokoro.LoadAsync(new TtsLoadContext { Backend = backend }, "", CancellationToken.None);

        Assert.Throws<OperationCanceledException>(() =>
            runner.Synthesize(backend, new TtsJob { Text = TwoSentences, Cancel = new CancellationToken(canceled: true) }));
    }

    [Fact]
    public async Task Kokoro_StreamedSentences_AreIntelligible_WhisperVerified()
    {
        if (!KokoroPresent(out _)) return;
        const string WhisperRepo = "openai/whisper-base";
        string whisperWeights = Path.Combine(AudioModelCache.GetRepoDirectory(WhisperRepo, "stt"), "model.safetensors");
        if (!RealWeightGate.Require(_output.WriteLine, whisperWeights)) return;

        using CpuBackend backend = new();
        using ITtsRunner runner = await TtsCatalog.Kokoro.LoadAsync(new TtsLoadContext { Backend = backend }, "", CancellationToken.None);
        IStreamingTtsRunner streaming = Assert.IsAssignableFrom<IStreamingTtsRunner>(runner);

        List<AudioChunk> chunks = [];
        long started = Environment.TickCount64;
        long firstChunkMs = -1;
        await foreach (AudioChunk chunk in streaming.SynthesizeStream(backend, new TtsJob { Text = TwoSentences }, CancellationToken.None))
        {
            if (firstChunkMs < 0)
            {
                firstChunkMs = Environment.TickCount64 - started;
            }
            chunks.Add(chunk);
        }
        long totalMs = Environment.TickCount64 - started;
        Assert.Equal(2, chunks.Count);
        Assert.Equal(0, chunks[0].StartSampleOffset);
        Assert.Equal(chunks[0].Samples.Length, chunks[1].StartSampleOffset);
        float[] joined = PcmDigest.Concat(chunks.Select(c => c.Samples));
        _output.WriteLine($"Kokoro stream: first chunk after {firstChunkMs} ms, both after {totalMs} ms, "
            + $"{joined.Length / 24_000.0:F2} s of audio (CPU), digest {PcmDigest.Of(joined)}");

        using WhisperPipeline stt = await WhisperPipeline.LoadAsync(WhisperRepo);
        string heard = stt.TranscribeAudio(backend, joined, runner.SampleRate,
            new WhisperOptions { Language = "en", Translate = false, WithTimestamps = false }).Trim();
        _output.WriteLine($"Whisper heard: \"{heard}\"");
        string lower = heard.ToLowerInvariant();
        string[] hits = [.. TwoSentenceContentWords.Where(w => lower.Contains(w, StringComparison.Ordinal))];
        double recall = hits.Length / (double)TwoSentenceContentWords.Length;
        _output.WriteLine($"Content-word recall: {hits.Length}/{TwoSentenceContentWords.Length} ({recall:P0})");
        Assert.True(recall >= 0.8, $"Whisper recall {recall:P0} on \"{heard}\" — the sentence-chunked Kokoro stream is not intelligible");
    }

    /// <summary>Kokoro's config, weights (the published repack or the in-engine conversion of the canonical .pth), the
    /// default voice and the G2P dictionaries. Checked before any load so nothing is downloaded by a test.</summary>
    private bool KokoroPresent(out string[] g2pFiles)
    {
        string repoDir = AudioModelCache.GetRepoDirectory("hexgrad/Kokoro-82M", "tts");
        string repack = Path.Combine(AudioModelCache.GetRepoDirectory("Hartsy/kokoro-82m-safetensors", "tts"), "kokoro-82m.safetensors");
        string converted = Path.Combine(repoDir, "kokoro-82m.safetensors");
        string weights = File.Exists(repack) ? repack : converted;
        g2pFiles = [AudioModelRoot.SharedFile("misaki_us_gold.json"), AudioModelRoot.SharedFile("misaki_us_silver.json"),
            AudioModelRoot.SharedFile("cmudict.dict")];
        return RealWeightGate.Require(_output.WriteLine,
            [Path.Combine(repoDir, "config.json"), weights, Path.Combine(repoDir, "voices", "af_heart.bin"), .. g2pFiles]);
    }

    /// <summary>Mirrors <c>EspeakPhonemizer.FromCache</c>'s search, so the Piper test skips rather than throws without the data.</summary>
    private static bool EspeakDataPresent()
    {
        string? env = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        string[] candidates =
        [
            env ?? "",
            Path.Combine(AudioModelCache.CacheRoot, "Hartsy--espeak-ng-data", "espeak-ng-data"),
            Path.Combine(AudioModelCache.CacheRoot, "Hartsy--espeak-ng-data"),
            "/usr/lib/x86_64-linux-gnu/espeak-ng-data",
            "/usr/share/espeak-ng-data",
        ];
        return candidates.Any(dir => dir.Length > 0 && File.Exists(Path.Combine(dir, "phontab")));
    }
}
