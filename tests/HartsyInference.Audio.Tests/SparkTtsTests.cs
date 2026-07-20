using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.SparkTts;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Sampling;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Checkpoint-free tests for Spark-TTS's testable surface: the config preset and the shared
/// <see cref="NucleusSampler"/> core (now used by both CosyVoice and Spark-TTS). The LM forward + BiCodec
/// decode need the ~3.95 GB checkpoint.</summary>
public sealed class SparkTtsTests
{
    private readonly ITestOutputHelper _out;
    public SparkTtsTests(ITestOutputHelper o) => _out = o;

    /// <summary>Real-weight end-to-end, fully in-engine (controllable mode): text + coarse style → Spark prompt
    /// tokenizer (Qwen2.5 BPE + added tokens) → Qwen2.5-0.5B greedy AR → 32 global + N semantic BiCodec tokens →
    /// 16 kHz waveform. Gated on <c>SPARK_DIR</c> (the Spark-TTS-0.5B root holding <c>LLM/</c> and <c>BiCodec/</c>).
    /// Greedy is deterministic; logits, BiCodec decode, and these token streams were validated bit-exact against
    /// the reference. First asserts the tokenizer reproduces the validated prompt ids, then the audio.</summary>
    [Fact]
    public void EndToEnd_RealWeights_ControllablePrompt()
    {
        string? dir = Environment.GetEnvironmentVariable("SPARK_DIR");
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(Path.Combine(dir, "LLM"))) return;

        const string text = "Hello, this is a test of the speech synthesizer.";

        // Tokenizer parity: the in-engine prompt build + tokenize must reproduce the validated 20 ids.
        SparkTtsTokenizer tok = SparkTtsTokenizer.FromDirectory(Path.Combine(dir, "LLM"));
        int[] gotIds = tok.Encode(SparkTtsPipeline.BuildControllablePrompt(text, "male", "moderate", "moderate"));
        int[] wantIds = [165143, 165146, 9707, 11, 419, 374, 264, 1273, 315, 279, 8806, 51289, 3135, 13,
            165152, 165147, 165033, 164967, 164988, 165153];
        Assert.Equal(wantIds, gotIds);

        using SparkTtsPipeline pipe = SparkTtsPipeline.LoadFromDirectory(dir);
        using CpuBackend backend = new();
        float[] audio = pipe.SynthesizeControllable(backend, text, "male", "moderate", "moderate",
            maxTokens: 800, greedy: true);

        Assert.True(audio.Length > 0 && audio.Length % 320 == 0);   // hop 320 → 16 kHz
        double sumSq = 0;
        foreach (float a in audio) { Assert.True(float.IsFinite(a)); sumSq += (double)a * a; }
        double rms = Math.Sqrt(sumSq / audio.Length);
        _out.WriteLine($"{audio.Length} samples ({audio.Length / 16000.0:F2}s) rms={rms:F4}");
        Assert.True(rms > 0.01, $"audio near-silent (rms {rms:F4})");
    }

    [Fact]
    public void Config_V0_5B_HasExpectedValues()
    {
        SparkTtsConfig c = SparkTtsConfig.V0_5B;
        Assert.Equal(166_000, c.Llm.VocabSize);
        Assert.Equal(896, c.Llm.HiddenSize);
        Assert.Equal(24, c.Llm.NumHiddenLayers);
        Assert.Equal(8_192, c.SemanticVocab);
        Assert.Equal(4_096, c.GlobalVocab);
        Assert.Equal(32, c.NumGlobalTokens);
        Assert.Equal(16_000, c.SampleRate);
        // Real added_tokens.json layout: global tokens precede semantic tokens, contiguously.
        Assert.Equal(151_665, c.GlobalTokenBase);
        Assert.Equal(c.GlobalTokenBase + c.GlobalVocab, c.SemanticTokenBase);
        // BiCodec global FSQ levels multiply to the global vocab.
        int fsq = 1;
        foreach (int l in c.BiCodec.FsqLevels) fsq *= l;
        Assert.Equal(c.GlobalVocab, fsq);
    }

    [Fact]
    public void NucleusSampler_TopK1_ReturnsArgmax()
    {
        uint rng = DeterministicRng.Seed(1);
        Span<float> logits = [0.1f, 0.2f, 5.0f, 0.3f];
        Assert.Equal(2, NucleusSampler.Draw(logits, 4, temperature: 1f, topK: 1, topP: 1f, ref rng));
    }

    [Fact]
    public void NucleusSampler_IsDeterministic_ForFixedRngState()
    {
        Span<float> logits = [1f, 2f, 1.5f, 0.5f, 3f, 0.2f];
        uint a = DeterministicRng.Seed(42);
        uint b = DeterministicRng.Seed(42);
        for (int i = 0; i < 20; i++)
            Assert.Equal(
                NucleusSampler.Draw(logits, 6, 0.8f, 25, 0.9f, ref a),
                NucleusSampler.Draw(logits, 6, 0.8f, 25, 0.9f, ref b));
    }

    [Fact]
    public void NucleusSampler_RespectsCandidateCount()
    {
        uint rng = DeterministicRng.Seed(7);
        // 8-wide buffer but only the first 4 are candidates; indices 4..7 (huge) must never be drawn.
        Span<float> logits = [1f, 1f, 1f, 1f, 99f, 99f, 99f, 99f];
        for (int i = 0; i < 100; i++)
            Assert.InRange(NucleusSampler.Draw(logits, 4, 1f, 25, 1f, ref rng), 0, 3);
    }

    [Fact]
    public void NucleusSampler_MaskToken_IsNeverDrawn()
    {
        uint rng = DeterministicRng.Seed(3);
        Span<float> logits = [0f, 20f, 0.5f, 0.5f];   // index 1 dominates
        for (int i = 0; i < 50; i++)
            Assert.NotEqual(1, NucleusSampler.Draw(logits, 4, 1f, 5, 1f, ref rng, maskToken: 1));
    }
}
