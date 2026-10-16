using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>A tiny random Drama 3 checkpoint run through fish-speech's own <c>FishOmniBackbone</c> and <c>FishOmniAudioDecoder</c>
/// (<c>tools/fish_audio/drama3_reference.py</c>, float32): the port must reproduce the prompt's backbone hidden states, the main-token
/// logits, every codebook head's logits under forced frames, and the hidden state after feeding a frame (the incremental decode).</summary>
public sealed class FishAudioDrama3ParityTests(ITestOutputHelper output)
{
    private const float Tolerance = 2e-4f;
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "FishAudioDrama3");

    [Fact]
    public void ScoredFrames_MatchOfficialImplementation()
    {
        using Expected expected = new();
        FishAudioDrama3Score score = Score("drama3_tiny.safetensors", FishAudioDrama3Lm.DefaultPrefillChunk, expected);
        AssertClose(expected.Read("ref.hidden"), score.PromptHidden, "prompt hidden");
        AssertClose(Candidates(expected.Read("ref.logits0"), score.Candidates), score.CandidateLogits[0], "candidate logits 0");
        AssertClose(expected.Read("ref.head1"), score.HeadLogits[0], "head logits 1");
        AssertClose(expected.Read("ref.hidden1"), score.Hidden[0], "hidden after frame 1");
        AssertClose(Candidates(expected.Read("ref.logits1"), score.Candidates), score.CandidateLogits[1], "candidate logits 1");
        AssertClose(expected.Read("ref.head2"), score.HeadLogits[1], "head logits 2");
    }

    [Fact]
    public void FusedExpertStacks_LoadAsThePerExpertTensors()
    {
        using Expected expected = new();
        FishAudioDrama3Score split = Score("drama3_tiny.safetensors", FishAudioDrama3Lm.DefaultPrefillChunk, expected);
        FishAudioDrama3Score fused = Score("drama3_tiny_fused.safetensors", FishAudioDrama3Lm.DefaultPrefillChunk, expected);
        Assert.Equal(split.PromptHidden, fused.PromptHidden);
        Assert.Equal(split.HeadLogits[1], fused.HeadLogits[1]);
    }

    [Fact]
    public void ChunkedPrefill_MatchesOnePass()
    {
        using Expected expected = new();
        FishAudioDrama3Score whole = Score("drama3_tiny.safetensors", FishAudioDrama3Lm.DefaultPrefillChunk, expected);
        FishAudioDrama3Score chunked = Score("drama3_tiny.safetensors", 3, expected);
        AssertClose(whole.PromptHidden, chunked.PromptHidden, "prompt hidden");
        AssertClose(whole.HeadLogits[0], chunked.HeadLogits[0], "head logits 1");
        AssertClose(whole.Hidden[1], chunked.Hidden[1], "hidden after frame 2");
    }

    [Fact]
    public void Q8Cache_HoldsExactlyTheQuantizedWeights_AndRuns()
    {
        using Expected expected = new();
        string cache = Path.Combine(Path.GetTempPath(), $"hartsy-drama3-q8-{Guid.NewGuid():N}.gguf");
        List<Tensor> owned = [];
        try
        {
            FishAudioDrama3Config config = Config();
            using HfQuantCheckpoint source = HfQuantCheckpoint.OpenFiles([Path.Combine(Dir, "drama3_tiny.safetensors")]);
            HfQuantCheckpoint checkpoint = HfQuantCheckpoint.OpenFiles([Path.Combine(Dir, "drama3_tiny.safetensors")]);
            Assert.False(FishAudioDrama3Weights.UsesStoredWeights(checkpoint, FishAudioDrama3Precision.Q8_0));
            (IReadOnlyDictionary<string, Tensor> weights, IDisposable owner) = FishAudioDrama3Weights.Load(checkpoint, FishAudioDrama3Precision.Q8_0, cache);
            using (owner)
            {
                Assert.Equal(DType.Q8_0, weights["model.language_model.layers.0.mlp.experts.1.gate_proj.weight"].DType);
                Assert.Equal(DType.Q8_0, weights["model.language_model.layers.1.self_attn.q_proj.weight"].DType);
                Assert.Equal(DType.F32, weights["model.language_model.layers.0.mlp.gate.weight"].DType);
                Assert.Equal(DType.F32, weights["audio_decoder_mtp.heads.0.weight"].DType);
                // The CPU backend multiplies F32 only; the engine widens block weights for it the same way.
                Dictionary<string, Tensor> widened = new(StringComparer.Ordinal);
                foreach ((string name, Tensor t) in weights)
                {
                    using Tensor original = source.Borrow(name);
                    if (!t.DType.IsQuantized)
                    {
                        Assert.True(original.AsReadOnlySpan<float>().SequenceEqual(t.AsReadOnlySpan<float>()), $"{name} {t.Shape} {original.Shape}");
                        widened[name] = t;
                        continue;
                    }
                    using Tensor requantized = GgufQuantizer.Quantize(original, t.DType);
                    using Tensor want = GgufDequantizer.Dequantize(requantized, DType.F32);
                    Tensor got = GgufDequantizer.Dequantize(t, DType.F32);
                    owned.Add(got);
                    Assert.True(want.AsReadOnlySpan<float>().SequenceEqual(got.AsReadOnlySpan<float>()), $"{name} {t.Shape} {original.Shape}");
                    widened[name] = got;
                }
                using CpuBackend backend = new();
                using FishAudioDrama3Lm lm = new(config, widened, capacity: 64);
                FishAudioDrama3Score score = lm.Score(backend, expected.Prompt(), [expected.Frame("in.frame1"), expected.Frame("in.frame2")]);
                Assert.All(score.HeadLogits[1], v => Assert.True(float.IsFinite(v)));
                // Informational: on a random tiny model a near-tied top-2 route flips under any perturbation.
                output.WriteLine($"Q8_0 vs reference: prompt hidden max |Δ| = {MaxAbs(expected.Read("ref.hidden"), score.PromptHidden):G3}");
            }
        }
        finally
        {
            foreach (Tensor t in owned) t.Dispose();
            File.Delete(cache);
        }
    }

    [Fact]
    public void Generate_IsDeterministicPerSeed_AndKeepsCodesInRange()
    {
        using Expected expected = new();
        FishAudioDrama3Config config = Config();
        using HfQuantCheckpoint checkpoint = HfQuantCheckpoint.OpenFiles([Path.Combine(Dir, "drama3_tiny.safetensors")]);
        Dictionary<string, Tensor> weights = FishAudioDrama3Weights.Borrow(checkpoint);
        using CpuBackend backend = new();
        using FishAudioDrama3Lm lm = new(config, weights, capacity: 64);
        int[,] prompt = expected.Prompt();
        FishAudioDrama3Generation first = lm.Generate(backend, prompt, 12, new FishAudioDrama3Sampler(config, seed: 5));
        FishAudioDrama3Generation again = lm.Generate(backend, prompt, 12, new FishAudioDrama3Sampler(config, seed: 5));
        Assert.Equal(first.Codes, again.Codes);
        Assert.Equal(first.ReachedEos, again.ReachedEos);
        Assert.Equal(config.NumCodebooks, first.Codes.GetLength(0));
        foreach (int code in first.Codes) Assert.InRange(code, 0, config.CodebookSize - 1);
    }

    private static FishAudioDrama3Score Score(string file, int prefillChunk, Expected expected)
    {
        FishAudioDrama3Config config = Config();
        using HfQuantCheckpoint checkpoint = HfQuantCheckpoint.OpenFiles([Path.Combine(Dir, file)]);
        Dictionary<string, Tensor> weights = FishAudioDrama3Weights.Borrow(checkpoint);
        using CpuBackend backend = new();
        using FishAudioDrama3Lm lm = new(config, weights, capacity: 64, prefillChunk);
        return lm.Score(backend, expected.Prompt(), [expected.Frame("in.frame1"), expected.Frame("in.frame2")]);
    }

    private static FishAudioDrama3Config Config() => FishAudioDrama3Config.FromJson(File.ReadAllText(Path.Combine(Dir, "config.json")));

    private static float MaxAbs(float[] expected, float[] actual)
    {
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
        return worst;
    }

    private static float[] Candidates(float[] logits, int[] candidates) => [.. candidates.Select(id => logits[id])];

    private void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
        output.WriteLine($"{what}: max |Δ| = {worst:G3}");
        Assert.True(worst <= Tolerance, $"{what}: max |Δ| = {worst}");
    }

    private sealed class Expected : IDisposable
    {
        private readonly SafeTensorsLoader _loader = new();

        public Expected() => _loader.Load(Path.Combine(Dir, "drama3_tiny_expected.safetensors"));

        public float[] Read(string name) => _loader.GetTensor(name).AsReadOnlySpan<float>().ToArray();

        public int[] Frame(string name) => _loader.GetTensor(name).AsReadOnlySpan<int>().ToArray();

        public int[,] Prompt()
        {
            Tensor t = _loader.GetTensor("in.prompt");
            int rows = (int)t.Shape[0], width = (int)t.Shape[1];
            ReadOnlySpan<int> values = t.AsReadOnlySpan<int>();
            int[,] prompt = new int[rows, width];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < width; c++) prompt[r, c] = values[r * width + c];
            }
            return prompt;
        }

        public void Dispose() => _loader.Dispose();
    }
}
