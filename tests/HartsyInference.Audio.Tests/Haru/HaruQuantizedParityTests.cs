using System.Text.Json;
using HartsyInference.Audio.Models.Haru;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Haru;

/// <summary>The released Haru-4B at a quantized precision on a backend, replaying the upstream conversation of
/// <c>tools/haru/real_reference.py</c> (same card, voice clip, user recording and recorded random draws). Quantization changes the
/// numbers, so this does not demand identical tokens: it reports how far the text stream follows upstream and asserts the opening of the
/// conversation. <c>HARU_REAL_DIR</c> names the folder of <see cref="HaruRealWeightsTests"/>; <c>HARU_PRECISION</c> (q8, q4, ...) and
/// <c>HARU_BACKEND</c> (cuda, cpu; default cuda) select the run; <c>HARU_CACHE_DIR</c> holds the converted weights.</summary>
[Trait("Category", "Integration")]
public sealed class HaruQuantizedParityTests(ITestOutputHelper output)
{
    [Fact]
    public void Conversation_AtAQuantizedPrecision_FollowsTheUpstreamText()
    {
        string? dir = Environment.GetEnvironmentVariable("HARU_REAL_DIR");
        string? precisionName = Environment.GetEnvironmentVariable("HARU_PRECISION");
        if (string.IsNullOrEmpty(dir) || string.IsNullOrEmpty(precisionName) || !File.Exists(Path.Combine(dir, "real_session_expected.safetensors")))
        {
            Assert.False(Environment.GetEnvironmentVariable("HARTSY_REQUIRE_REAL_WEIGHTS") == "1", "HARU_REAL_DIR and HARU_PRECISION are not set.");
            return;
        }
        Assert.True(HaruWeights.TryParsePrecision(precisionName, out HaruPrecision precision), $"unknown precision {precisionName}");
        string backendName = Environment.GetEnvironmentVariable("HARU_BACKEND") ?? "cuda";
        string cacheDir = Environment.GetEnvironmentVariable("HARU_CACHE_DIR") ?? Path.GetTempPath();

        using IBackend backend = BackendFactory.Create(backendName);
        HaruConfig config = HaruConfig.FromJson(File.ReadAllText(Path.Combine(dir, "config.json")));
        HaruTextCodec text;
        using (FileStream json = File.OpenRead(Path.Combine(dir, "tokenizer.json")))
        {
            text = HaruTextCodec.From(HaruTokenizer.Load(json, config));
        }
        HfQuantCheckpoint checkpoint = HfQuantCheckpoint.OpenFiles([Path.Combine(dir, "model.safetensors")]);
        (IReadOnlyDictionary<string, Tensor> weights, IDisposable owner) = HaruWeights.Load(checkpoint, precision, Path.Combine(cacheDir, $"haru-test-{precision}.gguf"));
        using HaruPipeline pipeline = HaruPipeline.Build(backend, config, text, weights, owner, capacity: 256);

        using SafeTensorsLoader expected = new();
        expected.Load(Path.Combine(dir, "real_session_expected.safetensors"));
        List<float[]> draws = [];
        for (int i = 0; expected.Descriptors.ContainsKey($"run.draw.{i}"); i++)
        {
            draws.Add(HaruReferenceModel.Read(expected.GetTensor($"run.draw.{i}")));
        }
        HaruScriptedNoise noise = new(draws);
        using JsonDocument json2 = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "real_session_expected.json")));
        string card = json2.RootElement.GetProperty("card").GetString()!;
        float[] user = HaruReferenceModel.Read(expected.GetTensor("in.user"));
        float[] voice = HaruReferenceModel.Read(expected.GetTensor("in.voice"));
        int[] wantTokens = HaruReferenceModel.ReadInts(expected.GetTensor("run.tokens"));
        int frame = config.FrameSamples;

        using HaruDuplexSession session = pipeline.OpenSession(new HaruDuplexOptions { Temperature = 0f, Seed = 1 }, noise);
        List<int> tokens = [session.Start(card, voice).Token];
        for (int i = 0; i < wantTokens.Length - 1; i++)
        {
            tokens.Add(session.Step(user.AsSpan(i * frame, frame)).Token);
        }
        int agree = Enumerable.Range(0, wantTokens.Length).Count(i => tokens[i] == wantTokens[i]);
        int firstDivergence = Enumerable.Range(0, wantTokens.Length).FirstOrDefault(i => tokens[i] != wantTokens[i], -1);
        output.WriteLine($"{precision} on {backendName}: {agree}/{wantTokens.Length} tokens agree, first divergence frame {firstDivergence}");
        output.WriteLine($"upstream : {string.Join(' ', wantTokens)}");
        output.WriteLine($"engine   : {string.Join(' ', tokens)}");
        // Turn-taking is the F32 PAD/EOP decision: when the model is silent, starts and stops speaking must match upstream frame for frame,
        // even where a quantized greedy pick of the spoken token itself lands on a different word.
        bool Silent(int token) => token == config.PadTokenId || token == config.EopTokenId;
        Assert.Equal(wantTokens.Select(Silent), tokens.Select(Silent));
        Assert.Equal(wantTokens[..3], tokens.Take(3).ToArray());
    }
}
