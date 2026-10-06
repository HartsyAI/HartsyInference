using System.Text.Json;
using HartsyInference.Audio.Models.BreezeTts;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>A tiny random T5Gemma2 encoder run through the official Breeze compat implementation
/// (<c>tools/breeze/text_encoder_reference.py</c>) must match the port stage by stage — covering the symmetric sliding
/// window, the linearly scaled global RoPE, the (1+w) norms and the EOI embedding.</summary>
public sealed unsafe class T5Gemma2TextEncoderParityTests
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Breeze");

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    [Fact]
    public void Encode_MatchesOfficialImplementation()
    {
        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(Dir, "t5gemma2_tiny.safetensors"));
        Dictionary<string, Tensor> weights = new();
        foreach (string name in loader.Descriptors.Keys) weights[name] = loader.GetTensor(name);
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "t5gemma2_tiny.json")));
        int[] tokens = doc.RootElement.GetProperty("tokens").EnumerateArray().Select(e => e.GetInt32()).ToArray();

        T5Gemma2TextEncoderConfig cfg = new()
        {
            VocabSize = 64, HiddenSize = 32, IntermediateSize = 48, NumLayers = 7, NumHeads = 4, NumKvHeads = 1, HeadDim = 8,
            QueryPreAttnScalar = 8, SlidingWindow = 8, EoiTokenIndex = 60,
        };
        using IBackend backend = new CpuBackend();
        using T5Gemma2TextEncoder encoder = new(cfg);
        encoder.LoadWeights(weights);
        Dictionary<string, float[]> taps = new();
        float[] last = encoder.Encode(backend, tokens, (name, v) => taps[name] = v);

        foreach (KeyValuePair<string, float[]> tap in taps)
        {
            string key = tap.Key == "embed" ? "ref.embed" : $"ref.{tap.Key}";
            if (!weights.ContainsKey(key)) continue;
            float[] want = Read(weights[key]);
            float worst = want.Zip(tap.Value, (a, b) => MathF.Abs(a - b)).Max();
            Assert.True(worst <= 1e-4f * MathF.Max(1f, want.Max(MathF.Abs)), $"{tap.Key}: max |Δ| = {worst}");
        }
        float[] wantLast = Read(weights["ref.last"]);
        float finalWorst = wantLast.Zip(last, (a, b) => MathF.Abs(a - b)).Max();
        Assert.True(finalWorst <= 1e-4f * MathF.Max(1f, wantLast.Max(MathF.Abs)), $"final: max |Δ| = {finalWorst}");
    }
}
