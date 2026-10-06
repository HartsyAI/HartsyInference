using System.Text.Json;
using HartsyInference.Audio.Models.BreezeTts;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>A tiny random Breeze backbone + depth decoder run through the official classes
/// (<c>tools/breeze/backbone_depth_reference.py</c>) must match the port: frame embeddings, backbone hidden state and
/// <c>lm_head</c> logits over a mixed text/audio prompt, and the depth decoder's per-codebook logits.</summary>
public sealed unsafe class BreezeTts2ModelParityTests
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Breeze");

    private static BreezeTts2Config Tiny() => new()
    {
        HiddenSize = 32, NumBackboneLayers = 2, NumBackboneHeads = 4, NumBackboneKeyValueHeads = 2, BackboneHeadDim = 8,
        BackboneIntermediateSize = 48, NumDepthDecoderLayers = 2, DepthDecoderHiddenSize = 16,
        DepthDecoderIntermediateSize = 24, DepthDecoderHeads = 2, DepthDecoderKeyValueHeads = 1, DepthDecoderHeadDim = 8,
        DepthDecoderMaxPositions = 5, NumCodebooks = 4, AudioVocabSize = 19, CodebookSize = 16,
    };

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    private static void AssertClose(float[] want, float[] got, string what)
    {
        Assert.Equal(want.Length, got.Length);
        float worst = want.Zip(got, (a, b) => MathF.Abs(a - b)).Max();
        Assert.True(worst <= 2e-4f * MathF.Max(1f, want.Max(MathF.Abs)), $"{what}: max |Δ| = {worst}");
    }

    [Fact]
    public void BackboneAndDepthDecoder_MatchOfficialImplementation()
    {
        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(Dir, "breeze_core_tiny.safetensors"));
        Dictionary<string, Tensor> weights = new();
        foreach (string name in loader.Descriptors.Keys) weights[name] = loader.GetTensor(name);
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "breeze_core_tiny.json")));
        int[][] frames = doc.RootElement.GetProperty("codes").EnumerateArray()
            .Select(f => f.EnumerateArray().Select(e => e.GetInt32()).ToArray()).ToArray();
        int[] depthTokens = doc.RootElement.GetProperty("depth_tokens").EnumerateArray().Select(e => e.GetInt32()).ToArray();

        using IBackend backend = new CpuBackend();
        using BreezeTts2Model model = new(Tiny());
        model.LoadWeights(weights);

        // frame embeddings
        float[] wantAudio = Read(weights["ref.audio_embeds"]);
        float[] audio = frames.SelectMany(f => model.EmbedFrame(f)).ToArray();
        AssertClose(wantAudio, audio, "audio frame embeddings");

        // mixed prompt: 3 text embeddings then the 5 frames
        float[] text = Read(weights["ref.text"]);
        float[] embeds = [.. text, .. audio];
        using IKvCache cache = model.CreateBackboneCache(16);
        float[] hidden = model.BackboneStep(backend, embeds, 8, 0, cache);
        float[] wantHidden = Read(weights["ref.hidden"]);
        AssertClose(wantHidden[(7 * 32)..], hidden, "backbone last hidden state");
        AssertClose(Read(weights["ref.logits"]), model.BackboneLogits(backend, hidden), "lm_head logits");

        // depth decoder logits for [placeholder, t0, t1, t2]
        AssertClose(Read(weights["ref.depth_logits"]), model.DebugDepthLogits(backend, hidden, depthTokens), "depth decoder logits");
    }
}
