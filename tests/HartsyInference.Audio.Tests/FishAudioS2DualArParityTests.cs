using System.Text.Json;
using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The tiny checkpoint and reference logits come from the official fish-speech <c>DualARTransformer</c>
/// (<c>tools/fish_audio/s2_dual_ar_reference.py</c>, float32 rope tables): the port must reproduce its slow logits,
/// post-norm hidden states and fast-decoder logits.</summary>
public sealed unsafe class FishAudioS2DualArParityTests
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "FishAudioS2");

    private static FishAudioS2Config TinyConfig() => new()
    {
        Slow = new FishAudioTransformerConfig
        {
            HiddenSize = 32, NumHiddenLayers = 2, NumAttentionHeads = 4, NumKeyValueHeads = 2, IntermediateSize = 48,
            VocabSize = 160, MaxPositionEmbeddings = 32, HeadDim = 8, RopeTheta = 1_000_000f, RmsNormEps = 1e-6f,
        },
        Fast = new FishAudioTransformerConfig
        {
            HiddenSize = 32, NumHiddenLayers = 2, NumAttentionHeads = 4, NumKeyValueHeads = 2, IntermediateSize = 48,
            VocabSize = 16, MaxPositionEmbeddings = 5, HeadDim = 8, RopeTheta = 1_000_000f, RmsNormEps = 1e-6f, QkNorm = false,
        },
        NumCodebooks = 4, CodebookSize = 16, SemanticBeginId = 100, SemanticEndId = 115,
    };

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    private static void AssertClose(float[] expected, float[] actual, float tol, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
        Assert.True(worst <= tol, $"{what}: max |Δ| = {worst}");
    }

    [Fact]
    public void SlowLogitsHiddenAndFastLogits_MatchOfficialImplementation()
    {
        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(Dir, "s2_tiny.safetensors"));
        Dictionary<string, Tensor> weights = new();
        foreach (string name in loader.Descriptors.Keys) weights[name] = loader.GetTensor(name);

        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "s2_tiny.json")));
        JsonElement root = doc.RootElement;
        int[] tokens = root.GetProperty("tokens").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        int[][] codeRows = root.GetProperty("codes").EnumerateArray()
            .Select(r => r.EnumerateArray().Select(e => e.GetInt32()).ToArray()).ToArray();   // [codebook][frame]
        int[] fastPrev = [root.GetProperty("last_code0").GetInt32(), .. root.GetProperty("fast_prev").EnumerateArray().Select(e => e.GetInt32())];

        // Positions 0..2 are text; 3..6 carry codebook columns.
        int[]?[] codes = new int[]?[tokens.Length];
        for (int f = 0; f < 4; f++) codes[3 + f] = codeRows.Select(row => row[f]).ToArray();

        using IBackend backend = new CpuBackend();
        using FishAudioS2DualAr model = new(TinyConfig());
        model.LoadWeights(weights);

        (float[] logits, float[] hidden) = model.DebugSlow(backend, tokens, codes);
        AssertClose(Read(weights["ref.slow_logits"]), logits, 1e-4f, "slow logits");
        AssertClose(Read(weights["ref.hidden"]), hidden, 1e-4f, "post-norm hidden");

        float[] lastHidden = hidden[^32..];
        AssertClose(Read(weights["ref.fast_logits"]), model.DebugFastLogits(backend, lastHidden, fastPrev), 1e-4f, "fast logits");
    }
}
