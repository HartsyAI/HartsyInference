using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Duo;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Duo;

/// <summary>L1 parity: the Duo DiT forward on the H3 stack against upstream inclusionAI/Duo's DuoTransformer on one tiny layout with seeded
/// weights. The fixture is written by <c>tests/python-reference/duo_l1_oracle.py</c> at the commit it records. The forward is the causal,
/// masked one (ForwardDuo), so this covers the mask, the refiner, the AdaLN tables, the RoPE and the output heads together.</summary>
public sealed class DuoDitParityTests
{
    private const string ExpectedFile = "Fixtures/Duo/duo_l1_expected.json";

    [Fact]
    public void FixtureIsTheUpstreamRecord()
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(ExpectedFile));
        Assert.Equal("inclusionAI/Duo", doc.RootElement.GetProperty("source").GetString());
        Assert.Equal("068e357ede2bdea233b408645301e522f560da73", doc.RootElement.GetProperty("commit").GetString());
    }

    [Fact]
    public void CausalDitForwardMatchesUpstreamOnTheTinyLayout()
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(ExpectedFile));
        JsonElement root = doc.RootElement;
        JsonElement config = root.GetProperty("config");
        using MiniMaxH3Transformer transformer = new(new MiniMaxH3Config
        {
            HiddenSize = config.GetProperty("hiddenSize").GetInt32(),
            NumLayers = config.GetProperty("numLayers").GetInt32(),
            TokenRefinerNumLayers = config.GetProperty("tokenRefinerNumLayers").GetInt32(),
            NumAttentionHeads = config.GetProperty("numAttentionHeads").GetInt32(),
            AttentionHeadDim = config.GetProperty("attentionHeadDim").GetInt32(),
            FfnHiddenSize = config.GetProperty("ffnHiddenSize").GetInt32(),
            TextDim = config.GetProperty("textDim").GetInt32(),
            TimeEmbedHiddenSize = config.GetProperty("timeEmbedHiddenSize").GetInt32(),
            TimeEmbedDim = config.GetProperty("timeEmbedDim").GetInt32(),
            NormEps = config.GetProperty("normEps").GetSingle(),
            QkNormEps = config.GetProperty("qkNormEps").GetSingle(),
            FinalNormEps = config.GetProperty("finalNormEps").GetSingle(),
            RopeInvFreqLen = config.GetProperty("ropeInvFreqLen").GetInt32(),
            HasAudio = true,
            LatentsDim = 24,
            OutputHeadCount = 4,
        });

        Dictionary<string, Tensor> weights = new();
        foreach (JsonProperty entry in root.GetProperty("weights").EnumerateObject())
        {
            weights[entry.Name] = Tensor(entry.Value);
        }
        transformer.LoadWeights(weights);

        DuoLayout layout = Layout(root.GetProperty("layout"));
        float[] times = root.GetProperty("times").EnumerateArray().Select(e => e.GetSingle()).ToArray();
        using Tensor text = Tensor(root.GetProperty("textStates"));
        using Tensor video = Tensor(root.GetProperty("video"));
        IReadOnlyList<DuoGroup> refiner = [new DuoGroup(0, 0, DuoGroupKind.Text, DuoModality.None, 0, 2)];

        using IBackend backend = new CpuBackend();
        (Tensor outVideo, Tensor? outAudio) = transformer.ForwardDuo(backend, layout, DuoAttentionMode.Causal, refiner, text, video, null, times);
        float[] actual = outVideo.AsSpan<float>().ToArray();
        outVideo.Dispose();
        outAudio?.Dispose();

        JsonElement expected = root.GetProperty("expectedVideo");
        float[] want = expected.GetProperty("data").EnumerateArray().Select(e => e.GetSingle()).ToArray();
        Assert.Equal(want.Length, actual.Length);
        float worst = 0f;
        for (int i = 0; i < want.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(actual[i] - want[i]));
        }
        Assert.True(worst <= 1e-4f, $"Duo DiT forward differs from upstream by {worst}.");
    }

    private static DuoLayout Layout(JsonElement layout)
    {
        List<DuoGroup> groups = [];
        List<int> noisy = [];
        foreach (JsonElement g in layout.GetProperty("groups").EnumerateArray())
        {
            string kind = g.GetProperty("kind").GetString()!;
            DuoGroupKind groupKind = kind switch
            {
                "reference" => DuoGroupKind.Reference,
                "shared" => DuoGroupKind.Shared,
                "noisy" => DuoGroupKind.Noisy,
                "text" => DuoGroupKind.Text,
                _ => throw new InvalidOperationException($"Unknown upstream group kind '{kind}'."),
            };
            DuoModality modality = groupKind is DuoGroupKind.Reference or DuoGroupKind.Noisy ? DuoModality.Video : DuoModality.None;
            if (groupKind == DuoGroupKind.Noisy)
            {
                noisy.Add(groups.Count);
            }
            groups.Add(new DuoGroup(g.GetProperty("sample").GetInt32(), g.GetProperty("block").GetInt32(), groupKind, modality,
                g.GetProperty("start").GetInt32(), g.GetProperty("end").GetInt32()));
        }
        int length = layout.GetProperty("tags").GetArrayLength();
        return new DuoLayout(groups, length, length,
            layout.GetProperty("positions").EnumerateArray().Select(e => e.GetDouble()).ToArray(),
            layout.GetProperty("tags").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            layout.GetProperty("timeIds").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            layout.GetProperty("videoRows").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            layout.GetProperty("audioRows").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            layout.GetProperty("textRows").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            layout.GetProperty("textSourceRows").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            noisy);
    }

    /// <summary>A tensor from the fixture's <c>{shape, data}</c> form, F32.</summary>
    private static Tensor Tensor(JsonElement packed)
    {
        int[] shape = packed.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        float[] data = packed.GetProperty("data").EnumerateArray().Select(e => e.GetSingle()).ToArray();
        Tensor t = new(new TensorShape(shape.Select(d => (long)d).ToArray()), DType.F32);
        data.AsSpan().CopyTo(t.AsSpan<float>());
        return t;
    }
}
