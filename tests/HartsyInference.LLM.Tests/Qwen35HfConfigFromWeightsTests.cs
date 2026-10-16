using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Ssm;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary><see cref="Qwen35HfConfig.FromWeights"/> on the tiny transformers trunk of
/// <see cref="Qwen35HuggingFaceParityTests"/>: the shapes alone must give back what its <c>config.json</c> says, and the
/// trunk built from them must still match transformers' hidden states.</summary>
public sealed unsafe class Qwen35HfConfigFromWeightsTests
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Clef");

    [Fact]
    public void ShapeDerivedConfig_AgreesWithConfigJson()
    {
        using SafeTensorsLoader loader = new();
        Dictionary<string, Tensor> weights = Load(loader);
        Qwen35HfConfig json = Qwen35HfConfig.FromJson(File.ReadAllText(Path.Combine(Dir, "qwen35_tiny_config.json")));

        string prefix = Qwen35HfConfig.LanguageModelPrefix(weights);
        Qwen35HfConfig derived = Qwen35HfConfig.FromWeights(weights, prefix);

        Assert.Equal("model.language_model", prefix);
        Assert.Equal(json.HiddenSize, derived.HiddenSize);
        Assert.Equal(json.VocabSize, derived.VocabSize);
        Assert.Equal(json.NumLayers, derived.NumLayers);
        Assert.Equal(json.NumHeads, derived.NumHeads);
        Assert.Equal(json.NumKvHeads, derived.NumKvHeads);
        Assert.Equal(json.HeadDim, derived.HeadDim);
        Assert.Equal(json.LinearConvKernel, derived.LinearConvKernel);
        Assert.Equal(json.LinearKeyHeadDim, derived.LinearKeyHeadDim);
        Assert.Equal(json.LinearValueHeadDim, derived.LinearValueHeadDim);
        Assert.Equal(json.LinearNumKeyHeads, derived.LinearNumKeyHeads);
        Assert.Equal(json.LinearNumValueHeads, derived.LinearNumValueHeads);
        for (int i = 0; i < json.NumLayers; i++) Assert.Equal(json.IsLinearLayer(i), derived.IsLinearLayer(i));
    }

    [Fact]
    public void TrunkFromTheDerivedConfig_MatchesTransformers()
    {
        using SafeTensorsLoader loader = new();
        Dictionary<string, Tensor> weights = Load(loader);
        using SafeTensorsLoader expectedLoader = new();
        expectedLoader.Load(Path.Combine(Dir, "qwen35_tiny_expected.safetensors"));
        Tensor want = expectedLoader.GetTensor("hidden");
        Tensor idTensor = expectedLoader.GetTensor("input_ids");
        int[] ids = new ReadOnlySpan<int>((void*)idTensor.DataPointer, (int)idTensor.ElementCount).ToArray();
        Qwen35HfConfig cfg = Qwen35HfConfig.FromWeights(weights, "model.language_model") with { TieWordEmbeddings = false };

        using CpuBackend backend = new();
        using Qwen35Model model = Qwen35Model.FromHuggingFace(weights, cfg, maxSequenceLength: 64, lmHeadKey: null);
        float[] got = model.ForwardHiddenStates(backend, ids);

        ReadOnlySpan<float> expected = new((void*)want.DataPointer, (int)want.ElementCount);
        float worst = 0;
        for (int i = 0; i < got.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - got[i]));
        Assert.True(worst < 2e-4f, $"max |Δ| = {worst}");
    }

    private static Dictionary<string, Tensor> Load(SafeTensorsLoader loader)
    {
        loader.Load(Path.Combine(Dir, "qwen35_tiny.safetensors"));
        Dictionary<string, Tensor> weights = new();
        foreach (string name in loader.Descriptors.Keys) weights[name] = loader.GetTensor(name);
        return weights;
    }
}
