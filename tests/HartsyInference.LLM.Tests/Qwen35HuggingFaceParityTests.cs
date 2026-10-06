using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Ssm;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>A tiny random Qwen3.5 trunk run through transformers' <c>Qwen3_5ForConditionalGeneration</c>
/// (<c>tools/clef/backbone_reference.py</c>) must match the port built from the same HuggingFace-keyed weights.</summary>
public sealed unsafe class Qwen35HuggingFaceParityTests
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Clef");

    [Fact]
    public void HiddenStates_MatchTransformers()
    {
        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(Dir, "qwen35_tiny.safetensors"));
        Dictionary<string, Tensor> weights = new();
        foreach (string name in loader.Descriptors.Keys)
        {
            weights[name] = loader.GetTensor(name);
        }
        using SafeTensorsLoader expectedLoader = new();
        expectedLoader.Load(Path.Combine(Dir, "qwen35_tiny_expected.safetensors"));
        Tensor want = expectedLoader.GetTensor("hidden");
        Tensor idTensor = expectedLoader.GetTensor("input_ids");
        int[] ids = new ReadOnlySpan<int>((void*)idTensor.DataPointer, (int)idTensor.ElementCount).ToArray();
        Qwen35HfConfig cfg = Qwen35HfConfig.FromJson(File.ReadAllText(Path.Combine(Dir, "qwen35_tiny_config.json")));

        using CpuBackend backend = new();
        using Qwen35Model model = Qwen35Model.FromHuggingFace(weights, cfg, maxSequenceLength: 64);
        float[] got = model.ForwardHiddenStates(backend, ids);

        ReadOnlySpan<float> expected = new((void*)want.DataPointer, (int)want.ElementCount);
        Assert.Equal(expected.Length, got.Length);
        float worst = 0;
        for (int i = 0; i < got.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(expected[i] - got[i]));
        }
        Assert.True(worst < 2e-4f, $"max |Δ| = {worst}");
    }
}
