using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.Onnx;
using Xunit;

namespace HartsyInference.ModelHandler.Tests;

/// <summary>Verifies the hand-rolled ONNX weight loader against the real CosyVoice2 <c>campplus.onnx</c>:
/// initializer count, shapes, and raw values must round-trip exactly (golden values from the <c>onnx</c>
/// reference reader). Gated on <c>HARTSYINFERENCE_CAMPPLUS_ONNX</c>; skips cleanly when absent.</summary>
public sealed unsafe class OnnxWeightLoaderTests
{
    [Fact]
    public void Load_CampPlus_MatchesOnnxReference()
    {
        string? path = Environment.GetEnvironmentVariable("HARTSYINFERENCE_CAMPPLUS_ONNX");
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;

        using OnnxWeightLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> w = loader.GetAllTensors();

        // All 617 initializers are FLOAT, so every one materializes.
        Assert.Equal(617, w.Count);
        Assert.NotNull(loader.Model);
        Assert.Equal(3206, loader.Model!.Nodes.Count);     // node list parsed in graph order

        AssertTensor(w, "xvector.block1.tdnnd1.cam_layer.linear1.weight", [64, 128, 1],
            [0.001624f, -0.015454f, -0.004706f], 0.09743f);
        AssertTensor(w, "xvector.block1.tdnnd1.nonlinear1.batchnorm.running_mean", [128],
            [0.233154f, 0.237458f, 0.223026f], 32.11778f);
        AssertTensor(w, "xvector.block1.tdnnd1.cam_layer.linear2.bias", [32],
            [-0.037294f, -0.159337f, 0.075086f], -2.46712f);
    }

    private static void AssertTensor(Dictionary<string, Tensor> w, string name, int[] dims,
        float[] first3, float sum)
    {
        Assert.True(w.TryGetValue(name, out Tensor? t), $"missing initializer {name}");
        Assert.Equal(dims.Length, t!.Shape.Rank);
        for (int i = 0; i < dims.Length; i++) Assert.Equal(dims[i], (int)t.Shape[i]);

        float* p = (float*)t.DataPointer;
        for (int i = 0; i < first3.Length; i++) Assert.Equal(first3[i], p[i], precision: 5);
        float total = 0f;
        for (long i = 0; i < t.ElementCount; i++) total += p[i];
        Assert.Equal(sum, total, precision: 3);
    }
}
