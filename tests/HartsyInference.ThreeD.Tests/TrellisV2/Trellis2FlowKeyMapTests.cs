using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

public sealed class Trellis2FlowKeyMapTests
{
    [Fact]
    public void ShippedConfig_ListsTheCheckpointsSixHundredFortyTensors()
    {
        // 10 model-level tensors and 21 per block: modulation, qkv(2), self gammas(2), out(2), norm2(2), q(2), kv(2), cross gammas(2),
        // out(2), mlp(4).
        Trellis2FlowConfig config = Trellis2FlowConfig.FromJson(Trellis2TestData.ShippedStructureFlowJson);

        IReadOnlyList<ShapeVaeKeyMap.TensorSpec> specs = Trellis2FlowKeyMap.Specs(config);

        Assert.Equal(10 + 21 * 30, specs.Count);
        Assert.Equal(640, specs.Count);
        Assert.Equal(specs.Count, specs.Select(s => s.CheckpointKey).Distinct().Count());
    }

    [Fact]
    public void ShippedConfig_UsesTheDocumentedShapes()
    {
        Trellis2FlowConfig config = Trellis2FlowConfig.FromJson(Trellis2TestData.ShippedStructureFlowJson);
        Dictionary<string, long[]> shapes = Trellis2FlowKeyMap.Specs(config).ToDictionary(s => s.CheckpointKey, s => s.Shape);

        Assert.Equal(new long[] { 9216, 1536 }, shapes["adaLN_modulation.1.weight"]);
        Assert.Equal(new long[] { 9216 }, shapes["blocks.0.modulation"]);
        Assert.Equal(new long[] { 4608, 1536 }, shapes["blocks.29.self_attn.to_qkv.weight"]);
        Assert.Equal(new long[] { 12, 128 }, shapes["blocks.3.self_attn.q_rms_norm.gamma"]);
        Assert.Equal(new long[] { 3072, 1024 }, shapes["blocks.3.cross_attn.to_kv.weight"]);
        Assert.Equal(new long[] { 8192, 1536 }, shapes["blocks.3.mlp.mlp.0.weight"]);
        Assert.Equal(new long[] { 8, 1536 }, shapes["out_layer.weight"]);
        Assert.DoesNotContain("blocks.0.norm1.weight", shapes.Keys);
    }

    [Fact]
    public void TinyConfig_CompleteWeightsLoadAndEveryTensorIsReadable()
    {
        Trellis2FlowConfig config = Trellis2TestData.Tiny(dense: true);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 1);
        using Trellis2FlowWeights weights = Trellis2FlowWeights.FromTensors(config, tensors);

        Assert.Equal(Trellis2FlowKeyMap.Specs(config).Count, weights.Tensors.Count);
        Trellis2TestData.DisposeAll(tensors.Values);
    }

    [Fact]
    public void MissingTensorIsRejectedByName()
    {
        Trellis2FlowConfig config = Trellis2TestData.Tiny(dense: false);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 2);
        tensors.Remove("blocks.1.cross_attn.to_kv.bias");

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => Trellis2FlowWeights.FromTensors(config, tensors));

        Assert.Contains("blocks.1.cross_attn.to_kv.bias", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WrongShapeIsRejectedByName()
    {
        Trellis2FlowConfig config = Trellis2TestData.Tiny(dense: false);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 3);
        tensors["out_layer.weight"] = new Tensor(new TensorShape(5, 16), DType.F32);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => Trellis2FlowWeights.FromTensors(config, tensors));

        Assert.Contains("out_layer.weight", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ExtraTensorIsRejected()
    {
        Trellis2FlowConfig config = Trellis2TestData.Tiny(dense: false);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 4);
        tensors["blocks.0.norm1.weight"] = new Tensor(new TensorShape(16), DType.F32);

        Assert.Throws<HartsyInferenceException>(() => Trellis2FlowWeights.FromTensors(config, tensors));
    }

    [Fact]
    public void Bfloat16TensorsWidenExactlyToF32()
    {
        Trellis2FlowConfig config = Trellis2TestData.Tiny(dense: true);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 5);
        float[] original = Trellis2TestData.ToArray(tensors["blocks.0.modulation"]);
        Tensor bf16 = new(tensors["blocks.0.modulation"].Shape, DType.BF16);
        Span<ushort> bits = bf16.AsSpan<ushort>();
        for (int i = 0; i < bits.Length; i++)
            bits[i] = TensorCasts.F32ToBf16Bits(original[i]);
        tensors["blocks.0.modulation"] = bf16;

        using Trellis2FlowWeights weights = Trellis2FlowWeights.FromTensors(config, tensors);

        Tensor widened = weights.Tensors["blocks.0.modulation"];
        Assert.Equal(DType.F32, widened.DType);
        ReadOnlySpan<float> values = widened.AsReadOnlySpan<float>();
        for (int i = 0; i < values.Length; i++)
            Assert.Equal(BitConverter.UInt32BitsToSingle((uint)TensorCasts.F32ToBf16Bits(original[i]) << 16), values[i]);
    }
}
