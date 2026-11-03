using HartsyInference.Core.Tensors;
using HartsyInference.Vision.DepthAnything;
using HartsyInference.Vision.Dinov2;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary><see cref="DepthAnythingPreset.FromWeights"/> and <see cref="Dinov2Preset.FromWeights"/> pick the variant a
/// checkpoint was trained as from its tensor shapes. VisionService used to build the Small preset for any file, so a Base
/// or Large checkpoint passed by path failed to load or ran with the wrong head. Only the probed keys are synthesized.</summary>
public sealed class VisionPresetFromWeightsTests
{
    [Theory]
    [InlineData(384, 12, "depth-anything/Depth-Anything-V2-Small")]
    [InlineData(768, 12, "depth-anything/Depth-Anything-V2-Base")]
    [InlineData(1024, 24, "depth-anything/Depth-Anything-V2-Large")]
    public void DepthAnything_PicksTheReleasedVariantFromTheBackbone(int hidden, int blocks, string expected)
    {
        using WeightSet w = new WeightSet();
        w.Add("pretrained.cls_token", 1, 1, hidden);
        for (int i = 0; i < blocks; i++)
            w.Add($"pretrained.blocks.{i}.norm1.weight", hidden);

        Assert.Equal(expected, DepthAnythingPreset.FromWeights(w.Tensors).Name);
    }

    [Fact]
    public void DepthAnything_RefusesAWidthAndDepthNoVariantHas()
    {
        using WeightSet w = new WeightSet();
        w.Add("pretrained.cls_token", 1, 1, 1024);
        for (int i = 0; i < 12; i++)
            w.Add($"pretrained.blocks.{i}.norm1.weight", 1024);

        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => DepthAnythingPreset.FromWeights(w.Tensors));
        Assert.Contains("1024-wide, 12-block", ex.Message);
    }

    [Fact]
    public void DepthAnything_RefusesAFileWithoutTheBackbone()
    {
        using WeightSet w = new WeightSet();
        w.Add("depth_head.projects.0.weight", 48, 384, 1, 1);

        Assert.Throws<NotSupportedException>(() => DepthAnythingPreset.FromWeights(w.Tensors));
    }

    [Theory]
    [InlineData(384, 12, 0, "facebook/dinov2-small")]
    [InlineData(768, 12, 0, "facebook/dinov2-base")]
    [InlineData(1024, 24, 0, "facebook/dinov2-large")]
    [InlineData(1024, 24, 4, "dinov2_vitl14_reg")]
    [InlineData(1536, 40, 0, "facebook/dinov2-giant")]
    public void Dinov2_PicksTheVariantFromWidthLayersAndRegisters(int hidden, int layers, int registers, string expected)
    {
        using WeightSet w = Dinov2Weights(hidden, layers, registers, patch: 14);

        Assert.Equal(expected, Dinov2Preset.FromWeights(w.Tensors).Name);
    }

    [Fact]
    public void Dinov2_ReadsKeysUnderAPrefix()
    {
        using WeightSet w = Dinov2Weights(768, 12, 0, patch: 14, prefix: "image_encoder.");

        Assert.Equal("facebook/dinov2-base", Dinov2Preset.FromWeights(w.Tensors, "image_encoder").Name);
    }

    [Fact]
    public void Dinov2_RefusesDinov3PatchSize()
    {
        using WeightSet w = Dinov2Weights(384, 12, 4, patch: 16);

        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => Dinov2Preset.FromWeights(w.Tensors));
        Assert.Contains("DINOv3", ex.Message);
    }

    [Fact]
    public void Dinov2_RefusesAShapeNoVariantHas()
    {
        using WeightSet w = Dinov2Weights(384, 24, 0, patch: 14);

        Assert.Throws<NotSupportedException>(() => Dinov2Preset.FromWeights(w.Tensors));
    }

    private static WeightSet Dinov2Weights(int hidden, int layers, int registers, int patch, string prefix = "")
    {
        WeightSet w = new WeightSet();
        w.Add($"{prefix}embeddings.cls_token", 1, 1, hidden);
        w.Add($"{prefix}embeddings.patch_embeddings.projection.weight", hidden, 3, patch, patch);
        if (registers > 0)
            w.Add($"{prefix}embeddings.register_tokens", 1, registers, hidden);
        for (int i = 0; i < layers; i++)
            w.Add($"{prefix}encoder.layer.{i}.norm1.weight", hidden);
        return w;
    }

    /// <summary>Uninitialized tensors of the given shapes; FromWeights reads shapes only.</summary>
    private sealed class WeightSet : IDisposable
    {
        public Dictionary<string, Tensor> Tensors { get; } = new(StringComparer.Ordinal);

        public void Add(string key, params long[] dims) => Tensors[key] = new Tensor(new TensorShape(dims), DType.F32);

        public void Dispose()
        {
            foreach (Tensor t in Tensors.Values)
                t.Dispose();
        }
    }
}
