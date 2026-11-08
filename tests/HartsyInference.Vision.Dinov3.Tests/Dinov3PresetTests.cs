using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;
using HartsyInference.Vision.Dinov3;
using Xunit;

namespace HartsyInference.Vision.Dinov3.Tests;

public sealed class Dinov3PresetTests
{
    [Fact]
    public void V3Large16At448_HasTheHairDeckTokenLayout()
    {
        Dinov2Preset preset = Dinov2Preset.V3Large16 with { ImageSize = 448 };

        Assert.Equal(28, preset.PatchGrid);
        Assert.Equal(784, preset.NumPatches);
        Assert.Equal(4, preset.NumRegisterTokens);
        Assert.Equal(789, preset.SequenceLength);
        Assert.Equal(1024, preset.HiddenSize);
        Assert.Equal(24, preset.NumLayers);
        Assert.Equal(16, preset.NumHeads);
        Assert.Equal(4096, preset.IntermediateSize);
        Assert.Equal(1e-5f, preset.LayerNormEps);
        Assert.True(preset.UsesRotaryPositionEmbedding);
    }

    [Fact]
    public void V3Large16Default_StaysAt224()
    {
        Assert.Equal(224, Dinov2Preset.V3Large16.ImageSize);
        Assert.Equal(201, Dinov2Preset.V3Large16.SequenceLength);
    }

    [Fact]
    public void FromDinov3Weights_ReadsVariantFromShapesAndAppliesSize()
    {
        Dictionary<string, Tensor> weights = Dinov3Fixture.LargeHeaderOnly();
        try
        {
            Dinov2Preset preset = Dinov2Preset.FromDinov3Weights(weights, 448);

            Assert.Equal("facebook/dinov3-vitl16", preset.Name);
            Assert.Equal(448, preset.ImageSize);
            Assert.Equal(789, preset.SequenceLength);
        }
        finally
        {
            Dinov3Fixture.Dispose(weights);
        }
    }

    [Fact]
    public void FromDinov3Weights_RejectsSizesOffThePatchGrid()
    {
        Dictionary<string, Tensor> weights = Dinov3Fixture.LargeHeaderOnly();
        try
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => Dinov2Preset.FromDinov3Weights(weights, 450));
            Assert.Throws<ArgumentOutOfRangeException>(() => Dinov2Preset.FromDinov3Weights(weights, 0));
        }
        finally
        {
            Dinov3Fixture.Dispose(weights);
        }
    }

    [Fact]
    public void FromDinov3Weights_RejectsGatedMlp()
    {
        Dictionary<string, Tensor> weights = Dinov3Fixture.LargeHeaderOnly();
        try
        {
            weights["layer.0.mlp.gate_proj.weight"] = new Tensor(new TensorShape(4096, 1024), DType.F32);
            Assert.Throws<NotSupportedException>(() => Dinov2Preset.FromDinov3Weights(weights, 448));
        }
        finally
        {
            Dinov3Fixture.Dispose(weights);
        }
    }

    [Fact]
    public void FromDinov3Weights_RejectsDinov2Layout()
    {
        Dictionary<string, Tensor> weights = Dinov3Fixture.LargeHeaderOnly();
        try
        {
            Assert.Throws<NotSupportedException>(() => Dinov2Preset.FromDinov3Weights(weights, 448, "model"));
        }
        finally
        {
            Dinov3Fixture.Dispose(weights);
        }
    }

    [Fact]
    public void FromWeights_DoesNotPickUpDinov3Layout()
    {
        Dictionary<string, Tensor> weights = Dinov3Fixture.LargeHeaderOnly();
        try
        {
            // The DINOv2 selector keys on embeddings.patch_embeddings.projection, which a DINOv3 file does not have.
            Assert.Throws<NotSupportedException>(() => Dinov2Preset.FromWeights(weights));
        }
        finally
        {
            Dinov3Fixture.Dispose(weights);
        }
    }

    [Fact]
    public void FromWeights_StillResolvesDinov2Large()
    {
        Dictionary<string, Tensor> weights = Dinov3Fixture.Dinov2LargeHeader();
        try
        {
            Dinov2Preset preset = Dinov2Preset.FromWeights(weights);

            Assert.Equal("facebook/dinov2-large", preset.Name);
            Assert.Equal(518, preset.ImageSize);
            Assert.False(preset.UsesRotaryPositionEmbedding);
            Assert.Equal(1e-6f, preset.LayerNormEps);
        }
        finally
        {
            Dinov3Fixture.Dispose(weights);
        }
    }

    [Fact]
    public void DinoV2Encoder_RejectsDinov3Presets()
    {
        Assert.Throws<ArgumentException>(() => new Dinov2VisionEncoder(Dinov2Preset.V3Large16 with { ImageSize = 448 }));
    }

    [Fact]
    public void Dinov3Encoder_RejectsDinov2Presets()
    {
        Assert.Throws<ArgumentException>(() => new Dinov3VisionEncoder(Dinov2Preset.Large));
    }
}
