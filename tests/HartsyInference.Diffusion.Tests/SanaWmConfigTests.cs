using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>SANA-WM config and layout checks against the real DiT header fixture (names and shapes, no weights).</summary>
public sealed class SanaWmConfigTests
{
    [Fact]
    public void FromWeights_RealHeader_DerivesDitDimensions()
    {
        SanaWmConfig config = SanaWmConfig.FromWeights(SanaWmHeaderFixture.LoadDescriptors());
        AssertRealDitDimensions(config);
    }

    [Fact]
    public void FromWeights_LoadedTensors_DerivesTheSameDimensionsFromShapesAlone()
    {
        Dictionary<string, SafeTensorDescriptor> header = SanaWmHeaderFixture.LoadDescriptors();
        Dictionary<string, Tensor> weights = new(header.Count, StringComparer.Ordinal);
        try
        {
            // Tensors allocate host memory lazily, so reading only shapes costs nothing here.
            foreach (KeyValuePair<string, SafeTensorDescriptor> entry in header)
                weights[entry.Key] = new Tensor(entry.Value.Shape, entry.Value.DType);

            AssertRealDitDimensions(SanaWmConfig.FromWeights(weights));
        }
        finally
        {
            foreach (Tensor tensor in weights.Values)
                tensor.Dispose();
        }
    }

    [Fact]
    public void ValidateKeys_RealHeader_ReportsNoMissingOrUnexpectedKeys()
    {
        Dictionary<string, SafeTensorDescriptor> header = SanaWmHeaderFixture.LoadDescriptors();
        SanaWmConfig config = SanaWmConfig.FromWeights(header);

        Assert.Equal(872, header.Count);
        Assert.Equal(872, SanaWmCheckpointConverter.ExpectedTensors(config).Count);
        Assert.Empty(SanaWmCheckpointConverter.ValidateKeys(config, header));
    }

    [Fact]
    public void ExpectedTensors_RealHeader_StoresEachTensorAsTheDtypePolicySays()
    {
        SanaWmConfig config = SanaWmConfig.FromWeights(SanaWmHeaderFixture.LoadDescriptors());
        Dictionary<string, DType> stored = SanaWmCheckpointConverter.ExpectedTensors(config)
            .ToDictionary(spec => spec.Name, spec => spec.DType, StringComparer.Ordinal);

        Assert.Equal(DType.BF16, stored["blocks.0.attn.qkv.weight"]);
        Assert.Equal(DType.BF16, stored["blocks.0.attn.output_gate.weight"]);
        Assert.Equal(DType.BF16, stored["blocks.0.attn.conv_k.weight"]);
        Assert.Equal(DType.BF16, stored["blocks.0.mlp.point_conv.conv.weight"]);
        Assert.Equal(DType.BF16, stored["y_embedder.y_embedding"]);
        Assert.Equal(DType.F32, stored["blocks.0.attn.A_log"]);
        Assert.Equal(DType.F32, stored["blocks.0.attn.beta_proj.weight"]);
        Assert.Equal(DType.F32, stored["blocks.0.attn.gate_proj.weight"]);
        Assert.Equal(DType.F32, stored["blocks.0.attn.recall_gate"]);
        Assert.Equal(DType.F32, stored["blocks.0.attn.q_norm.weight"]);
        Assert.Equal(DType.F32, stored["blocks.0.attn.q_norm_cam.weight"]);
        Assert.Equal(DType.F32, stored["blocks.0.attn.proj.bias"]);
        Assert.Equal(DType.F32, stored["blocks.0.scale_shift_table"]);
        Assert.Equal(DType.F32, stored["t_block.1.weight"]);
        Assert.Equal(DType.F32, stored["pos_embed"]);
    }

    [Fact]
    public void FromWeights_PartialCameraBranch_Throws()
    {
        Dictionary<string, SafeTensorDescriptor> header = SanaWmHeaderFixture.LoadDescriptors();
        header.Remove("blocks.1.attn.q_proj_cam.weight");

        HartsyInferenceException error =
            Assert.Throws<HartsyInferenceException>(() => SanaWmConfig.FromWeights(header));

        Assert.Contains("camera branch", error.Message, StringComparison.Ordinal);
    }

    /// <summary>Asserts the dimensions of the real 1600M 720p DiT. Shared with the real-weight header check.</summary>
    internal static void AssertRealDitDimensions(SanaWmConfig config)
    {
        Assert.Equal(20, config.Depth);
        Assert.Equal(2240, config.HiddenSize);
        Assert.Equal(20, config.NumHeads);
        Assert.Equal(112, config.HeadDim);
        Assert.Equal(128, config.InChannels);
        Assert.Equal(128, config.OutChannels);
        Assert.Equal(6720, config.FfnHidden);
        Assert.Equal(2304, config.TextWidth);
        Assert.Equal(300, config.NullCaptionLength);
        Assert.Equal(300, config.CaptionMaxLength);
        Assert.Equal(256, config.TimestepWidth);
        Assert.Equal(4, config.ConvKernelSize);
        Assert.Equal(484, config.PosEmbedTokens);
        Assert.Equal(6, config.ModulationRows);
        Assert.Equal(new[] { 3, 7, 11, 15, 19 }, config.SoftmaxBlocks);
        Assert.True(config.IsSoftmaxBlock(3));
        Assert.False(config.IsSoftmaxBlock(4));
        Assert.True(config.HasCameraBranch);
        Assert.True(config.HasPluckerBranch);
        Assert.Equal(48, config.PluckerChannels);
        Assert.True(config.HasRaymapEmbedder);
        Assert.Equal(3, config.RaymapChannels);
    }
}
