using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Unit tests for F-Lite scaffolding — config, RoPE, transformer construction. End-to-end generation requires the actual Freepik/F-Lite checkpoint and is covered separately when assets land.</summary>
public sealed class FLiteTests
{
    [Fact]
    public void Config_V1_Matches_DiffusersJson()
    {
        FLiteConfig c = FLiteConfig.V1;
        Assert.Equal(3072, c.HiddenSize);
        Assert.Equal(40, c.Depth);
        Assert.Equal(12, c.NumHeads);
        Assert.Equal(256, c.HeadDim);
        Assert.Equal(16, c.InChannels);
        Assert.Equal(2, c.PatchSize);
        Assert.Equal(4096, c.CrossAttnInputSize);
        Assert.Equal(10000, c.RopeBase);
        Assert.False(c.TrainBiasAndRms);
        Assert.True(c.ResidualV);
        Assert.True(c.UseRope);
        Assert.Equal(16, c.NumRegisterTokens);
        Assert.Equal(12288, c.MlpDim);
        Assert.Equal(17, c.T5LayerIndex);
    }

    [Fact]
    public void Config_HeadDim_Computed_Correctly()
    {
        Assert.Equal(256, FLiteConfig.V1.HeadDim);
        Assert.Equal(256, FLiteConfig.V1_7B.HeadDim);
    }

    [Fact]
    public unsafe void Rope_Build_ProducesIdentityForRegisterTokens()
    {
        FLiteRope rope = new FLiteRope(headDim: 256, ropeBase: 10000, maxGrid: 32);
        (Tensor cos, Tensor sin) = rope.Build(hPacked: 8, wPacked: 8, numRegisterTokens: 16);
        try
        {
            int halfDim = 128;
            float* cosPtr = (float*)cos.DataPointer;
            float* sinPtr = (float*)sin.DataPointer;

            for (int r = 0; r < 16; r++)
            {
                for (int d = 0; d < halfDim; d++)
                {
                    Assert.Equal(1.0f, cosPtr[r * halfDim + d]);
                    Assert.Equal(0.0f, sinPtr[r * halfDim + d]);
                }
            }
        }
        finally
        {
            cos.Dispose();
            sin.Dispose();
        }
    }

    [Fact]
    public unsafe void Rope_Build_HasFrequencyPatternForImageTokens()
    {
        FLiteRope rope = new FLiteRope(headDim: 256, ropeBase: 10000, maxGrid: 32);
        (Tensor cos, Tensor sin) = rope.Build(hPacked: 4, wPacked: 4, numRegisterTokens: 16);
        try
        {
            int halfDim = 128;
            float* cosPtr = (float*)cos.DataPointer;
            int firstImageRow = 16 * halfDim;
            float baseCos = cosPtr[firstImageRow + 0];
            Assert.True(MathF.Abs(baseCos - 1.0f) < 1e-6f, $"cos at (h=0,w=0,d=0) should be 1.0, got {baseCos}");
        }
        finally
        {
            cos.Dispose();
            sin.Dispose();
        }
    }

    [Fact]
    public void Rope_RejectsOddHeadDim()
    {
        Assert.Throws<ArgumentException>(() => new FLiteRope(headDim: 255));
    }

    [Fact]
    public void Rope_RejectsOversizedGrid()
    {
        FLiteRope rope = new FLiteRope(headDim: 256, ropeBase: 10000, maxGrid: 32);
        Assert.Throws<ArgumentException>(() => rope.Build(hPacked: 64, wPacked: 64, numRegisterTokens: 16));
    }

    [Fact]
    public void Transformer_Construction_DoesNotAllocateWeights()
    {
        FLiteTransformer t = new FLiteTransformer(FLiteConfig.V1);
        Assert.Equal(FLiteConfig.V1, t.Config);
        Assert.Empty(t.EnumerateWeights());
        t.Dispose();
    }

    [Fact]
    public void Transformer_LoadWeights_ThrowsOnMissingRequiredKey()
    {
        FLiteTransformer t = new FLiteTransformer(FLiteConfig.V1);
        try
        {
            Dictionary<string, Tensor> empty = new();
            Assert.Throws<KeyNotFoundException>(() => t.LoadWeights(empty));
        }
        finally
        {
            t.Dispose();
        }
    }
}
