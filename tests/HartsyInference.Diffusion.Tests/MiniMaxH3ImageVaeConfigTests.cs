using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Image-VAE config detection on tiny fixtures. Width 64 keeps the heads and FFN small while preserving the
/// decoder's key layout; the 2D flag and patch size come from the shapes, the head and rope constants from the baseline.</summary>
public sealed class MiniMaxH3ImageVaeConfigTests
{
    private const int Width = 64;

    [Fact]
    public void SingleFrameDecoderIsDetectedFromRankFourPostQuantConv()
    {
        Dictionary<string, Tensor> weights = Fixture(postQuantRank: 4, projOutRows: 768);

        MiniMaxH3VideoVaeConfig config = MiniMaxH3VideoVaeConfig.Detect(weights);

        Assert.True(config.Decoder2D);
        Assert.Equal(768, config.PatchDim);
        Assert.Equal(2, config.RopeAxes);
        Assert.Equal(8, config.RopeInvFreqLen);
        Assert.Equal(1, config.Heads);
        Assert.Equal(1, config.NumLayers);
        Assert.Equal(24, config.ZChannels);
        Assert.Equal(4, config.FfnMult);
    }

    [Fact]
    public void VideoDecoderKeepsTheThreeDimensionalGeometry()
    {
        Dictionary<string, Tensor> weights = Fixture(postQuantRank: 5, projOutRows: 3072);

        MiniMaxH3VideoVaeConfig config = MiniMaxH3VideoVaeConfig.Detect(weights);

        Assert.False(config.Decoder2D);
        Assert.Equal(3072, config.PatchDim);
        Assert.Equal(3, config.RopeAxes);
        Assert.Equal(8, config.RopeInvFreqLen);
    }

    [Fact]
    public void ImageBaselineSuppliesTheCheckpointConstants()
    {
        MiniMaxH3VideoVaeConfig baseline = MiniMaxH3VideoVaeConfig.ImageBaseline;

        Assert.True(baseline.Decoder2D);
        Assert.Equal(16, baseline.Heads);
        Assert.Equal(64, baseline.DimHead);
        Assert.Equal(24, baseline.NumLayers);
        Assert.Equal(0.5f, baseline.RopeDimRatio);
        Assert.Equal(100.0, baseline.RopeTheta);
        Assert.Equal(0, baseline.TokenDrop);
    }

    [Fact]
    public void TwoDimensionalSignatureComesFromShapesOnly()
    {
        Assert.True(MiniMaxH3VideoVaeConfig.IsTwoDimensional(Fixture(postQuantRank: 4, projOutRows: 768)));
        Assert.False(MiniMaxH3VideoVaeConfig.IsTwoDimensional(Fixture(postQuantRank: 5, projOutRows: 3072)));
    }

    [Fact]
    public void DecoderAcceptsTheSingleFrameConfig()
    {
        MiniMaxH3VideoVaeConfig config = MiniMaxH3VideoVaeConfig.Detect(Fixture(postQuantRank: 4, projOutRows: 768));

        using MiniMaxH3VideoVaeDecoder decoder = new(config);

        Assert.Equal(1, config.TemporalPatch);
    }

    /// <summary>Minimal decoder with one block of width <see cref="Width"/> and the key names the detector reads.</summary>
    private static Dictionary<string, Tensor> Fixture(int postQuantRank, int projOutRows)
    {
        Tensor postQuant = postQuantRank == 4
            ? new Tensor(new TensorShape(new long[] { 24, 24, 1, 1 }), DType.F32)
            : new Tensor(new TensorShape(new long[] { 24, 24, 1, 1, 1 }), DType.F32);
        return new Dictionary<string, Tensor>
        {
            ["post_quant_conv.weight"] = postQuant,
            ["decoder.x_embedder.weight"] = new Tensor(new TensorShape(Width, 24), DType.F32),
            ["decoder.proj_out.weight"] = new Tensor(new TensorShape(projOutRows, Width), DType.F32),
            ["decoder.transformer_blocks.0.attn.to_qkv.weight"] = new Tensor(new TensorShape(3 * Width, Width), DType.F32),
            ["decoder.transformer_blocks.0.ff.w1.weight"] = new Tensor(new TensorShape(2 * Width * 4, Width), DType.F32),
            ["decoder.register_tokens"] = new Tensor(new TensorShape(1, 4, Width), DType.F32),
        };
    }
}
