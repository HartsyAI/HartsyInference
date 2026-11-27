using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The FFN run over row tiles matches the untiled FFN: the seams between tiles (a tile height that does not divide the token count, so
/// the last tile is short) change nothing beyond float rounding.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatFfnTilingTests : IDisposable
{
    private readonly LongCatConfig _config = LongCatTestData.Tiny;
    private readonly Dictionary<string, Tensor> _weights;
    private readonly CpuBackend _backend = new();

    public LongCatFfnTilingTests()
    {
        _weights = LongCatTestData.Weights(_config, 47);
    }

    public void Dispose()
    {
        LongCatTestData.DisposeAll(_weights);
        _backend.Dispose();
    }

    [Fact]
    public void ATiledFfnMatchesTheUntiledFfnWithAShortLastTile()
    {
        using Tensor latent = LongCatTestData.Latent(_config, 5, 4, 4, 2);
        using Tensor text = LongCatTestData.Encoder(_config, 5, 3);
        LongCatConditioning conditioning = new(NumCondLatents: 1);

        float[] untiled = Forward(latent, text, conditioning, tileRows: int.MaxValue);
        // 20 tokens (5 frames × 4 per frame) in tiles of 3: six full tiles and a tile of 2.
        float[] tiled = Forward(latent, text, conditioning, tileRows: 3);

        Assert.Equal(untiled.Length, tiled.Length);
        double worst = 0;
        for (int i = 0; i < untiled.Length; i++)
            worst = Math.Max(worst, Math.Abs(untiled[i] - tiled[i]));
        Assert.True(worst <= 1e-5, $"tiled FFN drifted from the untiled one by {worst}");
    }

    private float[] Forward(Tensor latent, Tensor text, LongCatConditioning conditioning, int tileRows)
    {
        using LongCatTransformer transformer = new(_config);
        transformer.LoadWeights(_weights);
        for (int b = 0; b < transformer.BlockCount; b++)
            ((LongCatBlock)transformer.GetBlock(b)).SetFfnTileRows(tileRows);
        using Tensor velocity = transformer.Forward(_backend, latent, text, conditioning.FrameTimesteps(300f, 5), in conditioning);
        return velocity.AsReadOnlySpan<float>().ToArray();
    }
}
