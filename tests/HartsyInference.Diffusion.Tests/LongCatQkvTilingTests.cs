using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The self-attention QKV projection run over row tiles matches the untiled projection, on both the uncached forward and the cached
/// denoising forward: a tile height that does not divide the token count leaves a short last tile at every seam.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatQkvTilingTests : IDisposable
{
    private readonly LongCatConfig _config = LongCatTestData.Tiny;
    private readonly Dictionary<string, Tensor> _weights;
    private readonly CpuBackend _backend = new();

    public LongCatQkvTilingTests()
    {
        _weights = LongCatTestData.Weights(_config, 53);
    }

    public void Dispose()
    {
        LongCatTestData.DisposeAll(_weights);
        _backend.Dispose();
    }

    [Fact]
    public void ATiledQkvMatchesTheUntiledForwardAndTheCachedForward()
    {
        using Tensor latent = LongCatTestData.Latent(_config, 5, 4, 4, 2);
        using Tensor cond = LongCatTestData.Latent(_config, 2, 4, 4, 3);
        using Tensor noisy = LongCatTestData.Latent(_config, 3, 4, 4, 4);
        using Tensor text = LongCatTestData.Encoder(_config, 5, 5);
        LongCatConditioning conditioning = new(NumCondLatents: 1);
        LongCatConditioning cachedConditioning = new(NumCondLatents: 2);

        // 20 tokens uncached (5 frames × 4), 12 noisy and 8 conditioning tokens cached: tiles of 3 leave a short last tile on each.
        float[] untiledUncached = Forward(null, latent, text, conditioning, tileRows: int.MaxValue);
        float[] tiledUncached = Forward(null, latent, text, conditioning, tileRows: 3);
        AssertClose(untiledUncached, tiledUncached, "uncached");

        float[] untiledCached = Forward(cond, noisy, text, cachedConditioning, tileRows: int.MaxValue);
        float[] tiledCached = Forward(cond, noisy, text, cachedConditioning, tileRows: 3);
        AssertClose(untiledCached, tiledCached, "cached");
    }

    private float[] Forward(Tensor? cond, Tensor latent, Tensor text, LongCatConditioning conditioning, int tileRows)
    {
        using LongCatTransformer transformer = new(_config);
        transformer.LoadWeights(_weights);
        for (int b = 0; b < transformer.BlockCount; b++)
            ((LongCatBlock)transformer.GetBlock(b)).SetQkvTileRows(tileRows);
        if (cond is null)
        {
            using Tensor uncached = transformer.Forward(_backend, latent, text, conditioning.FrameTimesteps(300f, 5), in conditioning);
            return uncached.AsReadOnlySpan<float>().ToArray();
        }

        using LongCatKvCache cache = transformer.BuildKvCache(_backend, cond, new LongCatConditioning(NumCondLatents: 2));
        using Tensor cached = transformer.ForwardCached(_backend, latent, text, conditioning.FrameTimesteps(300f, 3), in conditioning, cache);
        return cached.AsReadOnlySpan<float>().ToArray();
    }

    private static void AssertClose(float[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
            worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]));
        Assert.True(worst <= 1e-5, $"{label}: tiled QKV drifted from the untiled projection by {worst}");
    }
}
