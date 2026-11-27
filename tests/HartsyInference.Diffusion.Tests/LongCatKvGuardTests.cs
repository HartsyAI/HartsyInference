using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The input guards of the KV-cache path on the tiny base config: a non-zero mask range is refused, the cache pass needs a tiling grid, and
/// the denoising pass needs one timestep or one per noisy frame. The numerical checks are in <c>LongCatAvatarKvParityTests</c>.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatKvGuardTests
{
    [Fact]
    public void ANonZeroMaskRangeIsRefusedOnTheCachePass()
    {
        LongCatConfig config = LongCatTestData.Tiny;
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, seed: 41);
        try
        {
            using LongCatTransformer transformer = new(config);
            transformer.LoadWeights(weights);
            using Tensor cond = LongCatTestData.Latent(config, frames: 2, height: 4, width: 4, seed: 42);

            using CpuBackend backend = new();
            Assert.Throws<ArgumentException>(() => transformer.BuildKvCache(backend, cond,
                LongCatConditioning.Avatar(2, 10, maskRange: 2)));
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }

    [Fact]
    public void ACacheWithANonTilingGridIsRefused()
    {
        LongCatConfig config = LongCatTestData.Tiny;
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, seed: 43);
        try
        {
            using LongCatTransformer transformer = new(config);
            transformer.LoadWeights(weights);
            using Tensor cond = LongCatTestData.Latent(config, frames: 2, height: 3, width: 4, seed: 44);

            using CpuBackend backend = new();
            Assert.Throws<ArgumentException>(() => transformer.BuildKvCache(backend, cond, new LongCatConditioning(NumCondLatents: 2)));
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }

    [Fact]
    public void TheDenoisingPassRefusesATimestepCountThatIsNeitherOneNorTheNoisyFrames()
    {
        LongCatConfig config = LongCatTestData.Tiny;
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, seed: 45);
        try
        {
            using LongCatTransformer transformer = new(config);
            transformer.LoadWeights(weights);
            using CpuBackend backend = new();
            using Tensor cond = LongCatTestData.Latent(config, frames: 2, height: 4, width: 4, seed: 46);
            using LongCatKvCache cache = transformer.BuildKvCache(backend, cond, new LongCatConditioning(NumCondLatents: 2));
            using Tensor noisy = LongCatTestData.Latent(config, frames: 3, height: 4, width: 4, seed: 47);
            using Tensor encoder = LongCatTestData.Encoder(config, rows: 6, seed: 48);

            // Three noisy frames, but the caller passes the full cond + noisy span (five values): refused, not read as the noisy frames' timesteps.
            Assert.Throws<ArgumentException>(() => transformer.ForwardCached(backend, noisy, encoder, [0f, 0f, 0f, 1f, 1f],
                new LongCatConditioning(NumCondLatents: 2), cache));
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }
}
