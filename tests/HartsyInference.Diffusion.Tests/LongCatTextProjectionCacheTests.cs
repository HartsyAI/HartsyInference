using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The text projection is computed once per encoder tensor, so a denoise loop that keeps its prompt tensor alive projects the prompt once for
/// the whole run; a fresh tensor per pass projects every pass. The outputs are the same either way.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatTextProjectionCacheTests : IDisposable
{
    private readonly LongCatConfig _config = LongCatTestData.Tiny;
    private readonly Dictionary<string, Tensor> _weights;
    private readonly CpuBackend _backend = new();

    public LongCatTextProjectionCacheTests()
    {
        _weights = LongCatTestData.Weights(_config, 59);
    }

    public void Dispose()
    {
        LongCatTestData.DisposeAll(_weights);
        _backend.Dispose();
    }

    [Fact]
    public void AKeptPromptTensorIsProjectedOnceAcrossPasses()
    {
        using LongCatTransformer transformer = new(_config);
        transformer.LoadWeights(_weights);
        using Tensor latent = LongCatTestData.Latent(_config, 5, 4, 4, 2);
        using Tensor text = LongCatTestData.Encoder(_config, 5, 3);
        LongCatConditioning conditioning = new(NumCondLatents: 1);

        float[] first = Pass(transformer, latent, text, conditioning);
        float[] second = Pass(transformer, latent, text, conditioning);
        float[] third = Pass(transformer, latent, text, conditioning);

        Assert.Equal(1, transformer.TextProjectionBuilds);
        Assert.Equal(first, second);
        Assert.Equal(first, third);
    }

    [Fact]
    public void AFreshPromptTensorPerPassIsProjectedEveryPass()
    {
        using LongCatTransformer transformer = new(_config);
        transformer.LoadWeights(_weights);
        using Tensor latent = LongCatTestData.Latent(_config, 5, 4, 4, 2);
        LongCatConditioning conditioning = new(NumCondLatents: 1);

        for (int pass = 0; pass < 3; pass++)
        {
            using Tensor text = LongCatTestData.Encoder(_config, 5, 3);
            Pass(transformer, latent, text, conditioning);
        }

        Assert.Equal(3, transformer.TextProjectionBuilds);
    }

    private float[] Pass(LongCatTransformer transformer, Tensor latent, Tensor text, LongCatConditioning conditioning)
    {
        using Tensor velocity = transformer.Forward(_backend, latent, text, conditioning.FrameTimesteps(300f, 5), in conditioning);
        return velocity.AsReadOnlySpan<float>().ToArray();
    }
}
