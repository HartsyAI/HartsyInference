using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Recipes.Image;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Fibo 2.0 weight residency: the text encoder is preloaded for its encode and freed before the denoiser loads, and the free
/// happens on the failure path too. The recorder backend shows the order of the preload and free calls on the CPU lane.</summary>
public sealed class Fibo2ResidencyTests
{
    [Fact]
    public void GenerateFromTokens_FreesTheTextEncoder_EvenWhenItsEncodeFails()
    {
        using CpuBackend cpu = new();
        IBackend backend = ResidentWeightsTests.Recorder.Create(cpu, out ResidentWeightsTests.Recorder recorder);
        using Fibo2Pipeline pipeline = UnloadedPipeline(backend);

        // The components hold no weights, so the encode throws. The text encoder must still be freed, and the denoiser must not load.
        Assert.ThrowsAny<Exception>(() => pipeline.GenerateFromTokens([1], [1], 64, 64, 1, 1f, 1.15f, 7));

        Assert.Equal(new[] { "PreloadWeights", "FreeWeights" }, recorder.Calls.Select(c => c.Kind));
    }

    [Fact]
    public void TransformerEnumeratesNoWeightsBeforeItsWeightsLoad()
    {
        using Fibo2Transformer transformer = new(TinyConfig());

        Assert.Empty(transformer.EnumerateWeights());
    }

    private static Fibo2Pipeline UnloadedPipeline(IBackend backend)
    {
        Fibo2Config config = TinyConfig();
        LlamaStyleEncoderConfig encoderConfig = LlamaStyleEncoderConfig.Qwen3_VL_4B with
        {
            HiddenSize = 32, NumLayers = 36, NumQueryHeads = 2, NumKvHeads = 1, HeadDim = 16, IntermediateSize = 64, VocabSize = 64,
        };
        return new Fibo2Pipeline(backend, new LlamaStyleEncoder(encoderConfig), new Fibo2Transformer(config),
            new VaeDecoder(VaeConfig.Flux2), new Tensor(new TensorShape(128), DType.F32), new Tensor(new TensorShape(128), DType.F32),
            config);
    }

    private static Fibo2Config TinyConfig() => Fibo2Config.Checkpoint with
    {
        Dim = 64, NumHeads = 2, HeadDim = 32, NumLayers = 2, NumRefinerLayers = 1, InjectionLayers = [1], CapFeatDim = 96,
        PerceiverQueries = 8, PerceiverLayers = 1, FfnDim = 96, AxesDims = [8, 12, 12], MaxGist = 8, MinGist = 4, GistStep = 2,
        GistMinTextLen = 16, GistMaxTextLen = 64,
    };
}
