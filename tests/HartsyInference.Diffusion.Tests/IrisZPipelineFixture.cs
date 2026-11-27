using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;

namespace HartsyInference.Diffusion.Tests;

/// <summary>An <see cref="IrisZPipeline"/> on the CPU over <see cref="IrisZTestData"/>'s tiny random transformer and VAE.
/// The text encoder is built but carries no weights, so only prompt-free runs work.</summary>
internal sealed class IrisZPipelineFixture : IDisposable
{
    private readonly CpuBackend _backend = new();
    private readonly Flux2Transformer _transformer = new(IrisZTestData.Config(modalities: 4));
    private readonly Tensor _bnMean = IrisZTestData.Filled(0f, IrisZTestData.Channels);
    private readonly Tensor _bnVar = IrisZTestData.Filled(1f, IrisZTestData.Channels);
    private readonly Tensor _emptyPrompt = IrisZTestData.Random(new Random(8), 1, 5, IrisZTestData.Context);

    public IrisZPipelineFixture(float skyCeiling)
    {
        _transformer.LoadWeights(IrisZTestData.Converted(zeroModality: false));
        IrisZTestData.VaeWeights vaeWeights = new();
        VaeDecoder decoder = new(IrisZTestData.Vae);
        decoder.LoadWeights(vaeWeights);
        VaeEncoder encoder = new(IrisZTestData.Vae);
        encoder.LoadWeights(vaeWeights);
        Config = IrisZTestData.Config(modalities: 4);
        TextEncoder = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_8B);
        Pipeline = new IrisZPipeline(_backend, TextEncoder, _transformer, decoder, encoder, _bnMean, _bnVar, Config,
            _emptyPrompt, skyCeiling, [9, 18, 27]);
    }

    public Flux2Config Config { get; }

    public LlamaStyleEncoder TextEncoder { get; }

    public IrisZPipeline Pipeline { get; }

    public void Dispose()
    {
        Pipeline.Dispose();
        _transformer.Dispose();
        _bnMean.Dispose();
        _bnVar.Dispose();
        _emptyPrompt.Dispose();
        _backend.Dispose();
    }
}
