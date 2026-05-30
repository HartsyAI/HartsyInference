using SharpInference.Core.Exceptions;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Pipelines;
using SharpInference.ModelHandler.CheckpointConverters;
using Xunit;

namespace SharpInference.Diffusion.Tests;

/// <summary>Checkpoint-free tests for <see cref="LensPipelineFactory"/>'s fail-fast wiring contract.
/// The full happy path needs the 30.7 GB <c>microsoft/Lens</c> weights (covered by the env-gated
/// generation test); these pin the cheap, deterministic part — the VAE BatchNorm extraction that runs
/// before any multi-GB component load.</summary>
public sealed class LensPipelineFactoryTests
{
    private static LensCheckpointConverter.ConvertedWeights MakeConverted(Dictionary<string, Tensor> vae)
        => new()
        {
            Transformer = new Dictionary<string, Tensor>(),
            TextEncoder = new Dictionary<string, Tensor>(),
            Vae = vae,
        };

    [Fact]
    public void BuildFromConverted_MissingRunningMean_ThrowsBeforeLoadingTransformer()
    {
        using CpuBackend backend = new CpuBackend();
        LensCheckpointConverter.ConvertedWeights weights = MakeConverted(new Dictionary<string, Tensor>());

        SharpInferenceException ex = Assert.Throws<SharpInferenceException>(
            () => LensPipelineFactory.BuildFromConverted(backend, weights, LensConfig.Default));
        Assert.Contains("bn.running_mean", ex.Message);
    }

    [Fact]
    public void BuildFromConverted_MissingRunningVar_ThrowsWithVarMessage()
    {
        using CpuBackend backend = new CpuBackend();
        Dictionary<string, Tensor> vae = new()
        {
            ["bn.running_mean"] = new Tensor(new TensorShape(128), DType.F32),
        };
        LensCheckpointConverter.ConvertedWeights weights = MakeConverted(vae);

        SharpInferenceException ex = Assert.Throws<SharpInferenceException>(
            () => LensPipelineFactory.BuildFromConverted(backend, weights, LensConfig.Default));
        Assert.Contains("bn.running_var", ex.Message);
    }
}
