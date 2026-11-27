using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary><c>numerics.longcatFp8Ffn</c> on CUDA against the CPU on the same checkpoint: the forward, and the forward with a LoRA gated on. Tiny
/// width at the release's 48 blocks, synthetic weights.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class LongCatFp8FfnCudaTests(ITestOutputHelper output) : IDisposable
{
    private const int Frames = 3, Height = 8, Width = 8, TextRows = 6;

    /// <summary>Relative RMS against the CPU; the device multiplies in its own precision. Set before the first run.</summary>
    private const double Bound = 1e-2;

    private readonly string _root = Path.Combine(Path.GetTempPath(), "longcat-fp8-cuda-" + Guid.NewGuid().ToString("N"));
    private readonly LongCatConfig _config = LongCatTestData.Tiny with { Depth = 48 };

    public void Dispose()
    {
        if (Directory.Exists(_root))
            Directory.Delete(_root, recursive: true);
    }

    [CudaDeviceFact]
    public void TheFp8ForwardAndItsLoraMatchTheCpu()
    {
        Directory.CreateDirectory(_root);
        Dictionary<string, Tensor> dense = LongCatTestData.Weights(_config, 97);
        try
        {
            LongCatTestData.WriteCheckpoint(_root, dense, writeConfig: false, storage: DType.BF16);
            File.WriteAllText(Path.Combine(_root, "config.json"), LongCatTinyAvatarRelease.ConfigJson(_config));
        }
        finally
        {
            LongCatTestData.DisposeAll(dense);
        }
        using IDisposable on = KnobProfile.Create("longcat-fp8-ffn").With(EngineKnobs.LongCatFp8Ffn, true).Push();
        using LongCatLoraFixture fixture = new(Path.Combine(_root, "lora.safetensors"), _config, 7);
        (float[] cpuOff, float[] cpuOn) = Run(new CpuBackend(), fixture);
        (float[] cudaOff, float[] cudaOn) = Run(GpuDevices.OpenCuda(), fixture);

        double off = LongCatTestData.RelRms(cudaOff, cpuOff), gated = LongCatTestData.RelRms(cudaOn, cpuOn),
            moved = LongCatTestData.RelRms(cpuOn, cpuOff);
        output.WriteLine($"relRMS gate off {off:E3}, gate on {gated:E3}; the LoRA moves the output {moved:E3}");
        Assert.True(off <= Bound, $"gate off {off:E3}");
        Assert.True(gated <= Bound, $"gate on {gated:E3}");
        // A LoRA that moved the output by less than the bound would pass with the adapter dropped.
        Assert.True(moved > 5 * Bound, $"the LoRA moves the output by only {moved:E3}");
    }

    private (float[] Off, float[] On) Run(IBackend backend, LongCatLoraFixture fixture)
    {
        using IBackend owned = backend;
        using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(_root);
        Dictionary<string, Tensor> weights = new(checkpoint.Weights, StringComparer.Ordinal);
        using QuantizedWeightPolicy.PreparedWeights prepared = QuantizedWeightPolicy.PrepareForBackends(weights, [backend], DType.F32);
        using LoraFile file = LoraFile.Load(fixture.Path);
        using LoraStack stack = new();
        AdjunctGate gate = new();
        stack.AddGated(file, gate);
        stack.ApplyTo(weights, LoraTarget.Transformer, backend);
        using LongCatTransformer transformer = new(_config);
        transformer.LoadWeights(weights);
        float[] off = Forward(backend, transformer);
        float[] on;
        using (gate.Enable(1f))
            on = Forward(backend, transformer);
        return (off, on);
    }

    private float[] Forward(IBackend backend, LongCatTransformer transformer)
    {
        using Tensor latent = LongCatTestData.Latent(_config, Frames, Height, Width, 2);
        using Tensor text = LongCatTestData.Encoder(_config, TextRows, 3);
        LongCatConditioning conditioning = new(NumCondLatents: 1);
        using Tensor velocity = transformer.Forward(backend, latent, text, conditioning.FrameTimesteps(500f, Frames), in conditioning);
        return velocity.AsReadOnlySpan<float>().ToArray();
    }
}
