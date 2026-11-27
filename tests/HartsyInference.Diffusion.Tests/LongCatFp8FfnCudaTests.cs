using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary><c>numerics.longcatFp8Ffn</c> on CUDA against the CPU on the same checkpoint: the forward, and the forward with a LoRA gated on, in two
/// lanes. Tiny width at the release's 48 blocks, synthetic weights.</summary>
/// <remarks>The default GEMMs (TF32) over 48 blocks drift about 1e-2 from the CPU with the LoRA on, with or without FP8, so that lane compares the
/// FP8 drift with the BF16 DiT's from the same run. The exact lane (<see cref="ExactGemmScope"/>) has its own absolute bound.</remarks>
[Trait("Category", "GpuIntegration")]
public sealed class LongCatFp8FfnCudaTests(ITestOutputHelper output) : IDisposable
{
    private const int Frames = 3, Height = 8, Width = 8, TextRows = 6;

    /// <summary>The exact lane's relative RMS bound against the CPU. Measured on an RTX 3060 before it was set: 1.3e-3 gate off, 2.4e-3 gate
    /// on.</summary>
    private const double ExactBound = 5e-3;

    /// <summary>The default lane: how far FP8 may drift from the CPU as a multiple of the BF16 DiT's drift in the same run.</summary>
    private const double DefaultRatio = 1.5;

    private readonly string _root = Path.Combine(Path.GetTempPath(), "longcat-fp8-cuda-" + Guid.NewGuid().ToString("N"));
    private readonly LongCatConfig _config = LongCatTestData.Tiny with { Depth = 48 };

    public void Dispose()
    {
        if (Directory.Exists(_root))
            Directory.Delete(_root, recursive: true);
    }

    [CudaDeviceFact]
    public void TheFp8ForwardAndItsLoraStayWithTheCpu()
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
        using LongCatLoraFixture fixture = new(Path.Combine(_root, "lora.safetensors"), _config, 7);
        (double Off, double On, double Moved) control = Drift(fixture, exactGemm: false);
        using IDisposable on = KnobProfile.Create("longcat-fp8-ffn").With(EngineKnobs.LongCatFp8Ffn, true).Push();
        (double Off, double On, double Moved) fp8 = Drift(fixture, exactGemm: false);
        (double Off, double On, double Moved) exact = Drift(fixture, exactGemm: true);

        output.WriteLine($"relRMS against the CPU, gate off / on. default GEMMs: BF16 {control.Off:E3} / {control.On:E3}, "
            + $"FP8 {fp8.Off:E3} / {fp8.On:E3}; exact GEMMs: FP8 {exact.Off:E3} / {exact.On:E3}; the LoRA moves the output {fp8.Moved:E3}");
        Assert.True(exact.Off <= ExactBound && exact.On <= ExactBound, $"exact lane {exact.Off:E3} / {exact.On:E3}");
        Assert.True(fp8.Off <= DefaultRatio * control.Off, $"gate off: FP8 {fp8.Off:E3} against BF16 {control.Off:E3}");
        Assert.True(fp8.On <= DefaultRatio * control.On, $"gate on: FP8 {fp8.On:E3} against BF16 {control.On:E3}");
        // A LoRA that moved the output by less than the drift would pass with the adapter dropped.
        Assert.True(fp8.Moved > 0.1, $"the LoRA moves the output by only {fp8.Moved:E3}");
    }

    /// <summary>CUDA's relative RMS from the CPU with the LoRA off and on, and how far the LoRA moves the CPU's output, under the current
    /// knobs.</summary>
    private (double Off, double On, double Moved) Drift(LongCatLoraFixture fixture, bool exactGemm)
    {
        (float[] cpuOff, float[] cpuOn) = Run(new CpuBackend(), fixture, exactGemm: false);
        (float[] cudaOff, float[] cudaOn) = Run(GpuDevices.OpenCuda(), fixture, exactGemm);
        return (LongCatTestData.RelRms(cudaOff, cpuOff), LongCatTestData.RelRms(cudaOn, cpuOn), LongCatTestData.RelRms(cpuOn, cpuOff));
    }

    private (float[] Off, float[] On) Run(IBackend backend, LongCatLoraFixture fixture, bool exactGemm)
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
        using IDisposable? scope = exactGemm && backend is CudaBackend cuda ? new ExactGemmScope(cuda) : null;
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
