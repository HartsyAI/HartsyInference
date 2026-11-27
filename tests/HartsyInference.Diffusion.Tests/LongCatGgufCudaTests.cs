using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A LongCat GGUF repack on CUDA, its block quants packed where the backend decodes them, against the CPU on the same file: the forward,
/// and the forward with the release's distillation LoRA gated on. Tiny config, synthetic weights.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class LongCatGgufCudaTests(ITestOutputHelper output) : IDisposable
{
    private const int Frames = 3, Height = 8, Width = 8, TextRows = 6;

    /// <summary>Relative RMS against the CPU, which widens every weight to F32; the device decodes and multiplies in its own precision. Set before
    /// the first run.</summary>
    private const double Bound = 1e-2;

    private readonly string _root = Path.Combine(Path.GetTempPath(), "longcat-gguf-cuda-" + Guid.NewGuid().ToString("N"));
    private readonly LongCatConfig _config = LongCatTestData.Tiny;

    public void Dispose()
    {
        if (Directory.Exists(_root))
            Directory.Delete(_root, recursive: true);
    }

    [CudaDeviceFact]
    public void ThePackedForwardAndItsLoraMatchTheCpu()
    {
        // Two types the backend decodes packed, a mix whose joined weights are BF16, and one it widens on the host.
        List<string> failures = [];
        foreach ((string quant, string valueQuant) in new[] { ("Q8_0", "Q8_0"), ("Q4_0", "Q4_0"), ("Q4_0", "Q8_0"), ("Q5_1", "Q5_1") })
            Compare(quant, valueQuant, failures);

        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    private void Compare(string quant, string valueQuant, List<string> failures)
    {
        string root = Path.Combine(_root, quant + "-" + valueQuant);
        Directory.CreateDirectory(root);
        Dictionary<string, Tensor> dense = LongCatTestData.Weights(_config, 61);
        try
        {
            LongCatTestData.DisposeAll(LongCatGgufTestData.Write(root, dense, quant, valueQuant));
            using LongCatLoraFixture fixture = new(Path.Combine(root, "dmd.safetensors"), _config, 7);
            (float[] cpuOff, float[] cpuOn, _) = Run(new CpuBackend(), root, fixture);
            (float[] cudaOff, float[] cudaOn, string held) = Run(GpuDevices.OpenCuda(), root, fixture);

            double off = LongCatTestData.RelRms(cudaOff, cpuOff), on = LongCatTestData.RelRms(cudaOn, cpuOn),
                moved = LongCatTestData.RelRms(cpuOn, cpuOff);
            output.WriteLine($"{quant}/{valueQuant}: qkv held as {held}; relRMS gate off {off:E3}, gate on {on:E3}; "
                + $"the LoRA moves the output {moved:E3}");
            // A LoRA that moved the output by less than the bound would pass with the adapter dropped.
            if (!(off <= Bound && on <= Bound && moved > 5 * Bound))
                failures.Add($"{quant}/{valueQuant}: gate off {off:E3}, gate on {on:E3} (bound {Bound}), LoRA movement {moved:E3}");
        }
        finally
        {
            LongCatTestData.DisposeAll(dense);
        }
    }

    private (float[] Off, float[] On, string Held) Run(IBackend backend, string root, LongCatLoraFixture fixture)
    {
        using IBackend owned = backend;
        using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(root);
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
        return (off, on, weights["blocks.0.attn.qkv.weight"].DType.Name);
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
