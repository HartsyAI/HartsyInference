using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Vae;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The encoder's spatial downsample must never read an activation back to the host. It used to transpose,
/// asymmetrically zero-pad and transpose back with host pointer loops, at FULL encoder resolution — three device
/// round-trips per stage per chunk, which at 480x800/61f was 27 s of the Wan-Animate-2 VAE encode phase and showed
/// up in no op's profile because none of it was an op.</summary>
[Collection("CudaSerial")]
public sealed unsafe class Wan22ResampleDownsampleResidencyTests
{
    private readonly ITestOutputHelper _output;

    public Wan22ResampleDownsampleResidencyTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void Downsample2d_StaysDeviceResident_AndMatchesTheCpuReference()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        const int dim = 6, t = 3, h = 8, w = 10;
        using Tensor x = Filled(new TensorShape([1L, dim, t, h, w]), 7717, 0.6f);
        Dictionary<string, Tensor> weights = new()
        {
            ["down.resample.1.weight"] = Filled(new TensorShape(dim, dim, 3, 3), 7723, 0.25f),
            ["down.resample.1.bias"] = Filled(new TensorShape(dim), 7727, 0.1f),
        };

        Wan22Resample cpuLayer = new(dim, Wan22ResampleMode.Downsample2d);
        cpuLayer.LoadWeights(weights, "down");
        using CpuBackend cpu = new();
        using Tensor expected = cpuLayer.Forward(cpu, x);

        Wan22Resample gpuLayer = new(dim, Wan22ResampleMode.Downsample2d);
        gpuLayer.LoadWeights(weights, "down");
        using CudaBackend cuda = new(0, PtxDir());
        cuda.PreloadWeights(gpuLayer.EnumerateWeights());
        // The input must be a DEVICE-PRODUCED activation, which is what the encoder actually hands this layer: a
        // host-authoritative tensor can be read through its pointer for free, so it hides the round-trip entirely.
        using Tensor deviceX = new(x.Shape, DType.F32);
        cuda.Scale(deviceX, x, 1f);
        cuda.Sync();
        cuda.ResetD2hSyncCount();

        using Tensor actual = gpuLayer.Forward(cuda, deviceX);
        cuda.Sync();
        // The host-pointer transpose/pad forms produce at least one readback here; the device forms produce none.
        Assert.Equal(0, cuda.GetD2hSyncCount());
        Assert.Equal(new TensorShape([1L, dim, t, h / 2, w / 2]), actual.Shape);

        float* e = (float*)expected.DataPointer, a = (float*)actual.DataPointer;
        for (long i = 0; i < expected.Shape.ElementCount; i++)
            Assert.True(MathF.Abs(e[i] - a[i]) <= 2e-5f, $"element {i}: cpu {e[i]} vs cuda {a[i]}");
        _output.WriteLine($"{expected.Shape.ElementCount} elements matched, intermediate D2H=0");

        foreach (Tensor weight in weights.Values) weight.Dispose();
    }

    private static Tensor Filled(TensorShape shape, int seed, float scale)
    {
        Tensor tensor = new(shape, DType.F32);
        float* p = (float*)tensor.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return tensor;
    }
}
