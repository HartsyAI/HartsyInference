using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>MovieRoll's streaming forwards on a device: the tiny DiT's <c>StreamEncode</c> and <c>StreamDenoise</c> on CUDA against the CPU run, with the
/// device-to-host read count recorded for each. The expected counts are not measured yet: until a 4090 run sets them, the test asserts only what
/// is certain: the encode returns nothing and the denoise's count is taken before its velocity is read back, so both forwards should read nothing.
/// Runs only through the GPU session (Category=GpuIntegration); it is skipped where no CUDA device is present.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class MovieRollStreamingGpuTests(ITestOutputHelper output)
{
    private const int Channels = 4;
    private const int Height = 2;
    private const int Width = 3;
    private const int Blocks = 2;
    private const int Hidden = 32;
    private const float Tolerance = 1e-3f;

    [CudaDeviceFact]
    public void Cuda_StreamEncode_ReadsNothingBackAndMatchesTheCpuCache()
    {
        using Fixture cpuFixture = new();
        using MovieRollKvCache cpuCache = new(cpuFixture.Backend);
        cpuCache.Init(cpuFixture.Dit.StreamLayerNames(), Height, Width);
        cpuFixture.Dit.StreamEncode(cpuFixture.Backend, Latent(2), new float[2], cpuCache, fullHistoryBlocks: 1, startFrame: 0);

        using Fixture device = new(GpuDevices.OpenCuda());
        ApplyGemmLane(device.Backend);
        using MovieRollKvCache cache = new(device.Backend);
        cache.Init(device.Dit.StreamLayerNames(), Height, Width);
        device.Backend.ResetD2hSyncCount();
        device.Dit.StreamEncode(device.Backend, Latent(2), new float[2], cache, fullHistoryBlocks: 1, startFrame: 0);
        long reads = device.Backend.GetD2hSyncCount();
        output.WriteLine($"D2H_SYNCS StreamEncode: {reads}");

        Assert.Equal(0, reads);
        AssertCacheClose(cpuCache, cache, cpuFixture.Dit.StreamLayerNames());
    }

    [CudaDeviceFact]
    public void Cuda_StreamDenoise_ReadsNothingBackAndMatchesTheCpuRun()
    {
        using Fixture cpuFixture = new();
        using MovieRollKvCache cpuCache = new(cpuFixture.Backend);
        cpuCache.Init(cpuFixture.Dit.StreamLayerNames(), Height, Width);
        using Tensor expected = cpuFixture.Dit.StreamDenoise(cpuFixture.Backend, Latent(1, 5), new float[1], 0f, cpuFixture.Qwen, cpuFixture.ByT5,
            cpuCache, fullHistoryBlocks: 1);

        using Fixture device = new(GpuDevices.OpenCuda());
        ApplyGemmLane(device.Backend);
        using MovieRollKvCache cache = new(device.Backend);
        cache.Init(device.Dit.StreamLayerNames(), Height, Width);
        device.Backend.ResetD2hSyncCount();
        using Tensor actual = device.Dit.StreamDenoise(device.Backend, Latent(1, 5), new float[1], 0f, device.Qwen, device.ByT5, cache,
            fullHistoryBlocks: 1);
        // Counted before the velocity is read back for the comparison below, so the count covers the forward alone.
        long reads = device.Backend.GetD2hSyncCount();
        output.WriteLine($"D2H_SYNCS StreamDenoise (forward only, before the comparison read): {reads}");

        Assert.Equal(0, reads);
        AssertClose("StreamDenoise velocity", expected, actual);
    }

    /// <summary>The GEMM lane. <c>exact</c> forces full-F32 GEMMs (<see cref="IBackend.HighPrecisionGemm"/>), so any remaining error is not TF32
    /// rounding; <c>tf32</c> (the default, and the device's own setting) leaves TF32 on. Set by <c>MOVIEROLL_GEMM_LANE</c>.</summary>
    private void ApplyGemmLane(IBackend backend)
    {
        string lane = Environment.GetEnvironmentVariable("MOVIEROLL_GEMM_LANE") ?? "tf32";
        if (lane == "exact") backend.HighPrecisionGemm = true;
        else if (lane != "tf32") throw new InvalidOperationException($"MOVIEROLL_GEMM_LANE must be exact or tf32; got {lane}.");
        output.WriteLine($"GEMM_LANE {lane} HighPrecisionGemm={backend.HighPrecisionGemm}");
    }

    private static Tensor Latent(int frames, long seed = 3) =>
        TorchCompatibleNoise.Randn(new TensorShape([1L, Channels, frames, Height, Width]), seed);

    private void AssertCacheClose(MovieRollKvCache expected, MovieRollKvCache actual, IEnumerable<string> layers)
    {
        foreach (string layer in layers)
        {
            (Tensor ek, Tensor ev)? e = expected.Get(layer);
            (Tensor ak, Tensor av)? a = actual.Get(layer);
            Assert.True(e is not null && a is not null, $"{layer}: a cache entry is missing.");
            AssertClose($"{layer} keys", e!.Value.ek, a!.Value.ak);
            AssertClose($"{layer} values", e.Value.ev, a.Value.av);
        }
    }

    /// <summary>Compares two tensors and reports the largest absolute error with its flat index and the CPU and device values there, so a failure
    /// names the element. The bound is <see cref="Tolerance"/> in every lane.</summary>
    private void AssertClose(string label, Tensor expected, Tensor actual)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        float max = 0f;
        int at = 0;
        for (int i = 0; i < e.Length; i++)
        {
            float d = Math.Abs(e[i] - a[i]);
            if (d > max) { max = d; at = i; }
        }
        output.WriteLine($"MAX_ERR {label}: {max:E3} at flat index {at} of {e.Length} (cpu {(e.Length > 0 ? e[at] : 0f):G9}, device {(a.Length > 0 ? a[at] : 0f):G9})");
        Assert.True(max <= Tolerance, $"{label}: max abs error {max:E3} at flat index {at} exceeds {Tolerance:E0}.");
    }

    private static HunyuanVideo15Config TinyConfig() => new()
    {
        NumBlocks = Blocks, HiddenSize = Hidden, NumHeads = 2, LatentChannels = Channels, TextDim = 24, ByT5Dim = 16, RopeAxesDim = [4, 6, 6],
    };

    private sealed class Fixture : IDisposable
    {
        public Fixture() : this(new CpuBackend()) { }

        public Fixture(IBackend backend)
        {
            Backend = backend;
            Weights = new SafeTensorsLoader();
            Weights.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors"));
            Dit = new HunyuanVideo15Dit(TinyConfig());
            Dit.LoadWeights(HunyuanVideo15CheckpointConverter.Convert(
                Weights.GetAllTensors()
                    .Where(pair => !pair.Key.StartsWith("ref.", StringComparison.Ordinal) && !pair.Key.StartsWith("in.", StringComparison.Ordinal)
                        && !pair.Key.StartsWith("win.", StringComparison.Ordinal) && !pair.Key.StartsWith("rec.", StringComparison.Ordinal))
                    .ToDictionary(pair => pair.Key, pair => pair.Value),
                Blocks, Hidden));
        }

        public IBackend Backend { get; }
        public SafeTensorsLoader Weights { get; }
        public HunyuanVideo15Dit Dit { get; }
        public Tensor Qwen => Weights.GetTensor("in.qwen");
        public Tensor ByT5 => Weights.GetTensor("in.byt5");

        public void Dispose()
        {
            Dit.Dispose();
            Backend.Dispose();
            Weights.Dispose();
        }
    }
}
