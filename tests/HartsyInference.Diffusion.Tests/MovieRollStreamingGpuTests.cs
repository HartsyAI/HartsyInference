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
/// is certain (the encode returns nothing, so it reads nothing back). The denoise returns the velocity, so one read of it is expected.
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
    public void Cuda_StreamDenoise_ReadsOnlyTheVelocityBackAndMatchesTheCpuRun()
    {
        using Fixture cpuFixture = new();
        using MovieRollKvCache cpuCache = new(cpuFixture.Backend);
        cpuCache.Init(cpuFixture.Dit.StreamLayerNames(), Height, Width);
        using Tensor expected = cpuFixture.Dit.StreamDenoise(cpuFixture.Backend, Latent(1, 5), new float[1], 0f, cpuFixture.Qwen, cpuFixture.ByT5,
            cpuCache, fullHistoryBlocks: 1);

        using Fixture device = new(GpuDevices.OpenCuda());
        using MovieRollKvCache cache = new(device.Backend);
        cache.Init(device.Dit.StreamLayerNames(), Height, Width);
        device.Backend.ResetD2hSyncCount();
        using Tensor actual = device.Dit.StreamDenoise(device.Backend, Latent(1, 5), new float[1], 0f, device.Qwen, device.ByT5, cache,
            fullHistoryBlocks: 1);
        long reads = device.Backend.GetD2hSyncCount();
        output.WriteLine($"D2H_SYNCS StreamDenoise (the velocity read included): {reads}");

        // The velocity is read once when compared; anything beyond that means a forward is reading back to the host.
        Assert.True(reads <= 1, $"StreamDenoise read back {reads} times; expected at most the velocity read.");
        AssertClose(expected, actual);
    }

    private static Tensor Latent(int frames, long seed = 3) =>
        TorchCompatibleNoise.Randn(new TensorShape([1L, Channels, frames, Height, Width]), seed);

    private static void AssertCacheClose(MovieRollKvCache expected, MovieRollKvCache actual, IEnumerable<string> layers)
    {
        foreach (string layer in layers)
        {
            (Tensor ek, Tensor ev)? e = expected.Get(layer);
            (Tensor ak, Tensor av)? a = actual.Get(layer);
            Assert.True(e is not null && a is not null, $"{layer}: a cache entry is missing.");
            AssertClose(e!.Value.ek, a!.Value.ak);
            AssertClose(e.Value.ev, a.Value.av);
        }
    }

    private static void AssertClose(Tensor expected, Tensor actual)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        float max = 0f;
        for (int i = 0; i < e.Length; i++) max = Math.Max(max, Math.Abs(e[i] - a[i]));
        Assert.True(max <= Tolerance, $"max abs error {max:E3} exceeds {Tolerance:E0}.");
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
