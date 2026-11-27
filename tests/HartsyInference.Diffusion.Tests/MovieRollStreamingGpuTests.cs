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

/// <summary>MovieRoll's streaming forwards on a device: the tiny DiT's <c>StreamEncode</c>, <c>StreamDenoise</c> and <c>StreamRecompute</c> (one chunk and
/// five chunks) on CUDA against the CPU run, in two GEMM lanes. Each forward must read nothing back from the device before its result is compared
/// (the count is asserted to be zero). Every tensor is compared and reported (<c>MAX_ERR</c> lines), not just the first to fail.
/// This does not run a full <c>MovieRollSampler</c> pass, so it shows zero reads for those three forwards only.
/// Runs only through the GPU session (Category=GpuIntegration); it is skipped where no CUDA device is present.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class MovieRollStreamingGpuTests(ITestOutputHelper output)
{
    private const int Channels = 4;
    private const int Height = 2;
    private const int Width = 3;
    private const int Blocks = 2;
    private const int Hidden = 32;

    /// <summary>Bound for the exact lane (full-F32 GEMMs). Measured on a 4090 at 800f8a182 and again at 5a22be77c (identical): the largest error over
    /// every tensor of the eight facts is 3.7e-4, in recompute with one chunk at block 2 keys. Block 1 is at most 8.3e-7. The margin to the bound is 26%.</summary>
    private const float ExactTolerance = 5e-4f;

    /// <summary>Bound for the default lane (TF32 GEMMs). Measured on a 4090 at 800f8a182 and again at 5a22be77c (identical): the largest error is
    /// 1.479e-3, in recompute with one chunk at block 2 keys. The encode values are block 2 keys 1.274e-3, block 1 keys 1.068e-3 and velocity 6.1e-4.
    /// The margin to the bound is 26%.</summary>
    private const float Tf32Tolerance = 2e-3f;

    [CudaDeviceFact]
    public void Cuda_StreamEncode_ExactGemms_ReadsNothingBackAndMatchesTheCpuCache() =>
        RunEncode(exactGemm: true, ExactTolerance);

    [CudaDeviceFact]
    public void Cuda_StreamEncode_Tf32Gemms_ReadsNothingBackAndMatchesTheCpuCache() =>
        RunEncode(exactGemm: false, Tf32Tolerance);

    [CudaDeviceFact]
    public void Cuda_StreamDenoise_ExactGemms_ReadsNothingBackAndMatchesTheCpuRun() =>
        RunDenoise(exactGemm: true, ExactTolerance);

    [CudaDeviceFact]
    public void Cuda_StreamDenoise_Tf32Gemms_ReadsNothingBackAndMatchesTheCpuRun() =>
        RunDenoise(exactGemm: false, Tf32Tolerance);

    private void RunEncode(bool exactGemm, float tolerance)
    {
        using Fixture cpuFixture = new();
        using MovieRollKvCache cpuCache = new(cpuFixture.Backend);
        cpuCache.Init(cpuFixture.Dit.StreamLayerNames(), Height, Width);
        cpuFixture.Dit.StreamEncode(cpuFixture.Backend, Latent(2), new float[2], cpuCache, fullHistoryBlocks: 1, startFrame: 0);

        using Fixture device = new(GpuDevices.OpenCuda());
        ApplyLane(device.Backend, exactGemm);
        using MovieRollKvCache cache = new(device.Backend);
        cache.Init(device.Dit.StreamLayerNames(), Height, Width);
        device.Backend.ResetD2hSyncCount();
        device.Dit.StreamEncode(device.Backend, Latent(2), new float[2], cache, fullHistoryBlocks: 1, startFrame: 0);
        long reads = device.Backend.GetD2hSyncCount();
        ReportRun(device.Backend, "StreamEncode", reads);
        Assert.Equal(0, reads);

        List<string> failures = [];
        foreach (string layer in cpuFixture.Dit.StreamLayerNames())
        {
            (Tensor ek, Tensor ev)? e = cpuCache.Get(layer);
            (Tensor ak, Tensor av)? a = cache.Get(layer);
            Assert.True(e is not null && a is not null, $"{layer}: a cache entry is missing.");
            Compare($"{layer} keys", e!.Value.ek, a!.Value.ak, tolerance, failures);
            Compare($"{layer} values", e.Value.ev, a.Value.av, tolerance, failures);
        }
        Assert.True(failures.Count == 0, $"Bound {tolerance:E0}: {string.Join("; ", failures)}");
    }

    private void RunDenoise(bool exactGemm, float tolerance)
    {
        using Fixture cpuFixture = new();
        using MovieRollKvCache cpuCache = new(cpuFixture.Backend);
        cpuCache.Init(cpuFixture.Dit.StreamLayerNames(), Height, Width);
        using Tensor expected = cpuFixture.Dit.StreamDenoise(cpuFixture.Backend, Latent(1, 5), new float[1], 0f, cpuFixture.Qwen, cpuFixture.ByT5,
            cpuCache, fullHistoryBlocks: 1);

        using Fixture device = new(GpuDevices.OpenCuda());
        ApplyLane(device.Backend, exactGemm);
        using MovieRollKvCache cache = new(device.Backend);
        cache.Init(device.Dit.StreamLayerNames(), Height, Width);
        device.Backend.ResetD2hSyncCount();
        using Tensor actual = device.Dit.StreamDenoise(device.Backend, Latent(1, 5), new float[1], 0f, device.Qwen, device.ByT5, cache,
            fullHistoryBlocks: 1);
        // Counted before the velocity is read back for the comparison below, so the count covers the forward alone.
        long reads = device.Backend.GetD2hSyncCount();
        ReportRun(device.Backend, "StreamDenoise (forward only, before the comparison read)", reads);
        Assert.Equal(0, reads);

        List<string> failures = [];
        Compare("StreamDenoise velocity", expected, actual, tolerance, failures);
        Assert.True(failures.Count == 0, $"Bound {tolerance:E0}: {string.Join("; ", failures)}");
    }

    private void RunRecompute(bool exactGemm, int[] chunkFrames, float tolerance)
    {
        int frames = chunkFrames.Sum();
        using Fixture cpuFixture = new();
        using MovieRollKvCache cpuCache = new(cpuFixture.Backend);
        cpuCache.Init(cpuFixture.Dit.StreamLayerNames(), Height, Width);
        int cpuFrames = cpuFixture.Dit.StreamRecompute(cpuFixture.Backend, Latent(frames, 7), chunkFrames, cpuCache, fullHistoryBlocks: 1);

        using Fixture device = new(GpuDevices.OpenCuda());
        ApplyLane(device.Backend, exactGemm);
        using MovieRollKvCache cache = new(device.Backend);
        cache.Init(device.Dit.StreamLayerNames(), Height, Width);
        device.Backend.ResetD2hSyncCount();
        int deviceFrames = device.Dit.StreamRecompute(device.Backend, Latent(frames, 7), chunkFrames, cache, fullHistoryBlocks: 1);
        long reads = device.Backend.GetD2hSyncCount();
        ReportRun(device.Backend, $"StreamRecompute chunks [{string.Join(", ", chunkFrames)}]", reads);
        Assert.Equal(0, reads);
        Assert.Equal(cpuFrames, deviceFrames);

        List<string> failures = [];
        foreach (string layer in cpuFixture.Dit.StreamLayerNames())
        {
            (Tensor ek, Tensor ev)? e = cpuCache.Get(layer);
            (Tensor ak, Tensor av)? a = cache.Get(layer);
            Assert.True(e is not null && a is not null, $"{layer}: a cache entry is missing.");
            Compare($"recompute {layer} keys", e!.Value.ek, a!.Value.ak, tolerance, failures);
            Compare($"recompute {layer} values", e.Value.ev, a.Value.av, tolerance, failures);
        }
        Assert.True(failures.Count == 0, $"Bound {tolerance:E0}: {string.Join("; ", failures)}");
    }

    [CudaDeviceFact]
    public void Cuda_StreamRecomputeFiveChunks_ExactGemms_ReadsNothingBackAndMatchesTheCpuCache() =>
        RunRecompute(exactGemm: true, [1, 1, 1, 1, 1], ExactTolerance);

    [CudaDeviceFact]
    public void Cuda_StreamRecomputeFiveChunks_Tf32Gemms_ReadsNothingBackAndMatchesTheCpuCache() =>
        RunRecompute(exactGemm: false, [1, 1, 1, 1, 1], Tf32Tolerance);

    [CudaDeviceFact]
    public void Cuda_StreamRecomputeOneChunk_ExactGemms_ReadsNothingBackAndMatchesTheCpuCache() =>
        RunRecompute(exactGemm: true, [5], ExactTolerance);

    [CudaDeviceFact]
    public void Cuda_StreamRecomputeOneChunk_Tf32Gemms_ReadsNothingBackAndMatchesTheCpuCache() =>
        RunRecompute(exactGemm: false, [5], Tf32Tolerance);

    /// <summary>Prints the device-to-host read count of a forward with two switches: <c>HighPrecisionGemm</c> (the lane's GEMM setting) and
    /// <c>CudnnSdpaEngaged</c> (whether the cuDNN scaled-dot-product path has run on this backend, which is true once any run has used it). It does
    /// not print the TF32 eligibility or the SDPA no-TF32 flag, so a lane's TF32 state is not fully recorded by this line.</summary>
    private void ReportRun(IBackend backend, string run, long reads)
    {
        output.WriteLine($"D2H_SYNCS {run}: {reads}");
        if (backend is HartsyInference.Cuda.CudaBackend cuda)
            output.WriteLine($"SWITCHES {run}: HighPrecisionGemm={backend.HighPrecisionGemm} CudnnSdpaEngaged={cuda.CudnnSdpaEngaged}");
    }

    /// <summary>Sets the GEMM lane on the device. <c>exactGemm</c> forces full-F32 GEMMs (<see cref="IBackend.HighPrecisionGemm"/>), so the error that
    /// remains is not TF32 rounding; otherwise the device's default (TF32) stays on.</summary>
    private void ApplyLane(IBackend backend, bool exactGemm)
    {
        backend.HighPrecisionGemm = exactGemm;
        output.WriteLine($"GEMM_LANE {(exactGemm ? "exact" : "tf32")} HighPrecisionGemm={backend.HighPrecisionGemm}");
    }

    /// <summary>Compares two tensors, prints the largest absolute error with its flat index and the two values there, and records the tensor in
    /// <paramref name="failures"/> when that error exceeds <paramref name="tolerance"/>. Every tensor is compared; nothing stops at the first failure.</summary>
    private void Compare(string label, Tensor expected, Tensor actual, float tolerance, List<string> failures)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        float max = 0f;
        int at = 0;
        for (int i = 0; i < e.Length; i++)
        {
            float d = Math.Abs(e[i] - a[i]);
            if (float.IsNaN(d))
            {
                max = float.NaN;
                at = i;
                break;
            }
            if (d > max) { max = d; at = i; }
        }
        string values = e.Length > 0 ? $"(cpu {e[at]:G9}, device {a[at]:G9})" : "(empty tensor)";
        output.WriteLine($"MAX_ERR {label}: {max:E3} at flat index {at} of {e.Length} {values}");
        // NaN never compares below a bound, so a NaN in either tensor fails here rather than passing as a zero error.
        if (!(max <= tolerance)) failures.Add($"{label} max abs error {max:E3} at index {at} exceeds {tolerance:E0} (NaN counts as a failure)");
    }

    private static Tensor Latent(int frames, long seed = 3) =>
        TorchCompatibleNoise.Randn(new TensorShape([1L, Channels, frames, Height, Width]), seed);

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
