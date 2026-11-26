using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.MovieRoll;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The whole MovieRoll sampler on a device: the tiny DiT's rolling loop over six latent frames (two windows of recompute and CFG), on CUDA
/// against the CPU run, in two GEMM lanes. The sampler must read nothing back from the device before its latents are compared (asserted zero).
/// Every tensor is compared and reported as <c>MAX_ERR</c>. The bounds are the streaming facts' bounds; the full-sampler error is not measured yet.
/// Runs only through the GPU session (Category=GpuIntegration); it is skipped where no CUDA device is present.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class MovieRollSamplerGpuTests(ITestOutputHelper output)
{
    private const int Channels = 4;
    private const int Height = 2;
    private const int Width = 3;
    private const int Blocks = 2;
    private const int Hidden = 32;

    /// <summary>Exact-lane bound, the same as the streaming facts. The full sampler measured 3.196e-4 on a 4090 (#1295, 10/10 passing, zero device reads):
    /// a 36% margin. The streaming facts' measured maximum is 3.7e-4 (recompute, one chunk, block 2 keys), at 800f8a182 and 5a22be77c.</summary>
    private const float ExactTolerance = 5e-4f;

    /// <summary>TF32-lane bound, the same as the streaming facts. The full sampler measured 1.381e-3 on the same 4090 run: a 31% margin. The streaming
    /// facts' measured maximum is 1.479e-3 (recompute, one chunk, block 2 keys), at 800f8a182 and 5a22be77c.</summary>
    private const float Tf32Tolerance = 2e-3f;

    private static readonly MovieRollSamplerOptions Options = new()
    {
        ChunkFrames = 1, FirstChunkFrames = 2, NumChunks = 5, FullHistoryBlocks = 1, ContextNoise = 0.05f,
        KvRecomputeEnabled = true, KvRecomputeThreshold = 3, KvRecomputeRecentK = 1, Cfg = new MovieRollCfg(1f, 4f),
    };

    [CudaDeviceFact]
    public void Cuda_FullSampler_ExactGemms_ReadsNothingBackAndMatchesTheCpuLatents() =>
        Run(exactGemm: true, ExactTolerance);

    [CudaDeviceFact]
    public void Cuda_FullSampler_Tf32Gemms_ReadsNothingBackAndMatchesTheCpuLatents() =>
        Run(exactGemm: false, Tf32Tolerance);

    private void Run(bool exactGemm, float tolerance)
    {
        using Fixture cpuFixture = new(new CpuBackend());
        using Tensor expected = cpuFixture.Sample(SeededNoise(7));

        using Fixture device = new(GpuDevices.OpenCuda());
        device.Backend.HighPrecisionGemm = exactGemm;
        output.WriteLine($"GEMM_LANE {(exactGemm ? "exact" : "tf32")} HighPrecisionGemm={device.Backend.HighPrecisionGemm}");
        device.Backend.ResetD2hSyncCount();
        using Tensor actual = device.Sample(SeededNoise(7));
        long reads = device.Backend.GetD2hSyncCount();

        // Everything is measured and printed before the first assertion, so a run that fails a check still reports its errors and counts.
        output.WriteLine($"D2H_SYNCS FullSampler: {reads}");
        output.WriteLine($"RECOMPUTES cpu={cpuFixture.LastRecomputeCount} device={device.LastRecomputeCount}");
        ReportSwitches(device.Backend, "FullSampler");
        List<string> failures = [];
        Compare("FullSampler latents", expected, actual, tolerance, failures);

        Assert.True(cpuFixture.LastRecomputeCount > 0, "The KV recompute did not fire on the CPU run, so the run does not cover it.");
        Assert.Equal(cpuFixture.LastRecomputeCount, device.LastRecomputeCount);
        Assert.Equal(0, reads);
        Assert.True(failures.Count == 0, $"Bound {tolerance:E0}: {string.Join("; ", failures)}");
    }

    /// <summary>Prints the attention and GEMM switches a run used, so its numbers can be read with the path that produced them.</summary>
    private void ReportSwitches(IBackend backend, string run)
    {
        if (backend is HartsyInference.Cuda.CudaBackend cuda)
            output.WriteLine($"SWITCHES {run}: HighPrecisionGemm={backend.HighPrecisionGemm} CudnnSdpaEngaged={cuda.CudnnSdpaEngaged}");
    }

    /// <summary>Compares two tensors, prints the largest absolute error with its flat index and the two values there, and records the tensor in
    /// <paramref name="failures"/> unless the error is within <paramref name="tolerance"/>. A NaN error fails.</summary>
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
        if (!(max <= tolerance)) failures.Add($"{label} max abs error {max:E3} at index {at} exceeds {tolerance:E0} (NaN counts as a failure)");
    }

    /// <summary>Draws the same noise sequence for a seed: each call takes the next seed, so the sampler's draws are reproducible.</summary>
    private static Func<int, Tensor> SeededNoise(long seed)
    {
        long next = seed;
        return frames => TorchCompatibleNoise.Randn(new TensorShape([1L, Channels, frames, Height, Width]), next++);
    }

    private static HunyuanVideo15Config TinyConfig() => new()
    {
        NumBlocks = Blocks, HiddenSize = Hidden, NumHeads = 2, LatentChannels = Channels, TextDim = 24, ByT5Dim = 16, RopeAxesDim = [4, 6, 6],
    };

    private sealed class Fixture : IDisposable
    {
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
            Qwen = Weights.GetTensor("in.qwen");
            ByT5 = Weights.GetTensor("in.byt5");
            // The negative prompt is a fixed random pair, as in MovieRollPipelineTests.
            Random random = new(77);
            NegativeQwen = RandomTensor(random, Qwen);
            NegativeByt5 = RandomTensor(random, ByT5);
        }

        public IBackend Backend { get; }
        public SafeTensorsLoader Weights { get; }
        public HunyuanVideo15Dit Dit { get; }
        public Tensor Qwen { get; }
        public Tensor ByT5 { get; }
        public Tensor NegativeQwen { get; }
        public Tensor NegativeByt5 { get; }

        /// <summary>KV recomputes the last <see cref="Sample"/> call made.</summary>
        public int LastRecomputeCount { get; private set; }

        /// <summary>The sampler on its own: one cache for the run, the schedule the pipeline tests use. The caller owns the returned latents.</summary>
        public Tensor Sample(Func<int, Tensor> noise)
        {
            using MovieRollKvCache cache = new(Backend);
            cache.Init(Dit.StreamLayerNames(), Height, Width);
            MovieRollSampler sampler = new(Backend, Dit, MovieRollSchedule.Base(4, 2, 5.0));
            Tensor latents = sampler.Sample(cache, Options, Qwen, ByT5, NegativeQwen, NegativeByt5, noise);
            LastRecomputeCount = sampler.RecomputeCount;
            return latents;
        }

        private static Tensor RandomTensor(Random random, Tensor like)
        {
            Tensor tensor = new(like.Shape, DType.F32);
            Span<float> values = tensor.AsSpan<float>();
            for (int i = 0; i < values.Length; i++) values[i] = (float)(random.NextDouble() * 2 - 1);
            return tensor;
        }

        public void Dispose()
        {
            NegativeQwen.Dispose();
            NegativeByt5.Dispose();
            Dit.Dispose();
            Backend.Dispose();
            Weights.Dispose();
        }
    }
}
