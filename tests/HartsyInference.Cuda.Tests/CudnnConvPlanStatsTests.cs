using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The cuDNN convolution plan-cache counters the audio first-synthesis bench reads: one plan per distinct
/// shape with the time extent included, and one family per shape without it. Skips when CUDA is unavailable.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudnnConvPlanStatsTests
{
    private const int InChannels = 8;
    private const int OutChannels = 16;
    private const int Kernel = 3;

    private readonly ITestOutputHelper _output;

    public CudnnConvPlanStatsTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void AudioConv1d_BuildsOnePlanPerLength_InOneFamily()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine($"SKIPPED: CUDA unavailable ({CudaContext.LastUnavailableReason})");
            return;
        }
        KnobStore.Set(EngineKnobs.AudioConvCudnn, true);
        try
        {
            using CudaBackend cuda = new CudaBackend(0, PtxDir());
            using Tensor weight = Random(new TensorShape(OutChannels, InChannels, Kernel), seed: 1);
            using Tensor bias = Random(new TensorShape(OutChannels), seed: 2);
            Convolve(cuda, weight, bias, length: 64);
            Convolve(cuda, weight, bias, length: 64);
            Convolve(cuda, weight, bias, length: 96);

            CudnnConvPlanStats stats = cuda.CudnnConvPlanStats;
            string families = cuda.DescribeCudnnConvPlanFamilies();
            _output.WriteLine(stats.ToString());
            _output.WriteLine(families);
            Assert.True(cuda.CudnnConvEngaged, "the audio conv did not take the cuDNN route");
            Assert.Equal(3, stats.Executions);
            Assert.Equal(2, stats.PlanBuilds);
            Assert.Equal(2, stats.CachedPlans);
            Assert.True(stats.ConfigsTried >= 2, $"each build finalizes at least one config, saw {stats.ConfigsTried}");
            Assert.True(stats.BuildMs > 0);
            Assert.True(stats.BuildMs + 1e-6 >= stats.GraphMs + stats.HeuristicMs + stats.FinalizeMs,
                $"the phases ({stats.GraphMs} + {stats.HeuristicMs} + {stats.FinalizeMs} ms) exceed the build ({stats.BuildMs} ms)");
            string[] lines = families.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            Assert.Single(lines);
            Assert.Contains("builds=2", lines[0], StringComparison.Ordinal);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.AudioConvCudnn);
        }
    }

    private static void Convolve(CudaBackend cuda, Tensor weight, Tensor bias, int length)
    {
        using Tensor input = Random(new TensorShape(1, InChannels, length), seed: length);
        using Tensor output = new Tensor(new TensorShape(1, OutChannels, length), DType.F32);
        ((IBackend)cuda).Conv1d(output, input, weight, bias, stride: 1, padLeft: 1, padRight: 1, dilation: 1, groups: 1);
        cuda.Sync();
        _ = *(float*)output.DataPointer;
    }

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        for (long i = 0; i < shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }
}
