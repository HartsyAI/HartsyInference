using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The cuDNN convolution plan cache: one plan per distinct shape with the time extent included, and for 1D convs
/// one engine choice per (family, power-of-two length bucket), made by the heuristic at the bucket's own length so the
/// audio a length produces does not depend on which length came first. The GPU tests skip when CUDA is unavailable.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudnnConvPlanStatsTests
{
    private const int InChannels = 8;
    private const int OutChannels = 16;
    private const int Kernel = 3;

    private readonly ITestOutputHelper _output;

    public CudnnConvPlanStatsTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(3, 4)]
    [InlineData(64, 64)]
    [InlineData(65, 128)]
    [InlineData(25_682, 32_768)]
    public void LengthBucket_IsTheNextPowerOfTwo(long length, long bucket) =>
        Assert.Equal(bucket, CudnnConv.LengthBucket(length));

    [Fact]
    public void AudioConv1d_PlansEachLengthFromItsBucket()
    {
        if (!CudaAvailable())
        {
            return;
        }
        WithBuckets(true, () =>
        {
            using CudaBackend cuda = new CudaBackend(0, PtxDir());
            using Tensor weight = Random(new TensorShape(OutChannels, InChannels, Kernel), seed: 1);
            using Tensor bias = Random(new TensorShape(OutChannels), seed: 2);
            Convolve(cuda, weight, bias, length: 64).Dispose();
            Convolve(cuda, weight, bias, length: 64).Dispose();
            Convolve(cuda, weight, bias, length: 96).Dispose();

            CudnnConvPlanStats stats = cuda.CudnnConvPlanStats;
            string families = cuda.DescribeCudnnConvPlanFamilies();
            _output.WriteLine(stats.ToString());
            _output.WriteLine(families);
            Assert.True(cuda.CudnnConvEngaged, "the audio conv did not take the cuDNN route");
            Assert.Equal(3, stats.Executions);
            Assert.Equal(2, stats.PlanBuilds);
            Assert.Equal(2, stats.CachedPlans);
            // Lengths 64 and 96 sit in buckets 64 and 128: one heuristic at each reference length, no other.
            Assert.Equal(2, stats.ReferenceBuilds);
            Assert.Equal(2, stats.BucketPlanBuilds);
            Assert.Equal(0, stats.BucketFallbacks);
            Assert.True(stats.BuildMs + 1e-6 >= stats.GraphMs + stats.HeuristicMs + stats.FinalizeMs,
                $"the phases ({stats.GraphMs} + {stats.HeuristicMs} + {stats.FinalizeMs} ms) exceed the build ({stats.BuildMs} ms)");
            string[] lines = families.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            Assert.Single(lines);
            Assert.Contains("plans=2 (from bucket 2) references=2", lines[0], StringComparison.Ordinal);
        });
    }

    [Fact]
    public void AudioConv1d_WithoutBuckets_RunsTheHeuristicPerLength()
    {
        if (!CudaAvailable())
        {
            return;
        }
        WithBuckets(false, () =>
        {
            using CudaBackend cuda = new CudaBackend(0, PtxDir());
            using Tensor weight = Random(new TensorShape(OutChannels, InChannels, Kernel), seed: 1);
            using Tensor bias = Random(new TensorShape(OutChannels), seed: 2);
            Convolve(cuda, weight, bias, length: 100).Dispose();
            Convolve(cuda, weight, bias, length: 120).Dispose();

            CudnnConvPlanStats stats = cuda.CudnnConvPlanStats;
            _output.WriteLine(stats.ToString());
            Assert.Equal(2, stats.PlanBuilds);
            Assert.Equal(0, stats.ReferenceBuilds);
            Assert.Equal(0, stats.BucketPlanBuilds);
            Assert.True(stats.ConfigsTried >= 2, $"each heuristic build finalizes at least one config, saw {stats.ConfigsTried}");
        });
    }

    /// <summary>Two backends see the same two lengths of one bucket in opposite orders and produce the same bytes for each
    /// length: the bucket's engine comes from its reference length, not from whichever length arrived first.</summary>
    [Fact]
    public void AudioConv1d_OutputDoesNotDependOnLengthOrder()
    {
        if (!CudaAvailable())
        {
            return;
        }
        WithBuckets(true, () =>
        {
            using Tensor weight = Random(new TensorShape(OutChannels, InChannels, Kernel), seed: 1);
            using Tensor bias = Random(new TensorShape(OutChannels), seed: 2);
            float[] forward100, forward120, reverse100, reverse120;
            using (CudaBackend cuda = new CudaBackend(0, PtxDir()))
            {
                forward100 = Read(Convolve(cuda, weight, bias, length: 100));
                forward120 = Read(Convolve(cuda, weight, bias, length: 120));
                Assert.Equal(1, cuda.CudnnConvPlanStats.ReferenceBuilds);
            }
            using (CudaBackend cuda = new CudaBackend(0, PtxDir()))
            {
                reverse120 = Read(Convolve(cuda, weight, bias, length: 120));
                reverse100 = Read(Convolve(cuda, weight, bias, length: 100));
                Assert.Equal(1, cuda.CudnnConvPlanStats.ReferenceBuilds);
            }
            Assert.Equal(forward100, reverse100);
            Assert.Equal(forward120, reverse120);
        });
    }

    /// <summary>A bucket's choice computes the same convolution as each length's own heuristic: TF32-close, and byte-equal
    /// whenever the heuristic would have picked the same engine and knobs (reported, not asserted).</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AudioConv_BucketMatchesPerLengthHeuristic(bool transposed)
    {
        if (!CudaAvailable())
        {
            return;
        }
        // Transposed conv weight is [C_in, C_out, K]; stride 2, kernel 4, pads 1/1 gives tOut = 2·tIn.
        using Tensor weight = transposed
            ? Random(new TensorShape(InChannels, OutChannels, 4), seed: 3)
            : Random(new TensorShape(OutChannels, InChannels, Kernel), seed: 1);
        using Tensor bias = Random(new TensorShape(OutChannels), seed: 2);
        float[] bucketed = [], perLength = [];
        WithBuckets(true, () =>
        {
            using CudaBackend cuda = new CudaBackend(0, PtxDir());
            Convolve(cuda, weight, bias, length: 100, transposed).Dispose();
            bucketed = Read(Convolve(cuda, weight, bias, length: 77, transposed));
            Assert.Equal(1, cuda.CudnnConvPlanStats.ReferenceBuilds);
            Assert.Equal(2, cuda.CudnnConvPlanStats.BucketPlanBuilds);
        });
        WithBuckets(false, () =>
        {
            using CudaBackend cuda = new CudaBackend(0, PtxDir());
            perLength = Read(Convolve(cuda, weight, bias, length: 77, transposed));
        });
        Assert.Equal(perLength.Length, bucketed.Length);
        float maxDiff = 0f, maxMag = 0f;
        for (int i = 0; i < perLength.Length; i++)
        {
            maxDiff = MathF.Max(maxDiff, MathF.Abs(perLength[i] - bucketed[i]));
            maxMag = MathF.Max(maxMag, MathF.Abs(perLength[i]));
        }
        _output.WriteLine($"{(transposed ? "transposed" : "forward")}: max|Δ| {maxDiff:E3} of max|y| {maxMag:F3}, "
            + $"byte-identical {perLength.AsSpan().SequenceEqual(bucketed)}");
        Assert.True(maxDiff <= 5e-3f * maxMag, $"bucketed plan differs from the per-length plan by {maxDiff:E3} (max|y| {maxMag:F3})");
    }

    private bool CudaAvailable()
    {
        if (CudaContext.IsAvailable())
        {
            return true;
        }
        _output.WriteLine($"SKIPPED: CUDA unavailable ({CudaContext.LastUnavailableReason})");
        return false;
    }

    private static void WithBuckets(bool on, Action body)
    {
        KnobStore.Set(EngineKnobs.AudioConvCudnn, true);
        KnobStore.Set(EngineKnobs.AudioConvLengthBuckets, on);
        try
        {
            body();
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.AudioConvCudnn);
            KnobStore.Clear(EngineKnobs.AudioConvLengthBuckets);
        }
    }

    /// <summary>One conv of input length <paramref name="length"/>: forward k3 pad 1/1, or transposed k4 stride 2 pad 1/1.</summary>
    private static Tensor Convolve(CudaBackend cuda, Tensor weight, Tensor bias, int length, bool transposed = false)
    {
        using Tensor input = Random(new TensorShape(1, InChannels, length), seed: length);
        Tensor output = new Tensor(new TensorShape(1, OutChannels, transposed ? 2 * length : length), DType.F32);
        if (transposed)
            ((IBackend)cuda).ConvTranspose1d(output, input, weight, bias, stride: 2, padLeft: 1, padRight: 1, dilation: 1, groups: 1);
        else
            ((IBackend)cuda).Conv1d(output, input, weight, bias, stride: 1, padLeft: 1, padRight: 1, dilation: 1, groups: 1);
        cuda.Sync();
        _ = *(float*)output.DataPointer;
        return output;
    }

    private static float[] Read(Tensor tensor)
    {
        using (tensor)
        {
            return tensor.AsSpan<float>().ToArray();
        }
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
