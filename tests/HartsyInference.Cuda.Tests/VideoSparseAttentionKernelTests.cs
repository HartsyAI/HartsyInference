using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>SM86/SM89 parity and residency gates for MiniMax-H3's persistent CUDA VSA session.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class VideoSparseAttentionKernelTests
{
    private readonly ITestOutputHelper _output;

    /// <summary>Creates the GPU integration fixture.</summary>
    public VideoSparseAttentionKernelTests(ITestOutputHelper output) => _output = output;

    /// <summary>Both published 64-token routing profiles match the deterministic eager oracle and do not read
    /// device tensors back while executing a layer.</summary>
    [Theory]
    [InlineData(0, VideoSparseAttentionProfileKind.ComfySol64V1)]
    [InlineData(0, VideoSparseAttentionProfileKind.FastVideoVsa64V1)]
    [InlineData(1, VideoSparseAttentionProfileKind.ComfySol64V1)]
    [InlineData(1, VideoSparseAttentionProfileKind.FastVideoVsa64V1)]
    public void Execute_MatchesReferenceWithoutHotLoopDeviceReadback(int device,
        VideoSparseAttentionProfileKind profile)
    {
        if (!CudaContext.IsAvailable() || CudaContext.GetDeviceCount() <= device)
        {
            _output.WriteLine($"SKIPPED: CUDA device {device} unavailable");
            return;
        }
        string ptxDir = PtxDir();
        if (!File.Exists(Path.Combine(ptxDir, "h3_vsa.ptx")))
        {
            _output.WriteLine("SKIPPED: h3_vsa.ptx not built");
            return;
        }

        const int sequence = 8;
        VideoSparseAttentionPlan plan = new VideoSparseAttentionPlan
        {
            Profile = profile,
            SequenceLength = sequence,
            BlockOffsets = [0, 1, 2, 3, 4, 5, 6, 7, 8],
            SourceIndices = [0, 1, 2, 3, 4, 5, 6, 7],
            SegmentClasses = [0, 1, 2, 3, 3, 3, 3, 3],
            PrefixSinkBlocks = 3,
            KeepFraction = 0.2f,
        };
        TensorShape shape = new TensorShape(1, 56, sequence, 128);
        using Tensor query = Random(shape, 101, 0.08f);
        using Tensor key = Random(shape, 102, 0.08f);
        using Tensor value = Random(shape, 103, 0.4f);
        using Tensor gate = Random(shape, 104, 0.3f);
        using Tensor expected = new Tensor(shape, DType.F32);
        using (VideoSparseAttentionReferenceSession reference = new VideoSparseAttentionReferenceSession(plan))
        {
            reference.Execute(expected, query, key, value, gate);
        }

        using Tensor actual = new Tensor(shape, DType.F32);
        using CudaBackend cuda = new CudaBackend(device, ptxDir);
        Assert.True(cuda.SupportsVideoSparseAttention);
        using IVideoSparseAttentionSession session = cuda.CreateVideoSparseAttentionSession(plan);
        cuda.ResetD2hSyncCount();
        session.Execute(actual, query, key, value, gate);
        Assert.Equal(0, cuda.GetD2hSyncCount());
        cuda.Sync();

        float maxError = MaximumError(expected, actual);
        _output.WriteLine($"device={device} profile={profile} maxError={maxError:G9}");
        Assert.InRange(maxError, 0f, 2e-4f);
    }

    /// <summary>The CUDA backend does not advertise VSA without the dedicated PTX artifact.</summary>
    [Fact]
    public void Capability_IsFalseWhenKernelArtifactIsAbsent()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }
        using CudaBackend cuda = new CudaBackend(0, ptxDir: null);
        Assert.False(cuda.SupportsVideoSparseAttention);
    }

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
        {
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path,
                "src", "HartsyInference.Cuda", "Ptx");
        }
        return dir;
    }

    private static Tensor Random(TensorShape shape, int seed, float amplitude)
    {
        Tensor tensor = new Tensor(shape, DType.F32);
        float* values = (float*)tensor.DataPointer;
        Random random = new Random(seed);
        for (long i = 0; i < tensor.ElementCount; i++)
        {
            values[i] = ((float)random.NextDouble() * 2f - 1f) * amplitude;
        }
        return tensor;
    }

    private static float MaximumError(Tensor expected, Tensor actual)
    {
        float* left = (float*)expected.DataPointer;
        float* right = (float*)actual.DataPointer;
        float maximum = 0f;
        for (long i = 0; i < expected.ElementCount; i++)
        {
            maximum = MathF.Max(maximum, MathF.Abs(left[i] - right[i]));
        }
        return maximum;
    }
}
