using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Guards the phase-boundary operation that releases fused-attention plans/workspaces while preserving
/// resident model weights. Z-Image uses this before a same-device Qwen prompt-cache miss.</summary>
[Collection("CudaSerial")]
public sealed unsafe class AttentionExecutionCacheReleaseTests
{
    private readonly ITestOutputHelper _output;

    public AttentionExecutionCacheReleaseTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void ReleaseAttentionExecutionCache_PreservesWeights_AndRecreatesCudnnSession()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }
        CudnnRuntime.EnsureProbed();
        if (!CudnnRuntime.SupportsSdpa)
        {
            _output.WriteLine($"SKIPPED: {CudnnRuntime.Reason}");
            return;
        }

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        const int Heads = 2, Sequence = 32, HeadDim = 64;
        float scale = 1f / MathF.Sqrt(HeadDim);
        string? previous = Environment.GetEnvironmentVariable("HARTSY_SDPA_CUDNN");
        Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", "1");
        try
        {
            using CudaBackend backend = new(0, ptxDir);
            using Tensor sentinelWeight = RandomTensor(new TensorShape(4096), 17);
            using Tensor q = RandomTensor(new TensorShape(1, Heads, Sequence, HeadDim), 23);
            using Tensor k = RandomTensor(new TensorShape(1, Heads, Sequence, HeadDim), 29);
            using Tensor v = RandomTensor(new TensorShape(1, Heads, Sequence, HeadDim), 31);
            using Tensor first = new(q.Shape, DType.F32);
            using Tensor alternateScale = new(q.Shape, DType.F32);
            using Tensor second = new(q.Shape, DType.F32);

            backend.PreloadWeights(new[] { sentinelWeight });
            Assert.True(GpuTransferHelper.IsWeightCached(sentinelWeight));

            backend.ScaledDotProductAttention(first, q, k, v, null, scale, allowF16: true);
            // A distinct exact scale is a distinct PlanKey. Releasing this session must walk both cached plans,
            // not stop after the first independently owned descriptor/scalar/workspace.
            backend.ScaledDotProductAttention(alternateScale, q, k, v, null, scale * 0.75f, allowF16: true);
            backend.Sync();
            Assert.Equal(2, backend.CudnnSdpaExecutionCount);
            Assert.Equal(1, backend.CudnnSdpaSessionGeneration);
            Assert.Equal(0, backend.CudnnSdpaDisposedSessionCount);

            backend.ReleaseAttentionExecutionCache();
            Assert.True(GpuTransferHelper.IsWeightCached(sentinelWeight));
            Assert.Equal(2, backend.CudnnSdpaExecutionCount);
            Assert.Equal(1, backend.CudnnSdpaSessionGeneration);
            Assert.Equal(1, backend.CudnnSdpaDisposedSessionCount);

            backend.ScaledDotProductAttention(second, q, k, v, null, scale, allowF16: true);
            backend.Sync();
            Assert.Equal(3, backend.CudnnSdpaExecutionCount);
            Assert.Equal(2, backend.CudnnSdpaSessionGeneration);
            Assert.Equal(1, backend.CudnnSdpaDisposedSessionCount);
            Assert.True(GpuTransferHelper.IsWeightCached(sentinelWeight));

            float maxDifference = MaxDifference(first, second);
            _output.WriteLine($"recreated cuDNN SDPA session: max repeat difference={maxDifference:E3}");
            Assert.True(maxDifference <= 1e-5f,
                $"Recreated cuDNN SDPA session changed a same-input result by {maxDifference:E3}.");

            // The broader OOM/model-switch sweep must also destroy the recreated session and continue far enough
            // to evict ordinary weights; a teardown failure must not strand the rest of the cache walk.
            backend.FreeAllDeviceMemory();
            Assert.Equal(2, backend.CudnnSdpaDisposedSessionCount);
            Assert.False(GpuTransferHelper.IsWeightCached(sentinelWeight));
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", previous);
        }
    }

    private static Tensor RandomTensor(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        float* values = (float*)tensor.DataPointer;
        Random random = new(seed);
        for (long i = 0; i < tensor.ElementCount; i++)
            values[i] = (float)(random.NextDouble() - 0.5);
        return tensor;
    }

    private static float MaxDifference(Tensor expected, Tensor actual)
    {
        float* expectedValues = (float*)expected.DataPointer;
        float* actualValues = (float*)actual.DataPointer;
        float maxDifference = 0f;
        for (long i = 0; i < expected.ElementCount; i++)
        {
            Assert.True(float.IsFinite(expectedValues[i]), $"First result is non-finite at {i}.");
            Assert.True(float.IsFinite(actualValues[i]), $"Second result is non-finite at {i}.");
            maxDifference = MathF.Max(maxDifference, MathF.Abs(expectedValues[i] - actualValues[i]));
        }
        return maxDifference;
    }
}
