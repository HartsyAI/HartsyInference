using System.Reflection;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Guards the Z-Image phase-boundary contract for its two explicitly preloaded RoPE tables.</summary>
[Collection("CudaSerial")]
public sealed unsafe class ZImageRopeGpuTableLifecycleTests
{
    private readonly ITestOutputHelper _output;

    public ZImageRopeGpuTableLifecycleTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void ReleaseGpuTables_EvictsBothWeights_AndApplyGpuReuploadsEquivalentTables()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path,
                "src", "HartsyInference.Cuda", "Ptx");

        const int Sequence = 4;
        const int Heads = 2;
        const int HeadDim = 8;
        ZImageRope rope = new([2, 2, 4]);
        using CudaBackend backend = new(0, ptxDir);
        using Tensor positions = PositionIds(Sequence);
        using Tensor qFirst = Input(new TensorShape(1, Sequence, Heads, HeadDim), offset: 0.25f);
        using Tensor kFirst = Input(new TensorShape(1, Sequence, Heads, HeadDim), offset: -0.5f);
        using Tensor qSecond = Input(new TensorShape(1, Sequence, Heads, HeadDim), offset: 0.25f);
        using Tensor kSecond = Input(new TensorShape(1, Sequence, Heads, HeadDim), offset: -0.5f);

        rope.Precompute(positions);
        try
        {
            rope.ApplyGpu(backend, qFirst, kFirst, Heads);
            backend.Sync();

            Tensor firstCos = GpuTable(rope, "_gpuCos");
            Tensor firstSin = GpuTable(rope, "_gpuSin");
            Assert.True(GpuTransferHelper.IsWeightCached(firstCos));
            Assert.True(GpuTransferHelper.IsWeightCached(firstSin));

            float[] expectedQ = Snapshot(qFirst);
            float[] expectedK = Snapshot(kFirst);
            long tableBytes = 2L * Sequence * HeadDim * sizeof(float);
            long cachedBeforeRelease = backend.GetGpuCacheStats().cachedBytes;

            rope.ReleaseGpuTables(backend);

            long cachedAfterRelease = backend.GetGpuCacheStats().cachedBytes;
            Assert.False(GpuTransferHelper.IsWeightCached(firstCos));
            Assert.False(GpuTransferHelper.IsWeightCached(firstSin));
            Assert.Equal(tableBytes, cachedBeforeRelease - cachedAfterRelease);

            rope.ApplyGpu(backend, qSecond, kSecond, Heads);
            backend.Sync();

            Tensor secondCos = GpuTable(rope, "_gpuCos");
            Tensor secondSin = GpuTable(rope, "_gpuSin");
            Assert.NotSame(firstCos, secondCos);
            Assert.NotSame(firstSin, secondSin);
            Assert.True(GpuTransferHelper.IsWeightCached(secondCos));
            Assert.True(GpuTransferHelper.IsWeightCached(secondSin));
            Assert.Equal(expectedQ, Snapshot(qSecond));
            Assert.Equal(expectedK, Snapshot(kSecond));

            long cachedAfterReupload = backend.GetGpuCacheStats().cachedBytes;
            Assert.Equal(tableBytes, cachedAfterReupload - cachedAfterRelease);
            _output.WriteLine($"evicted and reuploaded two RoPE tables ({tableBytes} bytes total)");
        }
        finally
        {
            rope.ReleaseGpuTables(backend);
        }
    }

    private static Tensor GpuTable(ZImageRope rope, string fieldName)
    {
        FieldInfo field = typeof(ZImageRope).GetField(fieldName,
            BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException($"Missing ZImageRope field {fieldName}.");
        return Assert.IsType<Tensor>(field.GetValue(rope));
    }

    private static Tensor PositionIds(int sequence)
    {
        Tensor tensor = new(new TensorShape(sequence, 3), DType.F32);
        float* values = (float*)tensor.DataPointer;
        for (int token = 0; token < sequence; token++)
        {
            values[token * 3] = token;
            values[token * 3 + 1] = token % 2;
            values[token * 3 + 2] = token / 2;
        }
        return tensor;
    }

    private static Tensor Input(TensorShape shape, float offset)
    {
        Tensor tensor = new(shape, DType.F32);
        float* values = (float*)tensor.DataPointer;
        for (long i = 0; i < tensor.ElementCount; i++)
            values[i] = offset + i * 0.03125f;
        return tensor;
    }

    private static float[] Snapshot(Tensor tensor)
    {
        float[] result = new float[tensor.ElementCount];
        new ReadOnlySpan<float>(tensor.DataPointer, result.Length).CopyTo(result);
        return result;
    }
}
