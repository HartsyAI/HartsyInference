using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>CUDA <see cref="IBackend.UpsampleNearest2DToSize"/> at the one-short size an odd SDXL latent produces,
/// against the nearest-neighbour rule directly (source row <c>oh / 2</c>). A pure gather, so bit-equality. Skips
/// cleanly without CUDA.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class UpsampleNearestToSizeKernelTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OneShortOutput_MatchesHostDefault(bool f16)
    {
        DType dtype = f16 ? DType.F16 : DType.F32;
        if (!CudaContext.IsAvailable()) return;
        string ptx = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptx)) ptx = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        using Tensor input = new(new TensorShape(1, 4, 23, 40), dtype);
        int elem = (int)dtype.ComputeByteCount(1);
        for (long i = 0; i < input.ElementCount; i++)
        {
            if (dtype == DType.F32) ((float*)input.DataPointer)[i] = i;
            else ((Half*)input.DataPointer)[i] = (Half)(i % 1024);
        }
        using Tensor expected = new(new TensorShape(1, 4, 45, 79), dtype);
        byte* src = (byte*)input.DataPointer, dst = (byte*)expected.DataPointer;
        for (int plane = 0; plane < 4; plane++)
            for (int y = 0; y < 45; y++)
                for (int x = 0; x < 79; x++)
                    Buffer.MemoryCopy(src + ((plane * 23 + y / 2) * 40 + x / 2) * elem, dst + ((plane * 45 + y) * 79 + x) * elem, elem, elem);

        using CudaBackend cuda = new(0, ptx);
        using Tensor actual = new(new TensorShape(1, 4, 45, 79), dtype);
        ((IBackend)cuda).UpsampleNearest2DToSize(actual, input, 2);
        cuda.Sync();

        Assert.True(new ReadOnlySpan<byte>(expected.DataPointer, (int)expected.ElementCount * elem)
            .SequenceEqual(new ReadOnlySpan<byte>(actual.DataPointer, (int)actual.ElementCount * elem)));
    }
}
