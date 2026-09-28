using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>The CUDA <c>wan_vae_dup_up3d</c> kernel against the managed <see cref="IBackend.DupUp3dVae"/> default, on
/// shapes where H != W and T > 1 so a transposed axis, a mis-strided frame or a wrong dropped-frame offset cannot pass.
/// The op is a pure gather, so this is bit-equality. Skips cleanly when CUDA is unavailable.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class WanVaeDupUp3dKernelTests
{
    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    /// <summary>The Wan 2.2 decoder's shortcut geometries: temporal+spatial with and without the first-chunk drop, and
    /// spatial-only; channel ratios both repeating (outC·factor > inC) and not.</summary>
    [Theory]
    [InlineData(8, 4, 2, 1)]
    [InlineData(8, 4, 2, 0)]
    [InlineData(16, 8, 1, 0)]
    [InlineData(4, 8, 2, 1)]
    public void DupUp3d_CudaMatchesManagedReference(int inC, int outC, int factorT, int dropT)
    {
        if (!CudaContext.IsAvailable()) return;
        const int t = 3, h = 5, w = 7, factorS = 2;
        using Tensor input = new Tensor(new TensorShape([1L, inC, t, h, w]), DType.F32);
        float* ip = (float*)input.DataPointer;
        for (long i = 0; i < input.ElementCount; i++) ip[i] = i;
        TensorShape outShape = new([1L, outC, t * factorT - dropT, h * factorS, w * factorS]);
        using Tensor expected = new Tensor(outShape, DType.F32);
        ((IBackend)new CpuBackend()).DupUp3dVae(expected, input, factorT, factorS, dropT);

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor actual = new Tensor(outShape, DType.F32);
        cuda.DupUp3dVae(actual, input, factorT, factorS, dropT);
        cuda.Sync();

        float* e = (float*)expected.DataPointer;
        float* a = (float*)actual.DataPointer;
        for (long i = 0; i < expected.ElementCount; i++)
            Assert.True(e[i] == a[i], $"mismatch at {i}: cuda {a[i]} vs managed {e[i]}");
    }
}
