using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>`LtxVideo25NaBlock.Modulate` is a host loop over `x.DataPointer` applied to a tensor a CUDA op just
/// wrote, whose result is then consumed by another CUDA op. This pins whether such a host write is actually
/// observed downstream, or silently discarded because the activation cache still holds the pre-write device
/// buffer — the failure mode would be AdaLN modulation never reaching attention or the MLP.</summary>
[Collection("CudaSerial")]
public sealed unsafe class Ltx25HostWriteAfterDeviceOpTests
{
    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Filled(TensorShape shape, float value)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < shape.ElementCount; i++) p[i] = value;
        return t;
    }

    [Fact]
    public void HostWriteBetweenTwoDeviceOpsIsVisibleToTheSecond()
    {
        const int rows = 8, dim = 16;
        using CudaBackend cuda = new CudaBackend(0, PtxDir());

        using Tensor src = Filled(new TensorShape(rows, dim), 1f);
        using Tensor normW = Filled(new TensorShape(dim), 1f);
        using Tensor normed = new Tensor(new TensorShape(rows, dim), DType.F32);

        // 1. device op writes `normed` (result lands in its activation cache)
        cuda.RmsNorm(normed, src, normW, 1e-6f);
        cuda.Sync();

        // 2. host op mutates it in place, exactly as Modulate does
        float* p = (float*)normed.DataPointer;
        for (long i = 0; i < normed.ElementCount; i++) p[i] += 1000f;

        // 3. device op consumes it
        using Tensor zeros = Filled(new TensorShape(rows, dim), 0f);
        using Tensor result = new Tensor(new TensorShape(rows, dim), DType.F32);
        cuda.Add(result, normed, zeros);
        cuda.Sync();

        float* r = (float*)result.DataPointer;
        Assert.True(r[0] > 900f,
            $"host write between two device ops was discarded: downstream op saw {r[0]}, expected ~1001. "
            + "The activation cache still holds the pre-write device buffer.");
    }
}
