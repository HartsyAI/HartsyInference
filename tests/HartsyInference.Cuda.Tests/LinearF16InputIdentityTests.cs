using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>A Linear of F16 weights casts an F32 input to F16 itself, with <see cref="IBackend.CastToF16"/>'s kernel, before the
/// same F16 GEMM: so one shared cast feeding several such Linears (Wan's q/k/v) writes their F32 outputs bit for bit.</summary>
[Collection("CudaSerial")]
public sealed unsafe class LinearF16InputIdentityTests
{
    private readonly ITestOutputHelper _out;
    public LinearF16InputIdentityTests(ITestOutputHelper output) => _out = output;

    private static CudaBackend Make()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return new CudaBackend(0, ptxDir);
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(512, 3072, 3072, true)]
    [InlineData(300, 384, 256, false)]
    public void APreCastF16InputWritesTheF32InputsOutputBitForBit(int m, int k, int n, bool bias)
    {
        if (!CudaContext.IsAvailable()) { _out.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = Make();
        IBackend b = backend;
        using Tensor x = Seeded(new TensorShape(m, k), 1, 4f);
        using Tensor w = Half(Seeded(new TensorShape(n, k), 2, 0.05f));
        using Tensor? biasT = bias ? Half(Seeded(new TensorShape(n), 3, 0.5f)) : null;

        using Tensor fromF32 = new(new TensorShape(m, n), DType.F32);
        b.Linear(fromF32, x, w, biasT);
        using Tensor x16 = new(x.Shape, DType.F16);
        b.CastToF16(x16, x);
        using Tensor fromF16 = new(new TensorShape(m, n), DType.F32);
        b.Linear(fromF16, x16, w, biasT);
        backend.Sync();

        ReadOnlySpan<float> a = new(fromF32.DataPointer, m * n), c = new(fromF16.DataPointer, m * n);
        int differing = 0;
        for (int i = 0; i < a.Length; i++)
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(c[i])) differing++;
        Assert.True(differing == 0, $"{differing} of {a.Length} outputs differ between the F32 input and its F16 cast.");
    }

    private static Tensor Seeded(TensorShape shape, uint seed, float scale)
    {
        Tensor t = new(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        uint state = seed * 2654435761u + 0x9E3779B9u;
        for (long i = 0; i < t.ElementCount; i++)
        {
            state = state * 1664525u + 1013904223u;
            p[i] = (((state >> 8) & 0xFFFF) / 32768f - 1f) * scale;
        }
        return t;
    }

    private static Tensor Half(Tensor f32)
    {
        Tensor h = f32.CastTo(DType.F16);
        f32.Dispose();
        return h;
    }
}
