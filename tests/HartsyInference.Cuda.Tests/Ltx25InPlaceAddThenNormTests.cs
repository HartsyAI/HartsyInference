using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>The LTX-2.5 decoder block diverges from the ComfyUI reference at its SECOND RmsNorm while the residual
/// feeding it still matches to 2e-4 — i.e. the value the norm consumes is not the value a host read of that tensor
/// reports. The block's shape is: norm → (attention) → in-place `Add(x, x, y)` → norm(x). This reproduces that
/// sequence minimally on both backends.</summary>
[Collection("CudaSerial")]
public sealed unsafe class Ltx25InPlaceAddThenNormTests
{
    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Rand(TensorShape shape, int seed, float scale = 0.2f)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        for (long i = 0; i < shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return t;
    }

    private static Tensor Sequence(IBackend backend, Tensor x0, Tensor y, Tensor w1, Tensor w2, long rows, int dim,
        bool readBackBetween)
    {
        using Tensor x = new Tensor(new TensorShape(rows, dim), DType.F32);
        long bytes = x.ElementCount * sizeof(float);
        Buffer.MemoryCopy((void*)x0.DataPointer, (void*)x.DataPointer, bytes, bytes);

        using (Tensor normed = new Tensor(new TensorShape(rows, dim), DType.F32))
            backend.RmsNorm(normed, x, w1, 1e-6f);
        backend.Add(x, x, y);
        // The harness tap reads the tensor here; whether that read changes what the next op sees is the question.
        if (readBackBetween) { float* probe = (float*)x.DataPointer; GC.KeepAlive(probe[0]); }
        Tensor normed2 = new Tensor(new TensorShape(rows, dim), DType.F32);
        backend.RmsNorm(normed2, x, w2, 1e-6f);
        return normed2;
    }

    [Theory]
    [InlineData(160, false)]
    [InlineData(160, true)]
    [InlineData(80, false)]
    public void InPlaceAddThenNormMatchesCpu(int rows, bool readBackBetween)
    {
        const int dim = 2048;
        using Tensor x0 = Rand(new TensorShape(rows, dim), seed: 1);
        using Tensor y = Rand(new TensorShape(rows, dim), seed: 2, scale: 0.07f);
        using Tensor w1 = Rand(new TensorShape(dim), seed: 3, scale: 0.1f);
        using Tensor w2 = Rand(new TensorShape(dim), seed: 4, scale: 0.1f);

        IBackend cpu = new CpuBackend();
        using Tensor expected = Sequence(cpu, x0, y, w1, w2, rows, dim, readBackBetween);

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor actual = Sequence(cuda, x0, y, w1, w2, rows, dim, readBackBetween);
        cuda.Sync();

        float* e = (float*)expected.DataPointer;
        float* a = (float*)actual.DataPointer;
        double num = 0, den = 0;
        for (long i = 0; i < expected.ElementCount; i++)
        {
            double d = (double)e[i] - a[i];
            num += d * d; den += (double)e[i] * e[i];
        }
        double rel = Math.Sqrt(num / Math.Max(den, 1e-30));
        Assert.True(rel < 1e-4, $"in-place Add then RmsNorm: CPU-vs-CUDA relL2 {rel:E3} "
            + $"(rows={rows}, readBackBetween={readBackBetween})");
    }
}
