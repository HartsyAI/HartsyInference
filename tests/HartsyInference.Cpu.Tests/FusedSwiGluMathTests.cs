using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Cpu.Tests;

/// <summary>Math gate for the fused-FFN forward pattern (Ideogram4Block.ForwardSwiGlu under
/// HARTSY_FUSED_FFN — INFERENCE_ACCEL_GRIND §H3.2): one Linear over row-concatenated [w1; w3] followed by
/// contiguous SliceLastDim splits must equal the two separate Linears EXACTLY in F32 (each output element
/// is the identical dot product — row concat only extends the GEMM's N dimension).</summary>
public sealed unsafe class FusedSwiGluMathTests
{
    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        long n = shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    [Fact]
    public void FusedLinearPlusSlices_EqualsSeparateLinears_F32()
    {
        const int batch = 2, seq = 3, hidden = 16, inner = 8;
        using Tensor x = Random(new TensorShape(batch, seq, hidden), 11);
        using Tensor w1 = Random(new TensorShape(inner, hidden), 12);
        using Tensor w3 = Random(new TensorShape(inner, hidden), 13);

        using CpuBackend cpu = new CpuBackend();
        IBackend backend = cpu;

        // Reference: two separate projections.
        using Tensor gateRef = new Tensor(new TensorShape(batch, seq, inner), DType.F32);
        using Tensor upRef = new Tensor(new TensorShape(batch, seq, inner), DType.F32);
        backend.Linear(gateRef, x, w1, null);
        backend.Linear(upRef, x, w3, null);

        // Fused: manual row-concat (the ConcatRowsHost layout: [w1; w3]) + one Linear + slices.
        using Tensor w13 = new Tensor(new TensorShape(2 * inner, hidden), DType.F32);
        long half = (long)inner * hidden;
        new Span<float>((float*)w1.DataPointer, (int)half).CopyTo(new Span<float>((float*)w13.DataPointer, (int)half));
        new Span<float>((float*)w3.DataPointer, (int)half).CopyTo(new Span<float>((float*)w13.DataPointer + half, (int)half));

        using Tensor both = new Tensor(new TensorShape(batch, seq, 2 * inner), DType.F32);
        backend.Linear(both, x, w13, null);
        using Tensor gateFused = new Tensor(new TensorShape(batch, seq, inner), DType.F32);
        using Tensor upFused = new Tensor(new TensorShape(batch, seq, inner), DType.F32);
        backend.SliceLastDim(gateFused, both, 0);
        backend.SliceLastDim(upFused, both, inner);

        float* gr = (float*)gateRef.DataPointer;
        float* gf = (float*)gateFused.DataPointer;
        float* ur = (float*)upRef.DataPointer;
        float* uf = (float*)upFused.DataPointer;
        for (long i = 0; i < (long)batch * seq * inner; i++)
        {
            Assert.Equal(gr[i], gf[i]);
            Assert.Equal(ur[i], uf[i]);
        }
    }
}
