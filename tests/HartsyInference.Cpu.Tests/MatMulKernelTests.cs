using HartsyInference.Cpu.Kernels;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Backends;
using Xunit;

namespace HartsyInference.Cpu.Tests;

public sealed unsafe class MatMulKernelTests
{
    private const float Tolerance = 1e-5f;

    [Fact]
    public void MatMul_2x3_Times_3x2()
    {
        using Tensor a = new Tensor(new TensorShape(2, 3), DType.F32);
        using Tensor b = new Tensor(new TensorShape(3, 2), DType.F32);
        using Tensor output = new Tensor(new TensorShape(2, 2), DType.F32);

        Span<float> aSpan = a.AsSpan<float>();
        // A = [[1,2,3],[4,5,6]]
        aSpan[0] = 1.0f; aSpan[1] = 2.0f; aSpan[2] = 3.0f;
        aSpan[3] = 4.0f; aSpan[4] = 5.0f; aSpan[5] = 6.0f;

        Span<float> bSpan = b.AsSpan<float>();
        // B = [[7,8],[9,10],[11,12]]
        bSpan[0] = 7.0f;  bSpan[1] = 8.0f;
        bSpan[2] = 9.0f;  bSpan[3] = 10.0f;
        bSpan[4] = 11.0f; bSpan[5] = 12.0f;

        MatMulKernels.MatMul(output, a, b);

        Span<float> outSpan = output.AsSpan<float>();
        // Expected: [[58,64],[139,154]]
        Assert.Equal(58.0f, outSpan[0], Tolerance);
        Assert.Equal(64.0f, outSpan[1], Tolerance);
        Assert.Equal(139.0f, outSpan[2], Tolerance);
        Assert.Equal(154.0f, outSpan[3], Tolerance);
    }

    [Fact]
    public void BatchedMatMul_TwoBatches()
    {
        using Tensor a = new Tensor(new TensorShape(2, 2, 3), DType.F32);
        using Tensor b = new Tensor(new TensorShape(2, 3, 2), DType.F32);
        using Tensor output = new Tensor(new TensorShape(2, 2, 2), DType.F32);

        Span<float> aSpan = a.AsSpan<float>();
        // Batch 0: A0 = [[1,2,3],[4,5,6]]
        aSpan[0] = 1.0f; aSpan[1] = 2.0f; aSpan[2] = 3.0f;
        aSpan[3] = 4.0f; aSpan[4] = 5.0f; aSpan[5] = 6.0f;
        // Batch 1: A1 = [[2,0,1],[0,3,0]]
        aSpan[6] = 2.0f;  aSpan[7] = 0.0f;  aSpan[8] = 1.0f;
        aSpan[9] = 0.0f;  aSpan[10] = 3.0f; aSpan[11] = 0.0f;

        Span<float> bSpan = b.AsSpan<float>();
        // Batch 0: B0 = [[7,8],[9,10],[11,12]]
        bSpan[0] = 7.0f;  bSpan[1] = 8.0f;
        bSpan[2] = 9.0f;  bSpan[3] = 10.0f;
        bSpan[4] = 11.0f; bSpan[5] = 12.0f;
        // Batch 1: B1 = [[1,0],[0,1],[2,3]]
        bSpan[6] = 1.0f;  bSpan[7] = 0.0f;
        bSpan[8] = 0.0f;  bSpan[9] = 1.0f;
        bSpan[10] = 2.0f; bSpan[11] = 3.0f;

        MatMulKernels.BatchedMatMul(output, a, b);

        Span<float> outSpan = output.AsSpan<float>();

        // Batch 0: A0 @ B0 = [[58,64],[139,154]]
        Assert.Equal(58.0f, outSpan[0], Tolerance);
        Assert.Equal(64.0f, outSpan[1], Tolerance);
        Assert.Equal(139.0f, outSpan[2], Tolerance);
        Assert.Equal(154.0f, outSpan[3], Tolerance);

        // Batch 1: A1 @ B1 = [[2*1+0*0+1*2, 2*0+0*1+1*3],[0*1+3*0+0*2, 0*0+3*1+0*3]]
        //                   = [[4, 3],[0, 3]]
        Assert.Equal(4.0f, outSpan[4], Tolerance);
        Assert.Equal(3.0f, outSpan[5], Tolerance);
        Assert.Equal(0.0f, outSpan[6], Tolerance);
        Assert.Equal(3.0f, outSpan[7], Tolerance);
    }
}
