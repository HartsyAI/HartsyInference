using SharpInference.Cpu.Kernels;
using SharpInference.Core.Tensors;
using SharpInference.Core.Backends;
using Xunit;

namespace SharpInference.Cpu.Tests;

public sealed unsafe class MatMulKernelTests
{
    private const float Tolerance = 1e-5f;

    [Fact]
    public void MatMul_Identity_ReturnsInput()
    {
        using Tensor a = new Tensor(new TensorShape(2, 2), DType.F32);
        using Tensor b = new Tensor(new TensorShape(2, 2), DType.F32);
        using Tensor output = new Tensor(new TensorShape(2, 2), DType.F32);

        Span<float> aSpan = a.AsSpan<float>();
        aSpan[0] = 1.0f; aSpan[1] = 0.0f;
        aSpan[2] = 0.0f; aSpan[3] = 1.0f;

        Span<float> bSpan = b.AsSpan<float>();
        bSpan[0] = 5.0f; bSpan[1] = 6.0f;
        bSpan[2] = 7.0f; bSpan[3] = 8.0f;

        MatMulKernels.MatMul(output, a, b);

        Span<float> outSpan = output.AsSpan<float>();
        Assert.Equal(5.0f, outSpan[0], Tolerance);
        Assert.Equal(6.0f, outSpan[1], Tolerance);
        Assert.Equal(7.0f, outSpan[2], Tolerance);
        Assert.Equal(8.0f, outSpan[3], Tolerance);
    }

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
    public void MatMul_1x1_ScalarMultiply()
    {
        using Tensor a = new Tensor(new TensorShape(1, 1), DType.F32);
        using Tensor b = new Tensor(new TensorShape(1, 1), DType.F32);
        using Tensor output = new Tensor(new TensorShape(1, 1), DType.F32);

        a.AsSpan<float>()[0] = 3.0f;
        b.AsSpan<float>()[0] = 4.0f;

        MatMulKernels.MatMul(output, a, b);

        Assert.Equal(12.0f, output.AsSpan<float>()[0], Tolerance);
    }

    [Fact]
    public void MatMul_LargerMatrix_32x32()
    {
        using Tensor identity = new Tensor(new TensorShape(32, 32), DType.F32);
        using Tensor b = new Tensor(new TensorShape(32, 32), DType.F32);
        using Tensor output = new Tensor(new TensorShape(32, 32), DType.F32);

        Span<float> idSpan = identity.AsSpan<float>();
        for (int i = 0; i < 32; i++)
        {
            idSpan[i * 32 + i] = 1.0f;
        }

        Span<float> bSpan = b.AsSpan<float>();
        for (int i = 0; i < 32 * 32; i++)
        {
            bSpan[i] = i * 0.1f;
        }

        MatMulKernels.MatMul(output, identity, b);

        Span<float> outSpan = output.AsSpan<float>();
        for (int i = 0; i < 32 * 32; i++)
        {
            Assert.Equal(i * 0.1f, outSpan[i], Tolerance);
        }
    }

    [Fact]
    public void MatMul_ZeroMatrix_ReturnsZero()
    {
        using Tensor a = new Tensor(new TensorShape(3, 3), DType.F32);
        using Tensor b = new Tensor(new TensorShape(3, 3), DType.F32);
        using Tensor output = new Tensor(new TensorShape(3, 3), DType.F32);

        // a is already zeroed by constructor
        Span<float> bSpan = b.AsSpan<float>();
        for (int i = 0; i < 9; i++)
        {
            bSpan[i] = (i + 1) * 1.0f;
        }

        MatMulKernels.MatMul(output, a, b);

        Span<float> outSpan = output.AsSpan<float>();
        for (int i = 0; i < 9; i++)
        {
            Assert.Equal(0.0f, outSpan[i], Tolerance);
        }
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
