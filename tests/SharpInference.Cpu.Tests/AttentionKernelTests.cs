using SharpInference.Cpu.Kernels;
using SharpInference.Core.Tensors;
using Xunit;

namespace SharpInference.Cpu.Tests;

public sealed unsafe class AttentionKernelTests
{
    private const float Tolerance = 1e-5f;

    [Fact]
    public void SDPA_IdentityAttention()
    {
        // Q=K=[1,1,2,4], V=[1,1,2,4]
        // When Q=K, Q@K^T has equal values in each row, so softmax gives uniform [0.5, 0.5]
        // Output = [0.5, 0.5] @ V = average of V rows
        using Tensor q = new Tensor(new TensorShape(1, 1, 2, 4), DType.F32);
        using Tensor k = new Tensor(new TensorShape(1, 1, 2, 4), DType.F32);
        using Tensor v = new Tensor(new TensorShape(1, 1, 2, 4), DType.F32);
        using Tensor output = new Tensor(new TensorShape(1, 1, 2, 4), DType.F32);

        Span<float> qSpan = q.AsSpan<float>();
        Span<float> kSpan = k.AsSpan<float>();

        // Set Q = K = [[1,0,1,0],[1,0,1,0]] (identical rows so Q@K^T is uniform per row)
        qSpan[0] = 1.0f; qSpan[1] = 0.0f; qSpan[2] = 1.0f; qSpan[3] = 0.0f;
        qSpan[4] = 1.0f; qSpan[5] = 0.0f; qSpan[6] = 1.0f; qSpan[7] = 0.0f;
        kSpan[0] = 1.0f; kSpan[1] = 0.0f; kSpan[2] = 1.0f; kSpan[3] = 0.0f;
        kSpan[4] = 1.0f; kSpan[5] = 0.0f; kSpan[6] = 1.0f; kSpan[7] = 0.0f;

        Span<float> vSpan = v.AsSpan<float>();
        // V = [[2,4,6,8],[10,20,30,40]]
        vSpan[0] = 2.0f;  vSpan[1] = 4.0f;  vSpan[2] = 6.0f;  vSpan[3] = 8.0f;
        vSpan[4] = 10.0f; vSpan[5] = 20.0f; vSpan[6] = 30.0f; vSpan[7] = 40.0f;

        float scale = 1.0f / MathF.Sqrt(4.0f);

        AttentionKernels.ScaledDotProductAttention(output, q, k, v, null, scale);

        Span<float> outSpan = output.AsSpan<float>();

        // Q@K^T: row0 dot row0 = 2, row0 dot row1 = 2 -> equal, softmax = [0.5, 0.5]
        // Output row 0 = 0.5*[2,4,6,8] + 0.5*[10,20,30,40] = [6,12,18,24]
        // Output row 1 = same (Q rows are identical)
        Assert.Equal(6.0f, outSpan[0], Tolerance);
        Assert.Equal(12.0f, outSpan[1], Tolerance);
        Assert.Equal(18.0f, outSpan[2], Tolerance);
        Assert.Equal(24.0f, outSpan[3], Tolerance);
        Assert.Equal(6.0f, outSpan[4], Tolerance);
        Assert.Equal(12.0f, outSpan[5], Tolerance);
        Assert.Equal(18.0f, outSpan[6], Tolerance);
        Assert.Equal(24.0f, outSpan[7], Tolerance);
    }

    [Fact]
    public void SDPA_SingleToken_ReturnsValue()
    {
        // Q,K,V all [1,1,1,4]. With seq=1, attention is trivially [1.0], so output = V
        using Tensor q = new Tensor(new TensorShape(1, 1, 1, 4), DType.F32);
        using Tensor k = new Tensor(new TensorShape(1, 1, 1, 4), DType.F32);
        using Tensor v = new Tensor(new TensorShape(1, 1, 1, 4), DType.F32);
        using Tensor output = new Tensor(new TensorShape(1, 1, 1, 4), DType.F32);

        Span<float> qSpan = q.AsSpan<float>();
        qSpan[0] = 1.0f; qSpan[1] = 2.0f; qSpan[2] = 3.0f; qSpan[3] = 4.0f;

        Span<float> kSpan = k.AsSpan<float>();
        kSpan[0] = 0.5f; kSpan[1] = 0.5f; kSpan[2] = 0.5f; kSpan[3] = 0.5f;

        Span<float> vSpan = v.AsSpan<float>();
        vSpan[0] = 10.0f; vSpan[1] = 20.0f; vSpan[2] = 30.0f; vSpan[3] = 40.0f;

        float scale = 1.0f / MathF.Sqrt(4.0f);

        AttentionKernels.ScaledDotProductAttention(output, q, k, v, null, scale);

        Span<float> outSpan = output.AsSpan<float>();
        Assert.Equal(10.0f, outSpan[0], Tolerance);
        Assert.Equal(20.0f, outSpan[1], Tolerance);
        Assert.Equal(30.0f, outSpan[2], Tolerance);
        Assert.Equal(40.0f, outSpan[3], Tolerance);
    }

    [Fact]
    public void SDPA_ScaleAffectsScores()
    {
        // Verify that different scale values produce different outputs
        using Tensor q = new Tensor(new TensorShape(1, 1, 2, 4), DType.F32);
        using Tensor k = new Tensor(new TensorShape(1, 1, 2, 4), DType.F32);
        using Tensor v = new Tensor(new TensorShape(1, 1, 2, 4), DType.F32);
        using Tensor output1 = new Tensor(new TensorShape(1, 1, 2, 4), DType.F32);
        using Tensor output2 = new Tensor(new TensorShape(1, 1, 2, 4), DType.F32);

        Span<float> qSpan = q.AsSpan<float>();
        // Q row 0 and row 1 are different so attention scores differ
        qSpan[0] = 1.0f; qSpan[1] = 0.0f; qSpan[2] = 0.0f; qSpan[3] = 0.0f;
        qSpan[4] = 0.0f; qSpan[5] = 1.0f; qSpan[6] = 0.0f; qSpan[7] = 0.0f;

        Span<float> kSpan = k.AsSpan<float>();
        kSpan[0] = 1.0f; kSpan[1] = 0.0f; kSpan[2] = 0.0f; kSpan[3] = 0.0f;
        kSpan[4] = 0.0f; kSpan[5] = 0.0f; kSpan[6] = 1.0f; kSpan[7] = 0.0f;

        Span<float> vSpan = v.AsSpan<float>();
        vSpan[0] = 1.0f; vSpan[1] = 2.0f; vSpan[2] = 3.0f; vSpan[3] = 4.0f;
        vSpan[4] = 5.0f; vSpan[5] = 6.0f; vSpan[6] = 7.0f; vSpan[7] = 8.0f;

        float scaleSmall = 0.1f;
        float scaleLarge = 10.0f;

        AttentionKernels.ScaledDotProductAttention(output1, q, k, v, null, scaleSmall);
        AttentionKernels.ScaledDotProductAttention(output2, q, k, v, null, scaleLarge);

        Span<float> out1 = output1.AsSpan<float>();
        Span<float> out2 = output2.AsSpan<float>();

        // With different scales, outputs should differ
        bool anyDifferent = false;
        for (int i = 0; i < 8; i++)
        {
            if (MathF.Abs(out1[i] - out2[i]) > Tolerance)
            {
                anyDifferent = true;
                break;
            }
        }

        Assert.True(anyDifferent, "Different scale values should produce different outputs.");
    }

    [Fact]
    public void SDPA_WithMask_MasksPositions()
    {
        // Q,K,V [1,1,2,4], mask [1,1,2,2] with mask[0,0,0,1]=-inf
        // After masking, token 0 should only attend to token 0 -> output[0,:] == V[0,:]
        using Tensor q = new Tensor(new TensorShape(1, 1, 2, 4), DType.F32);
        using Tensor k = new Tensor(new TensorShape(1, 1, 2, 4), DType.F32);
        using Tensor v = new Tensor(new TensorShape(1, 1, 2, 4), DType.F32);
        using Tensor mask = new Tensor(new TensorShape(1, 1, 2, 2), DType.F32);
        using Tensor output = new Tensor(new TensorShape(1, 1, 2, 4), DType.F32);

        Span<float> qSpan = q.AsSpan<float>();
        qSpan[0] = 1.0f; qSpan[1] = 0.0f; qSpan[2] = 0.0f; qSpan[3] = 0.0f;
        qSpan[4] = 0.0f; qSpan[5] = 1.0f; qSpan[6] = 0.0f; qSpan[7] = 0.0f;

        Span<float> kSpan = k.AsSpan<float>();
        kSpan[0] = 1.0f; kSpan[1] = 0.0f; kSpan[2] = 0.0f; kSpan[3] = 0.0f;
        kSpan[4] = 0.0f; kSpan[5] = 1.0f; kSpan[6] = 0.0f; kSpan[7] = 0.0f;

        Span<float> vSpan = v.AsSpan<float>();
        vSpan[0] = 10.0f; vSpan[1] = 20.0f; vSpan[2] = 30.0f; vSpan[3] = 40.0f;
        vSpan[4] = 50.0f; vSpan[5] = 60.0f; vSpan[6] = 70.0f; vSpan[7] = 80.0f;

        Span<float> maskSpan = mask.AsSpan<float>();
        // mask[row0] = [0, -inf] -> token 0 can only attend to token 0
        // mask[row1] = [0, 0] -> token 1 attends to both
        maskSpan[0] = 0.0f;
        maskSpan[1] = float.NegativeInfinity;
        maskSpan[2] = 0.0f;
        maskSpan[3] = 0.0f;

        float scale = 1.0f / MathF.Sqrt(4.0f);

        AttentionKernels.ScaledDotProductAttention(output, q, k, v, mask, scale);

        Span<float> outSpan = output.AsSpan<float>();

        // Token 0: after masking, softmax([score, -inf]) -> [1.0, 0.0]
        // Output[0,:] = 1.0 * V[0,:] = [10, 20, 30, 40]
        Assert.Equal(10.0f, outSpan[0], Tolerance);
        Assert.Equal(20.0f, outSpan[1], Tolerance);
        Assert.Equal(30.0f, outSpan[2], Tolerance);
        Assert.Equal(40.0f, outSpan[3], Tolerance);

        // Token 1: Q[1] dot K[0] = 0, Q[1] dot K[1] = 1
        // Scaled: [0*scale, 1*scale] = [0, 0.5], mask=[0,0], so [0, 0.5]
        // softmax([0, 0.5]) = [exp(0)/(exp(0)+exp(0.5)), exp(0.5)/(exp(0)+exp(0.5))]
        float exp0 = MathF.Exp(0.0f);
        float exp05 = MathF.Exp(0.5f);
        float sumExp = exp0 + exp05;
        float w0 = exp0 / sumExp;
        float w1 = exp05 / sumExp;
        Assert.Equal(w0 * 10.0f + w1 * 50.0f, outSpan[4], Tolerance);
        Assert.Equal(w0 * 20.0f + w1 * 60.0f, outSpan[5], Tolerance);
        Assert.Equal(w0 * 30.0f + w1 * 70.0f, outSpan[6], Tolerance);
        Assert.Equal(w0 * 40.0f + w1 * 80.0f, outSpan[7], Tolerance);
    }
}
