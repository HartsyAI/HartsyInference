using HartsyInference.Cpu.Kernels;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Cpu.Tests;

public sealed unsafe class Conv2DKernelTests
{
    private const float Tolerance = 1e-5f;

    [Fact]
    public void Conv2D_1x1_NopadNostride_EqualsMatMul()
    {
        // input [1,2,3,3] (2 channels, 3x3)
        using Tensor input = new Tensor(new TensorShape(1, 2, 3, 3), DType.F32);
        // weight [4,2,1,1] (4 output channels)
        using Tensor weight = new Tensor(new TensorShape(4, 2, 1, 1), DType.F32);
        // output [1,4,3,3]
        using Tensor output = new Tensor(new TensorShape(1, 4, 3, 3), DType.F32);

        Span<float> inSpan = input.AsSpan<float>();
        // Channel 0: values 1..9, Channel 1: values 10..18
        for (int i = 0; i < 9; i++)
        {
            inSpan[i] = (float)(i + 1);
            inSpan[9 + i] = (float)(i + 10);
        }

        Span<float> wSpan = weight.AsSpan<float>();
        // weight[cout, cin, 0, 0]:
        // cout=0: [1.0, 0.5]
        // cout=1: [0.0, 1.0]
        // cout=2: [-1.0, 2.0]
        // cout=3: [0.5, 0.5]
        wSpan[0] = 1.0f;  wSpan[1] = 0.5f;
        wSpan[2] = 0.0f;  wSpan[3] = 1.0f;
        wSpan[4] = -1.0f; wSpan[5] = 2.0f;
        wSpan[6] = 0.5f;  wSpan[7] = 0.5f;

        Conv2DKernels.Conv2D(output, input, weight, null, 1, 1, 0, 0);

        Span<float> outSpan = output.AsSpan<float>();

        // For 1x1 conv: output[cout, h, w] = sum_cin(weight[cout, cin] * input[cin, h, w])
        // cout=0: 1.0*input[0,:,:] + 0.5*input[1,:,:]
        // Position (0,0): 1.0*1 + 0.5*10 = 6.0
        // Position (0,1): 1.0*2 + 0.5*11 = 7.5
        // Position (0,2): 1.0*3 + 0.5*12 = 9.0
        float[] expectedCout0 = new float[]
        {
            1.0f * 1 + 0.5f * 10, 1.0f * 2 + 0.5f * 11, 1.0f * 3 + 0.5f * 12,
            1.0f * 4 + 0.5f * 13, 1.0f * 5 + 0.5f * 14, 1.0f * 6 + 0.5f * 15,
            1.0f * 7 + 0.5f * 16, 1.0f * 8 + 0.5f * 17, 1.0f * 9 + 0.5f * 18,
        };

        for (int i = 0; i < 9; i++)
        {
            Assert.Equal(expectedCout0[i], outSpan[i], Tolerance);
        }

        // cout=1: 0.0*input[0,:,:] + 1.0*input[1,:,:]
        for (int i = 0; i < 9; i++)
        {
            Assert.Equal((float)(i + 10), outSpan[9 + i], Tolerance);
        }

        // cout=2: -1.0*input[0,:,:] + 2.0*input[1,:,:]
        float[] expectedCout2 = new float[]
        {
            -1.0f * 1 + 2.0f * 10, -1.0f * 2 + 2.0f * 11, -1.0f * 3 + 2.0f * 12,
            -1.0f * 4 + 2.0f * 13, -1.0f * 5 + 2.0f * 14, -1.0f * 6 + 2.0f * 15,
            -1.0f * 7 + 2.0f * 16, -1.0f * 8 + 2.0f * 17, -1.0f * 9 + 2.0f * 18,
        };

        for (int i = 0; i < 9; i++)
        {
            Assert.Equal(expectedCout2[i], outSpan[18 + i], Tolerance);
        }

        // cout=3: 0.5*input[0,:,:] + 0.5*input[1,:,:]
        float[] expectedCout3 = new float[]
        {
            0.5f * 1 + 0.5f * 10, 0.5f * 2 + 0.5f * 11, 0.5f * 3 + 0.5f * 12,
            0.5f * 4 + 0.5f * 13, 0.5f * 5 + 0.5f * 14, 0.5f * 6 + 0.5f * 15,
            0.5f * 7 + 0.5f * 16, 0.5f * 8 + 0.5f * 17, 0.5f * 9 + 0.5f * 18,
        };

        for (int i = 0; i < 9; i++)
        {
            Assert.Equal(expectedCout3[i], outSpan[27 + i], Tolerance);
        }
    }

    [Fact]
    public void Conv2D_3x3_WithPadding()
    {
        // input [1,1,4,4] all ones
        using Tensor input = new Tensor(new TensorShape(1, 1, 4, 4), DType.F32);
        // weight [1,1,3,3] all ones
        using Tensor weight = new Tensor(new TensorShape(1, 1, 3, 3), DType.F32);
        // output [1,1,4,4] (stride=1, pad=1 preserves spatial)
        using Tensor output = new Tensor(new TensorShape(1, 1, 4, 4), DType.F32);

        Span<float> inSpan = input.AsSpan<float>();
        for (int i = 0; i < 16; i++)
        {
            inSpan[i] = 1.0f;
        }

        Span<float> wSpan = weight.AsSpan<float>();
        for (int i = 0; i < 9; i++)
        {
            wSpan[i] = 1.0f;
        }

        Conv2DKernels.Conv2D(output, input, weight, null, 1, 1, 1, 1);

        Span<float> outSpan = output.AsSpan<float>();

        // Expected output for 3x3 all-ones kernel with pad=1 on 4x4 all-ones input:
        // corners: 4, edges: 6, interior: 9
        float[] expected = new float[]
        {
            4.0f, 6.0f, 6.0f, 4.0f,
            6.0f, 9.0f, 9.0f, 6.0f,
            6.0f, 9.0f, 9.0f, 6.0f,
            4.0f, 6.0f, 6.0f, 4.0f,
        };

        for (int i = 0; i < 16; i++)
        {
            Assert.Equal(expected[i], outSpan[i], Tolerance);
        }
    }

    [Fact]
    public void Conv2D_WithBias()
    {
        // Same as 3x3 with padding test, but add bias of 1.0
        using Tensor input = new Tensor(new TensorShape(1, 1, 4, 4), DType.F32);
        using Tensor weight = new Tensor(new TensorShape(1, 1, 3, 3), DType.F32);
        using Tensor bias = new Tensor(new TensorShape(1), DType.F32);
        using Tensor output = new Tensor(new TensorShape(1, 1, 4, 4), DType.F32);

        Span<float> inSpan = input.AsSpan<float>();
        for (int i = 0; i < 16; i++)
        {
            inSpan[i] = 1.0f;
        }

        Span<float> wSpan = weight.AsSpan<float>();
        for (int i = 0; i < 9; i++)
        {
            wSpan[i] = 1.0f;
        }

        bias.AsSpan<float>()[0] = 1.0f;

        Conv2DKernels.Conv2D(output, input, weight, bias, 1, 1, 1, 1);

        Span<float> outSpan = output.AsSpan<float>();

        // Same as before plus bias=1.0
        float[] expected = new float[]
        {
            5.0f, 7.0f, 7.0f, 5.0f,
            7.0f, 10.0f, 10.0f, 7.0f,
            7.0f, 10.0f, 10.0f, 7.0f,
            5.0f, 7.0f, 7.0f, 5.0f,
        };

        for (int i = 0; i < 16; i++)
        {
            Assert.Equal(expected[i], outSpan[i], Tolerance);
        }
    }

    [Fact]
    public void Conv2D_Stride2_ReducesSpatial()
    {
        // input [1,1,4,4]
        using Tensor input = new Tensor(new TensorShape(1, 1, 4, 4), DType.F32);
        // weight [1,1,1,1] = 1.0 (identity kernel)
        using Tensor weight = new Tensor(new TensorShape(1, 1, 1, 1), DType.F32);
        // output [1,1,2,2] (stride=2, pad=0, 1x1 kernel: outH=(4-1)/2+1=2)
        using Tensor output = new Tensor(new TensorShape(1, 1, 2, 2), DType.F32);

        Span<float> inSpan = input.AsSpan<float>();
        // Fill with row-major indices
        for (int i = 0; i < 16; i++)
        {
            inSpan[i] = (float)i;
        }

        weight.AsSpan<float>()[0] = 1.0f;

        Conv2DKernels.Conv2D(output, input, weight, null, 2, 2, 0, 0);

        Span<float> outSpan = output.AsSpan<float>();

        // With stride=2 and 1x1 kernel, picks pixels at (0,0),(0,2),(2,0),(2,2)
        // input[0,0]=0, input[0,2]=2, input[2,0]=8, input[2,2]=10
        Assert.Equal(0.0f, outSpan[0], Tolerance);
        Assert.Equal(2.0f, outSpan[1], Tolerance);
        Assert.Equal(8.0f, outSpan[2], Tolerance);
        Assert.Equal(10.0f, outSpan[3], Tolerance);
    }
}
