using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using Xunit;

namespace SharpInference.Vision.Tests;

/// <summary>Sanity test for <see cref="IBackend.Conv2dDepthwise"/> — uses hand-computed
/// expected values to catch any indexing or per-channel weight bugs in the default fallback.</summary>
public sealed class Conv2dDepthwiseTest
{
    [Fact]
    public void Conv2dDepthwise_2Channels_Stride1_Pad1_3x3_MatchesHandComputed()
    {
        // Input: [1, 2, 3, 3] — 2 channels, 3×3 spatial.
        // Channel 0 input:
        //   1 2 3
        //   4 5 6
        //   7 8 9
        // Channel 1 input:
        //   9 8 7
        //   6 5 4
        //   3 2 1
        // Depthwise weight [2, 1, 3, 3]:
        // Channel 0 kernel: all ones (identity-sum) → output[c0, y, x] = sum of 3×3 patch
        // Channel 1 kernel: only center=1 (identity) → output[c1, y, x] = input[c1, y, x]
        // Bias: [10, -10]

        using IBackend backend = new CpuBackend();
        using Tensor input = new(new TensorShape(1, 2, 3, 3), DType.F32);
        Span<float> inData = input.AsSpan<float>();
        // C0
        inData[0] = 1; inData[1] = 2; inData[2] = 3;
        inData[3] = 4; inData[4] = 5; inData[5] = 6;
        inData[6] = 7; inData[7] = 8; inData[8] = 9;
        // C1
        inData[9] = 9; inData[10] = 8; inData[11] = 7;
        inData[12] = 6; inData[13] = 5; inData[14] = 4;
        inData[15] = 3; inData[16] = 2; inData[17] = 1;

        using Tensor weight = new(new TensorShape(2, 1, 3, 3), DType.F32);
        Span<float> wData = weight.AsSpan<float>();
        // C0 kernel: all 1s
        for (int i = 0; i < 9; i++) wData[i] = 1f;
        // C1 kernel: only center
        for (int i = 9; i < 18; i++) wData[i] = 0f;
        wData[9 + 4] = 1f; // center of 3×3 = index 4

        using Tensor bias = new(new TensorShape(2), DType.F32);
        bias.AsSpan<float>()[0] = 10f;
        bias.AsSpan<float>()[1] = -10f;

        using Tensor output = new(new TensorShape(1, 2, 3, 3), DType.F32);
        backend.Conv2dDepthwise(output, input, weight, bias, 1, 1, 1, 1);

        ReadOnlySpan<float> outData = output.AsReadOnlySpan<float>();
        // Channel 0 expected (with stride=1, pad=1, kernel=all-ones, then +10 bias):
        // Each output is the sum of the 3×3 patch (with zero-padding outside).
        // (0,0) = 0+0+0+0+1+2+0+4+5 + 10 = 12 + 10 = 22
        // (0,1) = 0+0+0+1+2+3+4+5+6 + 10 = 21 + 10 = 31
        // (0,2) = 0+0+0+2+3+0+5+6+0 + 10 = 16 + 10 = 26
        // (1,0) = 0+1+2+0+4+5+0+7+8 + 10 = 27 + 10 = 37
        // (1,1) = 1+2+3+4+5+6+7+8+9 + 10 = 45 + 10 = 55
        // (1,2) = 2+3+0+5+6+0+8+9+0 + 10 = 33 + 10 = 43
        // (2,0) = 0+4+5+0+7+8+0+0+0 + 10 = 24 + 10 = 34
        // (2,1) = 4+5+6+7+8+9+0+0+0 + 10 = 39 + 10 = 49
        // (2,2) = 5+6+0+8+9+0+0+0+0 + 10 = 28 + 10 = 38
        float[] expectedC0 = [22, 31, 26, 37, 55, 43, 34, 49, 38];
        for (int i = 0; i < 9; i++)
            Assert.Equal(expectedC0[i], outData[i], tolerance: 1e-4f);

        // Channel 1 expected (identity kernel + -10 bias):
        // (y, x) = input[1, y, x] - 10
        float[] inC1 = [9, 8, 7, 6, 5, 4, 3, 2, 1];
        for (int i = 0; i < 9; i++)
            Assert.Equal(inC1[i] - 10, outData[9 + i], tolerance: 1e-4f);
    }
}
