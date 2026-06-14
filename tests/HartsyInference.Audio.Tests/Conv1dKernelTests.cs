using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Numerical correctness tests for <c>IBackend.Conv1d</c> and
/// <c>IBackend.ConvTranspose1d</c>. These are the workhorses of every audio codec we
/// shipped (EnCodec / DAC / SNAC / Mimi / WavTokenizer / BiCodec / XCodec) so a
/// regression here cascades into wrong-sounding audio across the board.
///
/// <para>Strategy: hand-computed expected outputs for small fixed-weight examples.
/// Easy to reason about by eye; PyTorch-equivalent results verified separately
/// (the math is standard).</para></summary>
public sealed unsafe class Conv1dKernelTests
{
    [Fact]
    public void Conv1d_IdentityKernel_PreservesInput()
    {
        // Kernel size 1, weight = 1.0, no bias → output = input.
        using CpuBackend backend = new();
        int batch = 1, c = 2, t = 4;
        Tensor input = MakeInput(batch, c, t, valueAt: (b, ch, tt) => ch * 10 + tt);
        Tensor weight = new(new TensorShape(c, c, 1), DType.F32);
        Tensor output = new(new TensorShape(batch, c, t), DType.F32);
        try
        {
            float* wp = (float*)weight.DataPointer;
            // Identity per-channel: out_c = in_c.
            for (int oc = 0; oc < c; oc++)
                for (int ic = 0; ic < c; ic++)
                    wp[(oc * c + ic) * 1] = (oc == ic) ? 1f : 0f;

            backend.Conv1d(output, input, weight, bias: null,
                stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);

            float* ip = (float*)input.DataPointer;
            float* op = (float*)output.DataPointer;
            for (long i = 0; i < input.ElementCount; i++)
                Assert.Equal(ip[i], op[i], precision: 5);
        }
        finally
        {
            input.Dispose();
            weight.Dispose();
            output.Dispose();
        }
    }

    [Fact]
    public void Conv1d_SumKernel_AddsAcrossChannels()
    {
        // Kernel size 1, all weights = 1.0 → output[oc, t] = sum of inputs across all channels.
        using CpuBackend backend = new();
        int batch = 1, cIn = 3, cOut = 1, t = 3;
        Tensor input = MakeInput(batch, cIn, t, valueAt: (b, ch, tt) => 1f);     // all ones
        Tensor weight = new(new TensorShape(cOut, cIn, 1), DType.F32);
        Tensor output = new(new TensorShape(batch, cOut, t), DType.F32);
        try
        {
            float* wp = (float*)weight.DataPointer;
            for (int i = 0; i < cOut * cIn; i++) wp[i] = 1f;

            backend.Conv1d(output, input, weight, bias: null,
                stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);

            float* op = (float*)output.DataPointer;
            for (int j = 0; j < t; j++)
                Assert.Equal(cIn, op[j], precision: 5);     // sum of 3 ones = 3
        }
        finally
        {
            input.Dispose();
            weight.Dispose();
            output.Dispose();
        }
    }

    [Fact]
    public void Conv1d_DepthwiseGrouped_PreservesPerChannel()
    {
        // Depthwise mode: groups == in_channels, kernel applies per-channel independently.
        using CpuBackend backend = new();
        int batch = 1, c = 3, t = 4;
        Tensor input = MakeInput(batch, c, t, valueAt: (b, ch, tt) => ch + 1);     // channel 0 = 1, channel 1 = 2, channel 2 = 3
        Tensor weight = new(new TensorShape(c, 1, 3), DType.F32);                  // [c_out, c_in/groups=1, k=3]
        Tensor output = new(new TensorShape(batch, c, t), DType.F32);
        try
        {
            float* wp = (float*)weight.DataPointer;
            // Each channel's kernel = [1, 0, 0] (left-only). With pad=2 left, output at j=0 reads in[-2..0].
            // With pad=2 on each side and stride=1, output T preserved.
            for (int c0 = 0; c0 < c; c0++)
            {
                wp[c0 * 3 + 0] = 1f;
                wp[c0 * 3 + 1] = 0f;
                wp[c0 * 3 + 2] = 0f;
            }
            backend.Conv1d(output, input, weight, bias: null,
                stride: 1, padLeft: 1, padRight: 1, dilation: 1, groups: c);

            // With kernel [1, 0, 0] and pad=1 on each side, output[j] reads in[j-1].
            // out[c, 0] = in[c, -1] = 0 (pad). out[c, 1] = in[c, 0] = ch+1. out[c, 2] = in[c, 1] = ch+1.
            float* op = (float*)output.DataPointer;
            for (int c0 = 0; c0 < c; c0++)
            {
                Assert.Equal(0f, op[c0 * t + 0], precision: 5);
                for (int j = 1; j < t; j++)
                    Assert.Equal(c0 + 1, op[c0 * t + j], precision: 5);
            }
        }
        finally
        {
            input.Dispose();
            weight.Dispose();
            output.Dispose();
        }
    }

    [Fact]
    public void Conv1d_BiasIsAddedOncePerOutputChannel()
    {
        using CpuBackend backend = new();
        int batch = 1, c = 2, t = 3;
        Tensor input = MakeInput(batch, c, t, valueAt: (b, ch, tt) => 0f);     // zero input
        Tensor weight = new(new TensorShape(c, c, 1), DType.F32);
        Tensor bias = new(new TensorShape(c), DType.F32);
        Tensor output = new(new TensorShape(batch, c, t), DType.F32);
        try
        {
            float* wp = (float*)weight.DataPointer;
            for (int i = 0; i < c * c; i++) wp[i] = 1f;
            float* bp = (float*)bias.DataPointer;
            bp[0] = 5f; bp[1] = -3f;

            backend.Conv1d(output, input, weight, bias,
                stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);

            float* op = (float*)output.DataPointer;
            for (int j = 0; j < t; j++)
            {
                Assert.Equal(5f, op[j], precision: 5);                     // channel 0
                Assert.Equal(-3f, op[t + j], precision: 5);                // channel 1
            }
        }
        finally
        {
            input.Dispose();
            weight.Dispose();
            bias.Dispose();
            output.Dispose();
        }
    }

    [Fact]
    public void Conv1d_StridedDownsampling_HalvesTimeAtStride2()
    {
        using CpuBackend backend = new();
        int batch = 1, c = 1, tIn = 8;
        Tensor input = MakeInput(batch, c, tIn, valueAt: (b, ch, tt) => tt + 1);     // 1, 2, 3, ..., 8
        Tensor weight = new(new TensorShape(c, c, 2), DType.F32);                     // k=2
        int tOut = (tIn + 0 - (2 - 1) - 1) / 2 + 1;                                  // = 4
        Tensor output = new(new TensorShape(batch, c, tOut), DType.F32);
        try
        {
            float* wp = (float*)weight.DataPointer;
            wp[0] = 1f; wp[1] = 1f;     // sum-of-pair kernel

            backend.Conv1d(output, input, weight, bias: null,
                stride: 2, padLeft: 0, padRight: 0, dilation: 1, groups: 1);

            // With kernel [1, 1], stride 2: out[0] = in[0]+in[1] = 3, out[1] = in[2]+in[3] = 7,
            // out[2] = 11, out[3] = 15.
            float* op = (float*)output.DataPointer;
            Assert.Equal(3f, op[0], precision: 5);
            Assert.Equal(7f, op[1], precision: 5);
            Assert.Equal(11f, op[2], precision: 5);
            Assert.Equal(15f, op[3], precision: 5);
        }
        finally
        {
            input.Dispose();
            weight.Dispose();
            output.Dispose();
        }
    }

    [Fact]
    public void ConvTranspose1d_Stride2DoublesTimeWithSumKernel()
    {
        // Input [1, 2] @ stride=2 kernel=2 weight=[1,1] should produce [1, 1, 2, 2] (each
        // input value contributes to two consecutive output positions).
        using CpuBackend backend = new();
        int batch = 1, cIn = 1, cOut = 1, tIn = 2;
        Tensor input = MakeInput(batch, cIn, tIn, valueAt: (b, ch, tt) => tt + 1f);     // [1, 2]
        Tensor weight = new(new TensorShape(cIn, cOut, 2), DType.F32);                   // [c_in, c_out, k]
        try
        {
            float* wp = (float*)weight.DataPointer;
            wp[0] = 1f; wp[1] = 1f;

            // Output length = (T_in - 1) * stride + K - padLeft - padRight = 1*2 + 2 - 0 - 0 = 4.
            int tOut = (tIn - 1) * 2 + 2 - 0 - 0;
            Tensor output = new(new TensorShape(batch, cOut, tOut), DType.F32);
            try
            {
                backend.ConvTranspose1d(output, input, weight, bias: null,
                    stride: 2, padLeft: 0, padRight: 0, dilation: 1);

                float* op = (float*)output.DataPointer;
                // i=0 (value=1) contributes to j=0, j=1. i=1 (value=2) contributes to j=2, j=3.
                Assert.Equal(1f, op[0], precision: 5);
                Assert.Equal(1f, op[1], precision: 5);
                Assert.Equal(2f, op[2], precision: 5);
                Assert.Equal(2f, op[3], precision: 5);
            }
            finally
            {
                output.Dispose();
            }
        }
        finally
        {
            input.Dispose();
            weight.Dispose();
        }
    }

    [Fact]
    public void ConvTranspose1d_BiasAppliedBeforeAccumulate()
    {
        // With zero input, bias broadcast across the output time axis.
        using CpuBackend backend = new();
        int batch = 1, cIn = 1, cOut = 2, tIn = 3;
        Tensor input = MakeInput(batch, cIn, tIn, valueAt: (b, ch, tt) => 0f);
        Tensor weight = new(new TensorShape(cIn, cOut, 2), DType.F32);
        Tensor bias = new(new TensorShape(cOut), DType.F32);
        try
        {
            float* bp = (float*)bias.DataPointer;
            bp[0] = 2f; bp[1] = -1f;

            int tOut = (tIn - 1) * 2 + 2;     // = 6
            Tensor output = new(new TensorShape(batch, cOut, tOut), DType.F32);
            try
            {
                backend.ConvTranspose1d(output, input, weight, bias,
                    stride: 2, padLeft: 0, padRight: 0, dilation: 1);

                float* op = (float*)output.DataPointer;
                for (int j = 0; j < tOut; j++) Assert.Equal(2f, op[j], precision: 5);
                for (int j = 0; j < tOut; j++) Assert.Equal(-1f, op[tOut + j], precision: 5);
            }
            finally
            {
                output.Dispose();
            }
        }
        finally
        {
            input.Dispose();
            weight.Dispose();
            bias.Dispose();
        }
    }

    private static Tensor MakeInput(int batch, int channels, int t, Func<int, int, int, float> valueAt)
    {
        Tensor tensor = new(new TensorShape(batch, channels, t), DType.F32);
        float* p = (float*)tensor.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
                for (int j = 0; j < t; j++)
                    p[(b * channels + c) * t + j] = valueAt(b, c, j);
        return tensor;
    }
}
