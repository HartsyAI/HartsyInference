using HartsyInference.Core.Configuration;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Correctness of the 1D convolution kernels at sizes large enough to actually fan out across cores.
///
/// <para>Every other Conv1d test in this project runs at 32 channels or fewer, which is below
/// <see cref="CpuParallel.MinWorkForParallel"/> — so the whole existing suite exercises only the serial branch
/// and would keep passing if the parallel one were wrong. These shapes cross the threshold deliberately.</para>
///
/// <para>The reference is a plain nested-loop implementation written out in this file rather than the kernel run
/// with one worker: a self-comparison would pass if both paths shared the same mistake, and the point here is
/// the arithmetic, not just that the two branches agree. Comparisons are exact, because each output element is
/// summed by exactly one worker in the same order the reference uses — a mismatch means the work was divided
/// wrongly, not that floating point drifted.</para></summary>
public sealed unsafe class Conv1dParallelKernelTests
{
    /// <summary>Shapes taken from the layers that actually dominate a Piper synthesis: wide-channel 1x1
    /// projections, dilated residual stacks, and the vocoder's near-final layers where the channel count has
    /// collapsed but the time axis is enormous.</summary>
    public static TheoryData<string, int, int, int, int, int, int, int, int> Conv1dShapes => new()
    {
        // name,                        cIn, cOut,   tIn, kernel, stride, pad, dilation, groups
        { "wide 1x1 projection",        192,  768,   400,      1,      1,   0,        1,      1 },
        { "residual k3 dilated",        192,  192,   400,      3,      1,   2,        2,      1 },
        { "vocoder tail, 1 out channel",  32,    1, 20000,      7,      1,   3,        1,      1 },
        { "depthwise grouped",          128,  128,  2000,      5,      1,   2,        1,    128 },
        { "strided downsample",          64,  128,  4000,      4,      2,   1,        1,      1 },
    };

    [Theory]
    [MemberData(nameof(Conv1dShapes))]
    public void Conv1d_MatchesSerialReference_AboveParallelThreshold(
        string name, int cIn, int cOut, int tIn, int kernel, int stride, int pad, int dilation, int groups)
    {
        Assert.NotNull(name);
        using CpuBackend backend = new();
        int inPerGroup = cIn / groups;
        int tOut = (tIn + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;

        Tensor input = Filled(1, cIn, tIn, seed: 11);
        Tensor weight = Filled(cOut, inPerGroup, kernel, seed: 23);
        Tensor bias = Filled(1, 1, cOut, seed: 37);
        Tensor output = new(new TensorShape(1, cOut, tOut), DType.F32);
        try
        {
            backend.Conv1d(output, input, weight, bias, stride, pad, pad, dilation, groups);
            float[] expected = Conv1dReference(input, weight, bias, cIn, cOut, tIn, tOut, kernel, stride, pad, dilation, groups);
            AssertExact(expected, output);
        }
        finally
        {
            input.Dispose();
            weight.Dispose();
            bias.Dispose();
            output.Dispose();
        }
    }

    /// <summary>The transposed convolution's split is the risky one: the kernel accumulates into its output, so
    /// dividing the work along the wrong axis lets two workers read-modify-write the same element. Multiple
    /// input channels per group is exactly the case that would expose it.</summary>
    public static TheoryData<string, int, int, int, int, int, int, int> ConvTransposeShapes => new()
    {
        // name,                     cIn, ocPerG,  tIn, kernel, stride, pad, groups
        { "vocoder upsample x8",     256,    128,  500,     16,      8,   4,      1 },
        { "vocoder upsample x2",     128,     64, 2000,      4,      2,   1,      1 },
        { "grouped, many ic/group",   64,     32,  800,      6,      2,   2,      2 },
        { "dilated-free k3 stride1",  96,     96, 1500,      3,      1,   1,      1 },
    };

    [Theory]
    [MemberData(nameof(ConvTransposeShapes))]
    public void ConvTranspose1d_MatchesSerialReference_AboveParallelThreshold(
        string name, int cIn, int ocPerG, int tIn, int kernel, int stride, int pad, int groups)
    {
        Assert.NotNull(name);
        using CpuBackend backend = new();
        int cOut = ocPerG * groups;
        int tOut = (tIn - 1) * stride + (kernel - 1) + 1 - 2 * pad;

        Tensor input = Filled(1, cIn, tIn, seed: 5);
        Tensor weight = Filled(cIn, ocPerG, kernel, seed: 17);
        Tensor bias = Filled(1, 1, cOut, seed: 29);
        Tensor output = new(new TensorShape(1, cOut, tOut), DType.F32);
        try
        {
            backend.ConvTranspose1d(output, input, weight, bias, stride, pad, pad, dilation: 1, groups: groups);
            float[] expected = ConvTranspose1dReference(input, weight, bias, cIn, cOut, ocPerG, tIn, tOut, kernel, stride, pad, groups);
            AssertExact(expected, output);
        }
        finally
        {
            input.Dispose();
            weight.Dispose();
            bias.Dispose();
            output.Dispose();
        }
    }

    /// <summary>Capping the worker count must change only the schedule. A host that turns the knob down to leave
    /// room for another model has to get the same audio out, or the knob is a correctness switch.</summary>
    [Fact]
    public void Conv1d_SingleThreadedKnob_ProducesIdenticalOutput()
    {
        using CpuBackend backend = new();
        int cIn = 192, cOut = 256, tIn = 800, kernel = 3;
        int tOut = tIn + 2 - (kernel - 1) - 1 + 1;

        Tensor input = Filled(1, cIn, tIn, seed: 71);
        Tensor weight = Filled(cOut, cIn, kernel, seed: 83);
        Tensor parallel = new(new TensorShape(1, cOut, tOut), DType.F32);
        Tensor serial = new(new TensorShape(1, cOut, tOut), DType.F32);
        try
        {
            KnobStore.Clear(EngineKnobs.CpuThreads);
            backend.Conv1d(parallel, input, weight, null, 1, 1, 1, 1, 1);
            KnobStore.Set(EngineKnobs.CpuThreads, 1);
            try
            {
                backend.Conv1d(serial, input, weight, null, 1, 1, 1, 1, 1);
            }
            finally
            {
                KnobStore.Clear(EngineKnobs.CpuThreads);
            }

            float* a = (float*)parallel.DataPointer;
            float* b = (float*)serial.DataPointer;
            for (long i = 0; i < parallel.ElementCount; i++)
            {
                Assert.Equal(b[i], a[i]);
            }
        }
        finally
        {
            input.Dispose();
            weight.Dispose();
            parallel.Dispose();
            serial.Dispose();
        }
    }

    /// <summary>An argument the kernel rejects must still surface as itself. Work dispatched through
    /// <see cref="Parallel.For"/> arrives wrapped in an <see cref="AggregateException"/> unless it is unwrapped
    /// deliberately, and a caller catching <see cref="ArgumentException"/> would stop seeing it.</summary>
    [Fact]
    public void Conv1d_BadShape_ThrowsArgumentExceptionNotAggregate()
    {
        using CpuBackend backend = new();
        Tensor input = Filled(1, 8, 16, seed: 3);
        Tensor weight = Filled(4, 8, 3, seed: 4);
        Tensor wrong = new(new TensorShape(1, 4, 99), DType.F32);
        try
        {
            Assert.Throws<ArgumentException>(() =>
                backend.Conv1d(wrong, input, weight, null, 1, 0, 0, 1, 1));
        }
        finally
        {
            input.Dispose();
            weight.Dispose();
            wrong.Dispose();
        }
    }

    private static float[] Conv1dReference(Tensor input, Tensor weight, Tensor bias,
        int cIn, int cOut, int tIn, int tOut, int kernel, int stride, int pad, int dilation, int groups)
    {
        float* ip = (float*)input.DataPointer;
        float* wp = (float*)weight.DataPointer;
        float* bp = (float*)bias.DataPointer;
        int inPerGroup = cIn / groups;
        int outPerGroup = cOut / groups;
        float[] result = new float[cOut * tOut];
        for (int oc = 0; oc < cOut; oc++)
        {
            int icStart = oc / outPerGroup * inPerGroup;
            for (int j = 0; j < tOut; j++)
            {
                float acc = bp[oc];
                for (int ic = 0; ic < inPerGroup; ic++)
                {
                    for (int k = 0; k < kernel; k++)
                    {
                        int src = j * stride - pad + k * dilation;
                        if (src >= 0 && src < tIn)
                        {
                            acc += ip[(icStart + ic) * tIn + src] * wp[(oc * inPerGroup + ic) * kernel + k];
                        }
                    }
                }
                result[oc * tOut + j] = acc;
            }
        }
        return result;
    }

    private static float[] ConvTranspose1dReference(Tensor input, Tensor weight, Tensor bias,
        int cIn, int cOut, int ocPerG, int tIn, int tOut, int kernel, int stride, int pad, int groups)
    {
        float* ip = (float*)input.DataPointer;
        float* wp = (float*)weight.DataPointer;
        float* bp = (float*)bias.DataPointer;
        int icPerG = cIn / groups;
        float[] result = new float[cOut * tOut];
        for (int oc = 0; oc < cOut; oc++)
        {
            for (int j = 0; j < tOut; j++)
            {
                result[oc * tOut + j] = bp[oc];
            }
            int g = oc / ocPerG;
            int ocLocal = oc - g * ocPerG;
            for (int ic = g * icPerG; ic < (g + 1) * icPerG; ic++)
            {
                for (int i = 0; i < tIn; i++)
                {
                    float xv = ip[ic * tIn + i];
                    if (xv == 0f)
                    {
                        continue;
                    }
                    for (int k = 0; k < kernel; k++)
                    {
                        int j = i * stride - pad + k;
                        if (j >= 0 && j < tOut)
                        {
                            result[oc * tOut + j] += xv * wp[(ic * ocPerG + ocLocal) * kernel + k];
                        }
                    }
                }
            }
        }
        return result;
    }

    /// <summary>Deterministic pseudo-random fill. A constant or a ramp would let an index error in the split go
    /// unnoticed, because neighbouring elements would hold the same value.</summary>
    private static Tensor Filled(int d0, int d1, int d2, int seed)
    {
        Tensor t = new(new TensorShape(d0, d1, d2), DType.F32);
        float* p = (float*)t.DataPointer;
        uint state = (uint)seed * 2654435761u + 1u;
        for (long i = 0; i < t.ElementCount; i++)
        {
            state = state * 1664525u + 1013904223u;
            p[i] = ((state >> 8) & 0xFFFF) / 32768f - 1f;
        }
        return t;
    }

    private static void AssertExact(float[] expected, Tensor actual)
    {
        float* p = (float*)actual.DataPointer;
        Assert.Equal(expected.Length, (int)actual.ElementCount);
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] != p[i])
            {
                Assert.Fail($"element {i}: expected {expected[i]:R}, got {p[i]:R}");
            }
        }
    }
}
