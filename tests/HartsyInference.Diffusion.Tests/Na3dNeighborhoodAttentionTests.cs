using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins NATTEN's window rule for the LTX-2.5 diffusion video decoder: every query attends exactly
/// <c>kernel</c> keys per axis, and near a border the window slides inward rather than being truncated or
/// zero-padded. Getting this wrong still produces plausible output, so it would only ever surface as a decoder
/// parity miss.</summary>
public sealed unsafe class Na3dNeighborhoodAttentionTests
{
    private static Tensor Make(int b, int t, int h, int w, int heads, int headDim, Func<long, float> fill)
    {
        Tensor x = new Tensor(new TensorShape([(long)b, t, h, w, heads, headDim]), DType.F32);
        float* p = (float*)x.DataPointer;
        for (long i = 0; i < x.ElementCount; i++) p[i] = fill(i);
        return x;
    }

    /// <summary>The rule the reference implements: clamp the kernel to the axis, then slide the window inward.</summary>
    private static (int Start, int Kernel) ExpectedWindow(int index, int length, int kernel)
    {
        int k = Math.Min(kernel, length);
        int start = Math.Clamp(index - k / 2, 0, length - k);
        return (start, k);
    }

    [Theory]
    [InlineData(8, 3)]
    [InlineData(8, 5)]
    [InlineData(5, 5)]
    [InlineData(3, 7)]
    [InlineData(1, 11)]
    [InlineData(11, 11)]
    public void WindowSlidesInwardAndKeepsFullWidth(int length, int kernel)
    {
        int k = Math.Min(kernel, length);
        for (int i = 0; i < length; i++)
        {
            int start = IBackend.Na3dWindowStart(i, length, k);
            (int expectedStart, int expectedKernel) = ExpectedWindow(i, length, kernel);

            Assert.Equal(expectedStart, start);
            Assert.Equal(k, expectedKernel);
            Assert.True(start >= 0, $"window start {start} underflows at index {i}");
            Assert.True(start + k <= length, $"window [{start},{start + k}) overruns length {length} at index {i}");
        }
    }

    [Fact]
    public void EveryQueryAttendsTheSameKeyCount()
    {
        // The defining property of NATTEN vs a truncated window: border queries are not starved.
        const int length = 9, kernel = 5;
        for (int i = 0; i < length; i++)
        {
            int start = IBackend.Na3dWindowStart(i, length, kernel);
            Assert.Equal(kernel, Math.Min(start + kernel, length) - start);
        }
    }

    [Fact]
    public void KernelSpanningTheGridEqualsFullAttention()
    {
        // With the kernel covering every axis, neighborhood attention degenerates to dense attention, which is a
        // reference the test can compute independently.
        IBackend backend = new CpuBackend();
        const int t = 2, h = 2, w = 2, heads = 1, hd = 4;
        using Tensor q = Make(1, t, h, w, heads, hd, i => MathF.Sin(i * 0.7f));
        using Tensor k = Make(1, t, h, w, heads, hd, i => MathF.Cos(i * 0.4f));
        using Tensor v = Make(1, t, h, w, heads, hd, i => i * 0.03f);
        using Tensor outp = Make(1, t, h, w, heads, hd, _ => 0f);

        backend.Na3d(outp, q, k, v, t, h, w, scale: 1.0f);

        int tokens = t * h * w;
        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer;
        float* vp = (float*)v.DataPointer, op = (float*)outp.DataPointer;
        for (int i = 0; i < tokens; i++)
        {
            float[] scores = new float[tokens];
            float max = float.NegativeInfinity;
            for (int j = 0; j < tokens; j++)
            {
                float dot = 0f;
                for (int d = 0; d < hd; d++) dot += qp[i * hd + d] * kp[j * hd + d];
                scores[j] = dot;
                max = MathF.Max(max, dot);
            }
            float sum = 0f;
            for (int j = 0; j < tokens; j++) { scores[j] = MathF.Exp(scores[j] - max); sum += scores[j]; }
            for (int d = 0; d < hd; d++)
            {
                float acc = 0f;
                for (int j = 0; j < tokens; j++) acc += scores[j] / sum * vp[j * hd + d];
                Assert.Equal(acc, op[i * hd + d], 5);
            }
        }
    }

    [Fact]
    public void OutputIsAConvexCombinationOfTheWindowsValues()
    {
        // Softmax weights sum to 1, so with a constant V every query must reproduce that constant exactly —
        // this catches a mis-sized window (which would still normalize) only in combination with the locality
        // test below, but it does catch unnormalized or double-counted weights.
        IBackend backend = new CpuBackend();
        const int t = 4, h = 4, w = 4, heads = 2, hd = 3;
        using Tensor q = Make(1, t, h, w, heads, hd, i => MathF.Sin(i * 0.31f));
        using Tensor k = Make(1, t, h, w, heads, hd, i => MathF.Cos(i * 0.17f));
        using Tensor v = Make(1, t, h, w, heads, hd, _ => 2.5f);
        using Tensor outp = Make(1, t, h, w, heads, hd, _ => 0f);

        backend.Na3d(outp, q, k, v, 3, 3, 3, scale: 1.0f);

        float* op = (float*)outp.DataPointer;
        for (long i = 0; i < outp.ElementCount; i++) Assert.Equal(2.5f, op[i], 5);
    }

    [Fact]
    public void AttentionIsLocalToTheWindow()
    {
        // Perturbing a value far outside a query's window must not move that query's output; perturbing one
        // inside must. This is what distinguishes neighborhood attention from dense attention.
        IBackend backend = new CpuBackend();
        const int t = 1, h = 1, w = 9, heads = 1, hd = 2;

        float[] Run(int spikeAt)
        {
            using Tensor q = Make(1, t, h, w, heads, hd, i => MathF.Sin(i * 0.5f));
            using Tensor k = Make(1, t, h, w, heads, hd, i => MathF.Cos(i * 0.3f));
            using Tensor v = Make(1, t, h, w, heads, hd, i => i == spikeAt * hd ? 99f : 1f);
            using Tensor outp = Make(1, t, h, w, heads, hd, _ => 0f);
            backend.Na3d(outp, q, k, v, 1, 1, 3, scale: 1.0f);
            float[] copy = new float[outp.ElementCount];
            new Span<float>((void*)outp.DataPointer, copy.Length).CopyTo(copy);
            return copy;
        }

        float[] baseline = Run(-1);
        float[] nearSpike = Run(1);   // inside query 0's window [0,3)
        float[] farSpike = Run(8);    // outside it

        Assert.NotEqual(baseline[0], nearSpike[0], 4);
        Assert.Equal(baseline[0], farSpike[0], 5);
    }

    [Fact]
    public void ScaleMultipliesTheScores()
    {
        // The decoder passes scale=1.0 because it folds the 1/sqrt(headDim) into the query norm weight, so the
        // parameter has to be a true no-op at 1.0 and actually bite otherwise.
        IBackend backend = new CpuBackend();
        const int t = 2, h = 2, w = 2, heads = 1, hd = 4;
        using Tensor q = Make(1, t, h, w, heads, hd, i => MathF.Sin(i * 0.9f));
        using Tensor k = Make(1, t, h, w, heads, hd, i => MathF.Cos(i * 0.6f));
        using Tensor v = Make(1, t, h, w, heads, hd, i => i * 0.11f);
        using Tensor unscaled = Make(1, t, h, w, heads, hd, _ => 0f);
        using Tensor scaled = Make(1, t, h, w, heads, hd, _ => 0f);
        using Tensor preScaledQ = Make(1, t, h, w, heads, hd, i => MathF.Sin(i * 0.9f) * 0.5f);

        backend.Na3d(scaled, q, k, v, 2, 2, 2, scale: 0.5f);
        backend.Na3d(unscaled, preScaledQ, k, v, 2, 2, 2, scale: 1.0f);

        float* a = (float*)scaled.DataPointer;
        float* b = (float*)unscaled.DataPointer;
        for (long i = 0; i < scaled.ElementCount; i++) Assert.Equal(b[i], a[i], 5);
    }

    [Fact]
    public void MismatchedShapesAreRejected()
    {
        IBackend backend = new CpuBackend();
        using Tensor q = Make(1, 2, 2, 2, 1, 4, _ => 0f);
        using Tensor k = Make(1, 2, 2, 2, 1, 4, _ => 0f);
        using Tensor v = Make(1, 2, 2, 2, 1, 4, _ => 0f);
        using Tensor wrong = Make(1, 2, 2, 2, 1, 8, _ => 0f);

        Assert.Throws<ArgumentException>(() => backend.Na3d(wrong, q, k, v, 3, 3, 3, 1.0f));
        Assert.Throws<ArgumentException>(() => backend.Na3d(q, q, k, v, 0, 3, 3, 1.0f));
    }
}
