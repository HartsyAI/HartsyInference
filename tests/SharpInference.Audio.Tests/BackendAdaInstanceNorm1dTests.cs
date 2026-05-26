using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using Xunit;

namespace SharpInference.Audio.Tests;

/// <summary>Numerical correctness tests for <c>IBackend.AdaInstanceNorm1d</c> — the
/// AdaIN1d style-conditioning primitive used throughout Kokoro and StyleTTS 2. For each
/// (batch, channel) slice of a <c>[B, C, T]</c> input it normalizes to zero-mean / unit-
/// variance across the time axis, then applies a per-channel affine
/// <c>(1 + gamma[c]) * x_hat + beta[c]</c>.
///
/// <para>The "1 +" in the gamma term matters — it makes the AdaIN block an identity at
/// gamma=beta=0, which is the official AdaIN1d initialization. Tests confirm that
/// (a) the post-normalization mean is ~0, variance is ~1 at gamma=beta=0
/// (b) the affine matches the closed form
/// (c) the per-batch (rank-2 gamma/beta) and broadcast (rank-1) paths agree.</para></summary>
public sealed unsafe class BackendAdaInstanceNorm1dTests
{
    [Fact]
    public void IdentityWhenGammaBetaZero()
    {
        using CpuBackend backend = new();
        int batch = 2, channels = 3, t = 16;
        Tensor input = MakeRandom(batch, channels, t, seed: 1);
        Tensor gamma = ZeroTensor(batch, channels);
        Tensor beta = ZeroTensor(batch, channels);
        Tensor output = new(input.Shape, DType.F32);
        try
        {
            backend.AdaInstanceNorm1d(output, input, gamma, beta, eps: 1e-5f);
            float* op = (float*)output.DataPointer;

            // Each (b, c) slice should now have mean ~0, variance ~1.
            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    double sum = 0d, sumSq = 0d;
                    int row = (b * channels + c) * t;
                    for (int j = 0; j < t; j++)
                    {
                        sum += op[row + j];
                        sumSq += (double)op[row + j] * op[row + j];
                    }
                    double mean = sum / t;
                    double var = sumSq / t - mean * mean;
                    Assert.True(Math.Abs(mean) < 1e-4, $"slice ({b},{c}) mean {mean} not ~0");
                    Assert.InRange(var, 0.99, 1.01);
                }
            }
        }
        finally
        {
            input.Dispose(); gamma.Dispose(); beta.Dispose(); output.Dispose();
        }
    }

    [Fact]
    public void AffineMatchesClosedForm()
    {
        // For a known input with constant mean/var, the AdaIN output equals
        // (1 + gamma) * (x - mean) / std + beta.
        using CpuBackend backend = new();
        int batch = 1, channels = 2, t = 4;
        Tensor input = new(new TensorShape(batch, channels, t), DType.F32);
        Tensor gamma = new(new TensorShape(batch, channels), DType.F32);
        Tensor beta = new(new TensorShape(batch, channels), DType.F32);
        Tensor output = new(input.Shape, DType.F32);
        try
        {
            float* ip = (float*)input.DataPointer;
            float* gp = (float*)gamma.DataPointer;
            float* bp = (float*)beta.DataPointer;
            // Channel 0: x = [0, 1, 2, 3] → mean=1.5, var=1.25, std=√1.25
            // Channel 1: x = [10, 20, 30, 40] → mean=25, var=125, std=√125
            for (int j = 0; j < t; j++) ip[j] = j;
            for (int j = 0; j < t; j++) ip[t + j] = (j + 1) * 10f;
            gp[0] = 0.5f; gp[1] = -0.25f;
            bp[0] = 1.0f; bp[1] = -2.0f;

            backend.AdaInstanceNorm1d(output, input, gamma, beta, eps: 1e-5f);
            float* op = (float*)output.DataPointer;

            // Channel 0 closed form.
            float mean0 = 1.5f;
            float std0 = MathF.Sqrt(1.25f + 1e-5f);
            for (int j = 0; j < t; j++)
            {
                float expected = (1f + 0.5f) * (j - mean0) / std0 + 1.0f;
                Assert.Equal(expected, op[j], precision: 4);
            }
            // Channel 1.
            float mean1 = 25f;
            float std1 = MathF.Sqrt(125f + 1e-5f);
            for (int j = 0; j < t; j++)
            {
                float expected = (1f - 0.25f) * ((j + 1) * 10f - mean1) / std1 + -2.0f;
                Assert.Equal(expected, op[t + j], precision: 3);
            }
        }
        finally
        {
            input.Dispose(); gamma.Dispose(); beta.Dispose(); output.Dispose();
        }
    }

    [Fact]
    public void Rank1GammaBetaBroadcastsAcrossBatch()
    {
        // gamma / beta given as [C] should produce the same result as duplicating to [B, C].
        using CpuBackend backend = new();
        int batch = 3, channels = 4, t = 8;
        Tensor input = MakeRandom(batch, channels, t, seed: 9);
        Tensor gamma1 = new(new TensorShape(channels), DType.F32);
        Tensor beta1 = new(new TensorShape(channels), DType.F32);
        Tensor gammaN = new(new TensorShape(batch, channels), DType.F32);
        Tensor betaN = new(new TensorShape(batch, channels), DType.F32);
        Tensor outRank1 = new(input.Shape, DType.F32);
        Tensor outRank2 = new(input.Shape, DType.F32);
        try
        {
            float* g1 = (float*)gamma1.DataPointer;
            float* b1 = (float*)beta1.DataPointer;
            float* gN = (float*)gammaN.DataPointer;
            float* bN = (float*)betaN.DataPointer;
            for (int c = 0; c < channels; c++) { g1[c] = 0.1f * (c + 1); b1[c] = -0.05f * (c + 1); }
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < channels; c++) { gN[b * channels + c] = g1[c]; bN[b * channels + c] = b1[c]; }

            backend.AdaInstanceNorm1d(outRank1, input, gamma1, beta1, eps: 1e-5f);
            backend.AdaInstanceNorm1d(outRank2, input, gammaN, betaN, eps: 1e-5f);

            long n = input.ElementCount;
            float* o1 = (float*)outRank1.DataPointer;
            float* o2 = (float*)outRank2.DataPointer;
            for (long i = 0; i < n; i++) Assert.Equal(o2[i], o1[i], precision: 5);
        }
        finally
        {
            input.Dispose(); gamma1.Dispose(); beta1.Dispose();
            gammaN.Dispose(); betaN.Dispose(); outRank1.Dispose(); outRank2.Dispose();
        }
    }

    [Fact]
    public void PerBatchAffineActsIndependently()
    {
        // Two batch entries with the same input but different gamma/beta should
        // produce different per-batch outputs.
        using CpuBackend backend = new();
        int batch = 2, channels = 1, t = 4;
        Tensor input = new(new TensorShape(batch, channels, t), DType.F32);
        Tensor gamma = new(new TensorShape(batch, channels), DType.F32);
        Tensor beta = new(new TensorShape(batch, channels), DType.F32);
        Tensor output = new(input.Shape, DType.F32);
        try
        {
            float* ip = (float*)input.DataPointer;
            // Identical input slices.
            for (int b = 0; b < batch; b++)
                for (int j = 0; j < t; j++) ip[b * t + j] = j;

            float* gp = (float*)gamma.DataPointer;
            float* bp = (float*)beta.DataPointer;
            gp[0] = 0f; gp[1] = 1f;
            bp[0] = 0f; bp[1] = 5f;

            backend.AdaInstanceNorm1d(output, input, gamma, beta, eps: 1e-5f);
            float* op = (float*)output.DataPointer;

            // Batch-0 affine is identity: mean(x)=1.5, std=√1.25 → output should be
            // (j - 1.5) / √1.25 for j∈[0..3].
            float std = MathF.Sqrt(1.25f + 1e-5f);
            for (int j = 0; j < t; j++) Assert.Equal((j - 1.5f) / std, op[j], precision: 4);
            // Batch-1 has gamma=1 (scale 2x) + beta=5 → output = 2*(j-1.5)/std + 5.
            for (int j = 0; j < t; j++) Assert.Equal(2f * (j - 1.5f) / std + 5f, op[t + j], precision: 4);
        }
        finally
        {
            input.Dispose(); gamma.Dispose(); beta.Dispose(); output.Dispose();
        }
    }

    private static Tensor MakeRandom(int batch, int channels, int t, int seed)
    {
        Tensor x = new(new TensorShape(batch, channels, t), DType.F32);
        Random rng = new(seed);
        float* p = (float*)x.DataPointer;
        long n = x.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
    }

    private static Tensor ZeroTensor(int batch, int channels)
    {
        Tensor x = new(new TensorShape(batch, channels), DType.F32);
        float* p = (float*)x.DataPointer;
        for (long i = 0; i < x.ElementCount; i++) p[i] = 0f;
        return x;
    }
}
