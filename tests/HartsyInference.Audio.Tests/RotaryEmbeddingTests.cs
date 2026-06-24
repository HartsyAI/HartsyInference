using System;
using HartsyInference.Audio.Models.Moonshine;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests for <see cref="RotaryEmbedding"/> Llama-3 NTK-by-parts RoPE scaling. Verifies the no-scaling
/// path is preserved, the piecewise rule is applied correctly at the frequency-band endpoints, and the produced
/// cos/sin tables match an independent computation of the rescaled inverse frequencies.</summary>
public sealed class RotaryEmbeddingTests
{
    [Fact]
    public void NullScaling_IsBitIdenticalToUnscaledOverload()
    {
        const int rotaryDim = 64, maxPos = 8;
        const float theta = 500_000f;
        (float[] cosU, float[] sinU) = RotaryEmbedding.GetTables(rotaryDim, theta, maxPos);
        (float[] cosN, float[] sinN) = RotaryEmbedding.GetTables(rotaryDim, theta, maxPos, null);
        Assert.Equal(cosU, cosN);
        Assert.Equal(sinU, sinN);
    }

    [Fact]
    public void Llama3Scaling_HighFreqUntouched_LowFreqDividedByFactor()
    {
        const int rotaryDim = 64;
        const float theta = 500_000f;
        RopeScaling sc = RopeScaling.Llama3; // factor 8, low 1, high 4, original_max_pos 8192
        int half = rotaryDim / 2;

        double[] expected = ExpectedLlama3InvFreq(rotaryDim, theta, sc);

        // i=0 is the highest frequency (shortest wavelength) — must be untouched by the rule.
        double baseFirst = 1.0 / Math.Pow(theta, 0.0);
        Assert.Equal(baseFirst, expected[0], 12);

        // The lowest-frequency pair (longest wavelength) must be divided by the factor.
        double baseLast = 1.0 / Math.Pow(theta, (double)(2 * (half - 1)) / rotaryDim);
        Assert.Equal(baseLast / sc.Factor, expected[half - 1], 12);
        Assert.True(expected[half - 1] < baseLast * 0.5, "low-frequency invFreq should be markedly reduced");
    }

    [Fact]
    public void Llama3Scaling_TablesMatchIndependentInvFreqComputation()
    {
        const int rotaryDim = 64, maxPos = 16;
        const float theta = 500_000f;
        RopeScaling sc = RopeScaling.Llama3;
        int half = rotaryDim / 2;

        double[] invFreq = ExpectedLlama3InvFreq(rotaryDim, theta, sc);
        (float[] cos, float[] sin) = RotaryEmbedding.GetTables(rotaryDim, theta, maxPos, sc);

        for (int p = 0; p < maxPos; p++)
        {
            for (int i = 0; i < half; i++)
            {
                double angle = p * invFreq[i];
                Assert.Equal((float)Math.Cos(angle), cos[p * half + i], 5);
                Assert.Equal((float)Math.Sin(angle), sin[p * half + i], 5);
            }
        }
    }

    /// <summary>Independent reference for the Llama-3 piecewise rescale (HF <c>_compute_llama3_parameters</c>).</summary>
    private static double[] ExpectedLlama3InvFreq(int rotaryDim, float theta, RopeScaling sc)
    {
        int half = rotaryDim / 2;
        double lowWavelen = sc.OriginalMaxPos / sc.LowFreqFactor;
        double highWavelen = sc.OriginalMaxPos / sc.HighFreqFactor;
        double[] outInv = new double[half];
        for (int i = 0; i < half; i++)
        {
            double baseInv = 1.0 / Math.Pow(theta, (double)(2 * i) / rotaryDim);
            double wavelen = 2.0 * Math.PI / baseInv;
            if (wavelen < highWavelen)
            {
                outInv[i] = baseInv;
            }
            else if (wavelen > lowWavelen)
            {
                outInv[i] = baseInv / sc.Factor;
            }
            else
            {
                double smooth = (sc.OriginalMaxPos / wavelen - sc.LowFreqFactor) / (sc.HighFreqFactor - sc.LowFreqFactor);
                outInv[i] = (1.0 - smooth) * (baseInv / sc.Factor) + smooth * baseInv;
            }
        }
        return outInv;
    }
}
