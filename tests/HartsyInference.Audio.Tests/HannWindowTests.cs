using HartsyInference.Audio.Preprocessing;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Verifies our Hann window matches the PERIODIC convention
/// (matching torch.hann_window(N, periodic=True) and scipy.signal.windows.hann(N, sym=False)),
/// not the symmetric form. Getting this wrong silently breaks every mel-input audio model.</summary>
public sealed class HannWindowTests
{
    [Fact]
    public void Periodic_Length_Matches_Spec_N400()
    {
        // For N=400 (Whisper) the periodic Hann is w[n] = 0.5 - 0.5*cos(2*pi*n/N).
        // First sample is 0, last sample is 0.5 - 0.5*cos(2*pi*399/400) which is NOT 0.
        // The symmetric form would have the last sample at 0 because divisor would be N-1=399.
        float[] w = HannWindow.Get(400);
        Assert.Equal(400, w.Length);
        Assert.Equal(0f, w[0], precision: 6);

        // Symmetry check: periodic Hann is symmetric about N/2, so w[n] == w[N-n]
        // for n=1..N/2. But w[N-1] != w[0] (that's the periodic distinction).
        // fp32 cos rounding is asymmetric across the unit circle, so equal values up to
        // ~1e-6 absolute is the tightest contract we can hold.
        for (int n = 1; n < 200; n++)
        {
            Assert.True(MathF.Abs(w[n] - w[400 - n]) < 1e-5f,
                $"symmetry at n={n}: w[{n}]={w[n]}, w[{400 - n}]={w[400 - n]}");
        }

        // Peak at center: w[200] = 0.5 - 0.5 * cos(pi) = 1.0
        Assert.Equal(1.0f, w[200], precision: 6);
    }

    [Fact]
    public void Get_Is_Cached_And_Returns_Shared_Array()
    {
        float[] a = HannWindow.Get(1024);
        float[] b = HannWindow.Get(1024);
        Assert.Same(a, b);
    }

    [Fact]
    public void DifferentSizes_HaveDifferentArrays()
    {
        float[] a = HannWindow.Get(400);
        float[] b = HannWindow.Get(512);
        Assert.NotSame(a, b);
        Assert.Equal(400, a.Length);
        Assert.Equal(512, b.Length);
    }
}
