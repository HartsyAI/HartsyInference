using HartsyInference.Video.Pipelines;
using Xunit;

namespace HartsyInference.Video.Tests;

/// <summary>The Lab (D65) mean/std colour matching that keeps chunked Wan-Animate runs anchored to the
/// reference image. The identity and drift cases pin the conversion round-trip; the strength-0 case pins the
/// byte-level no-op guarantee single-chunk generations rely on.</summary>
public sealed class VideoColorMatchTests
{
    private const int W = 64, H = 48;

    /// <summary>Deterministic multi-hue test image with smooth gradients and enough spread per channel.</summary>
    private static byte[] TestImage()
    {
        byte[] rgb = new byte[W * H * 3];
        for (int y = 0; y < H; y++)
        {
            for (int x = 0; x < W; x++)
            {
                int o = (y * W + x) * 3;
                rgb[o] = (byte)(20 + x * 3);
                rgb[o + 1] = (byte)(200 - y * 3);
                rgb[o + 2] = (byte)(60 + ((x + y) * 2 % 160));
            }
        }
        return rgb;
    }

    [Fact]
    public void MatchingAFrameToItsOwnStatsIsIdentity()
    {
        byte[] frame = TestImage();
        byte[] original = (byte[])frame.Clone();
        VideoColorMatch.LabStats own = VideoColorMatch.ComputeStats(frame, W, H);
        VideoColorMatch.MatchToReference(frame, W, H, own, strength: 1f);
        // Scale 1 / offset 0 in Lab, so the only possible movement is conversion round-trip error; double
        // math keeps that below the byte quantization step.
        Assert.Equal(original, frame);
    }

    [Fact]
    public void StrengthZeroIsAByteLevelNoOp()
    {
        byte[] frame = TestImage();
        byte[] original = (byte[])frame.Clone();
        VideoColorMatch.LabStats reference = VideoColorMatch.ComputeStats(new byte[W * H * 3], W, H);   // all black — maximally different
        VideoColorMatch.MatchToReference(frame, W, H, reference, strength: 0f);
        Assert.Equal(original, frame);
        VideoColorMatch.MatchToReference(frame, W, H, reference, strength: -1f);
        Assert.Equal(original, frame);
    }

    [Fact]
    public void SyntheticDriftIsMatchedBackToTheReferenceStats()
    {
        byte[] reference = TestImage();
        VideoColorMatch.LabStats refStats = VideoColorMatch.ComputeStats(reference, W, H);

        // The observed failure mode: a brightened, warm-tinted, contrast-lifted copy (what compounding
        // VAE round-trips do to later chunks).
        byte[] drifted = new byte[reference.Length];
        for (int i = 0; i < reference.Length; i += 3)
        {
            drifted[i] = (byte)Math.Clamp(reference[i] * 1.15 + 25, 0, 255);
            drifted[i + 1] = (byte)Math.Clamp(reference[i + 1] * 1.05 + 12, 0, 255);
            drifted[i + 2] = (byte)Math.Clamp(reference[i + 2] * 0.9 + 4, 0, 255);
        }
        VideoColorMatch.LabStats before = VideoColorMatch.ComputeStats(drifted, W, H);
        Assert.True(Math.Abs(before.MeanL - refStats.MeanL) > 5,
            $"Drift setup too weak to be a meaningful test: ΔL = {before.MeanL - refStats.MeanL:F2}");

        VideoColorMatch.MatchToReference(drifted, W, H, refStats, strength: 1f);
        VideoColorMatch.LabStats after = VideoColorMatch.ComputeStats(drifted, W, H);

        // The drift saturates some pixels (as real drift does), and clamped pixels cannot be un-clamped by a
        // moment match — so within 1 Lab unit, not exact. 1.0 is well under a just-noticeable ΔE (~2.3) and
        // far under the injected drift.
        Assert.True(Math.Abs(after.MeanL - refStats.MeanL) < 1.0, $"Mean L off by {after.MeanL - refStats.MeanL:F3}");
        Assert.True(Math.Abs(after.MeanA - refStats.MeanA) < 1.0, $"Mean a off by {after.MeanA - refStats.MeanA:F3}");
        Assert.True(Math.Abs(after.MeanB - refStats.MeanB) < 1.0, $"Mean b off by {after.MeanB - refStats.MeanB:F3}");
        Assert.True(Math.Abs(after.StdL - refStats.StdL) < 1.5, $"Std L off by {after.StdL - refStats.StdL:F3}");
        Assert.True(Math.Abs(after.StdA - refStats.StdA) < 1.5, $"Std a off by {after.StdA - refStats.StdA:F3}");
        Assert.True(Math.Abs(after.StdB - refStats.StdB) < 1.5, $"Std b off by {after.StdB - refStats.StdB:F3}");
    }

    [Fact]
    public void HalfStrengthLandsBetweenUntouchedAndFullyMatched()
    {
        byte[] reference = TestImage();
        VideoColorMatch.LabStats refStats = VideoColorMatch.ComputeStats(reference, W, H);
        byte[] drifted = new byte[reference.Length];
        for (int i = 0; i < reference.Length; i++)
        {
            drifted[i] = (byte)Math.Clamp(reference[i] + 40, 0, 255);
        }
        double startGap = Math.Abs(VideoColorMatch.ComputeStats(drifted, W, H).MeanL - refStats.MeanL);
        VideoColorMatch.MatchToReference(drifted, W, H, refStats, strength: 0.5f);
        double halfGap = Math.Abs(VideoColorMatch.ComputeStats(drifted, W, H).MeanL - refStats.MeanL);
        Assert.True(halfGap < startGap * 0.7 && halfGap > startGap * 0.3,
            $"Half strength should close roughly half the L gap: {startGap:F2} → {halfGap:F2}");
    }

    [Fact]
    public void AFlatFrameMatchesByMeanShiftWithoutAmplifyingNoise()
    {
        byte[] reference = TestImage();
        VideoColorMatch.LabStats refStats = VideoColorMatch.ComputeStats(reference, W, H);
        byte[] flat = new byte[W * H * 3];
        Array.Fill(flat, (byte)128);
        VideoColorMatch.MatchToReference(flat, W, H, refStats, strength: 1f);
        // Zero std → scale guard holds it at 1: the frame shifts to the reference mean but stays flat.
        VideoColorMatch.LabStats after = VideoColorMatch.ComputeStats(flat, W, H);
        Assert.True(Math.Abs(after.MeanL - refStats.MeanL) < 1.0, $"Mean L off by {after.MeanL - refStats.MeanL:F3}");
        Assert.True(after.StdL < 1.0, $"A flat frame must stay flat; got std L {after.StdL:F3}");
    }

    [Fact]
    public void RejectsAShortBuffer()
    {
        Assert.Throws<ArgumentException>(() => VideoColorMatch.ComputeStats(new byte[10], W, H));
        Assert.Throws<ArgumentException>(() => VideoColorMatch.MatchToReference(new byte[10], W, H, default, 1f));
    }
}
