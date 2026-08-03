using Xunit;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Gates MiniMax-H3's dual flow schedule: the audio stream is derived from the video sigma in closed form,
/// and its velocity scaling must be the exact derivative of that map or the two streams desynchronise.</summary>
public class MiniMaxH3ScheduleTests
{
    private const double ShiftV = MiniMaxH3Schedule.DefaultShiftVideo;
    private const double ShiftA = MiniMaxH3Schedule.DefaultShiftAudio;

    [Fact]
    public void ShiftingToTheSameScheduleIsIdentity()
    {
        foreach (double s in new[] { 0.05, 0.3, 0.7, 1.0 })
        {
            Assert.Equal(s, MiniMaxH3Schedule.ShiftSigma(s, ShiftV, ShiftV), 10);
        }
    }

    [Fact]
    public void ShiftingIsInvertible()
    {
        foreach (double s in new[] { 0.05, 0.3, 0.7, 1.0 })
        {
            double a = MiniMaxH3Schedule.ShiftSigma(s, ShiftV, ShiftA);
            Assert.Equal(s, MiniMaxH3Schedule.ShiftSigma(a, ShiftA, ShiftV), 9);
        }
    }

    [Fact]
    public void EndpointsArePinned()
    {
        Assert.Equal(0.0, MiniMaxH3Schedule.ShiftSigma(0.0, ShiftV, ShiftA), 12);
        Assert.Equal(1.0, MiniMaxH3Schedule.ShiftSigma(1.0, ShiftV, ShiftA), 12);
    }

    [Fact]
    public void SlopeIsTheDerivativeOfTheSigmaMap()
    {
        // Central difference against the closed form — a wrong slope silently desynchronises audio from video.
        const double h = 1e-6;
        foreach (double s in new[] { 0.1, 0.35, 0.6, 0.9 })
        {
            double numeric = (MiniMaxH3Schedule.ShiftSigma(s + h, ShiftV, ShiftA)
                            - MiniMaxH3Schedule.ShiftSigma(s - h, ShiftV, ShiftA)) / (2 * h);
            double closed = MiniMaxH3Schedule.ShiftSlope(s, ShiftV, ShiftA);
            Assert.Equal(numeric, closed, 5);
        }
    }

    [Fact]
    public void AudioRunsAheadOfVideoUnderTheSmallerShift()
    {
        // A smaller shift front-loads the schedule, so at the same sampler position the audio sigma is lower.
        foreach (double s in new[] { 0.2, 0.5, 0.8 })
        {
            Assert.True(MiniMaxH3Schedule.ShiftSigma(s, ShiftV, ShiftA) < s);
        }
    }

    [Fact]
    public void VideoSigmasDescendFromShiftedOneToZero()
    {
        double[] sigmas = MiniMaxH3Schedule.VideoSigmas(8, ShiftV);
        Assert.Equal(9, sigmas.Length);
        Assert.Equal(1.0, sigmas[0], 9);
        Assert.Equal(0.0, sigmas[^1], 12);
        for (int i = 1; i < sigmas.Length; i++)
        {
            Assert.True(sigmas[i] < sigmas[i - 1], $"sigma must descend at {i}");
        }
    }

    [Fact]
    public void TimestepsAreOneMinusSigmaOnEachStreamsOwnSchedule()
    {
        (float tv, float ta) = MiniMaxH3Schedule.Timesteps(0.5, ShiftV, ShiftA);
        Assert.Equal(0.5f, tv, 5);
        Assert.Equal((float)(1.0 - MiniMaxH3Schedule.ShiftSigma(0.5, ShiftV, ShiftA)), ta, 5);
        // Audio is further along its schedule, so its timestep is larger.
        Assert.True(ta > tv);
    }

    [Fact]
    public void StepCountIsValidated()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => MiniMaxH3Schedule.VideoSigmas(0, ShiftV));
    }
}
