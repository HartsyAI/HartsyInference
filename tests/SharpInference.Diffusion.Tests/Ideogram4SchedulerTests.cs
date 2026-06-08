using Xunit;
using SharpInference.Diffusion.Schedulers;

namespace SharpInference.Diffusion.Tests;

/// <summary>Pure-logic tests for Ideogram 4's logit-normal schedule and sampler presets (no GPU needed).</summary>
public class Ideogram4SchedulerTests
{
    [Fact]
    public void Schedule_IsMonotonicDecreasing_AndWithinClampWindow()
    {
        LogitNormalSchedule schedule = new(mean: 1.0, std: 1.0);
        float[] grid = LogitNormalSchedule.MakeStepIntervals(20);

        // tMin = 1/(1+e^9) ≈ 1.23e-4 ; tMax = 1/(1+e^-7.5) ≈ 0.99944.
        const float tMin = 1.0e-4f;
        const float tMax = 0.99955f;

        float prev = float.MaxValue;
        for (int i = 0; i < grid.Length; i++)
        {
            float t = schedule.Map(grid[i]);
            Assert.InRange(t, tMin, tMax);
            Assert.True(t <= prev + 1e-6f, $"schedule not monotonic decreasing at grid[{i}]: {t} > {prev}");
            prev = t;
        }
    }

    [Fact]
    public void MakeStepIntervals_HasLinearGridOfLengthStepsPlusOne()
    {
        float[] grid = LogitNormalSchedule.MakeStepIntervals(12);
        Assert.Equal(13, grid.Length);
        Assert.Equal(0.0f, grid[0]);
        Assert.Equal(1.0f, grid[12], 5);
        Assert.Equal(0.5f, grid[6], 5);
    }

    [Fact]
    public void ForResolution_IncreasesMeanWithPixelCount()
    {
        // Larger images push the mean up (more noise budget), so the t at a fixed grid point drops.
        LogitNormalSchedule small = LogitNormalSchedule.ForResolution(512, 512, knownMean: 0.0, std: 1.0);
        LogitNormalSchedule large = LogitNormalSchedule.ForResolution(2048, 2048, knownMean: 0.0, std: 1.0);
        Assert.True(large.Map(0.5f) < small.Map(0.5f));
    }

    [Fact]
    public void Presets_HaveMatchingScheduleLengthsAndPolishTail()
    {
        foreach (Ideogram4SamplerPreset p in new[]
                 { Ideogram4SamplerPreset.Quality48, Ideogram4SamplerPreset.Default20, Ideogram4SamplerPreset.Turbo12 })
        {
            Assert.Equal(p.NumSteps, p.GuidanceSchedule.Length);
            // Index 0 is the LAST (polish) step → low guidance; the final entry is a main step → high guidance.
            Assert.Equal(3.0f, p.GuidanceSchedule[0]);
            Assert.Equal(7.0f, p.GuidanceSchedule[^1]);
        }
    }

    [Fact]
    public void ByName_ResolvesPresetsAndRejectsUnknown()
    {
        Assert.Equal(48, Ideogram4SamplerPreset.ByName("V4_QUALITY_48").NumSteps);
        Assert.Equal(20, Ideogram4SamplerPreset.ByName("v4_default_20").NumSteps);
        Assert.Throws<ArgumentException>(() => Ideogram4SamplerPreset.ByName("nope"));
    }
}
