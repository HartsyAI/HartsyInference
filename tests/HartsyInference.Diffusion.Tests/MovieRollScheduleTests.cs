using System.Text.Json;
using HartsyInference.Diffusion.Pipelines.MovieRoll;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>MovieRoll's sigma ladders and rolling step-index table, against <c>Fixtures/movieroll_schedule_reference.json</c>, a transcription of the
/// reference's <c>FlowMatchDiscreteScheduler.set_timesteps</c>, <c>set_custom_timesteps</c> and <c>StreamingSchedulerWrapper.get_step_index</c>
/// (run in the CPU reference image). The reference is a transcription of the scheduler math, not the diffusers module itself.</summary>
public sealed class MovieRollScheduleTests
{
    private const float Tolerance = 1e-5f;

    [Fact]
    public void BaseLadderMatchesTheReference()
    {
        using JsonDocument reference = Load();
        JsonElement baseRef = reference.RootElement.GetProperty("base");
        MovieRollSchedule schedule = MovieRollSchedule.Base(50, 5, 5.0);

        AssertClose(baseRef.GetProperty("sigmas"), schedule.Sigmas, "base sigmas");
        AssertClose(baseRef.GetProperty("timesteps"), schedule.Timesteps, "base timesteps", 1e-3f);
        Assert.Equal(baseRef.GetProperty("steps_per_window").GetInt32(), schedule.StepsPerWindow);
    }

    [Fact]
    public void DistilledLadderMatchesTheReference()
    {
        using JsonDocument reference = Load();
        JsonElement distilledRef = reference.RootElement.GetProperty("distilled");
        MovieRollSchedule schedule = MovieRollSchedule.Distilled([1000, 800, 600, 400, 200], 5, 5.0);

        AssertClose(distilledRef.GetProperty("sigmas"), schedule.Sigmas, "distilled sigmas");
        AssertClose(distilledRef.GetProperty("timesteps"), schedule.Timesteps, "distilled timesteps", 1e-3f);
        Assert.Equal(1, schedule.StepsPerWindow);
    }

    [Fact]
    public void RollingStepIndicesMatchTheReferenceForBothChains()
    {
        using JsonDocument reference = Load();
        AssertIndices(reference.RootElement.GetProperty("base"), MovieRollSchedule.Base(50, 5, 5.0), "base");
        AssertIndices(reference.RootElement.GetProperty("distilled"), MovieRollSchedule.Distilled([1000, 800, 600, 400, 200], 5, 5.0), "distilled");
    }

    [Fact]
    public void DistilledLadderNeedsOneStepPerWindowSlot()
    {
        Assert.Throws<ArgumentException>(() => MovieRollSchedule.Distilled([1000, 800, 600], 5, 5.0));
        Assert.Throws<ArgumentException>(() => MovieRollSchedule.Distilled([1000, 800, 800, 400, 200], 5, 5.0));
    }

    [Fact]
    public void BaseRefusesAStepCountThatTheWindowDoesNotDivide()
    {
        Assert.Throws<ArgumentException>(() => MovieRollSchedule.Base(50, 4, 5.0));
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-5.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void ShiftMustBeFiniteAndPositive(double shift)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => MovieRollSchedule.Base(50, 5, shift));
        Assert.Throws<ArgumentOutOfRangeException>(() => MovieRollSchedule.Distilled([1000, 800, 600, 400, 200], 5, shift));
    }

    private static void AssertIndices(JsonElement chain, MovieRollSchedule schedule, string label)
    {
        JsonElement globals = chain.GetProperty("index_globals");
        JsonElement expected = chain.GetProperty("index");
        int row = 0;
        foreach (JsonElement global in globals.EnumerateArray())
        {
            int[] actual = schedule.StepIndices(global.GetInt32());
            int col = 0;
            foreach (JsonElement value in expected[row].EnumerateArray())
                Assert.True(value.GetInt32() == actual[col++], $"{label} step indices at global {global.GetInt32()} differ.");
            row++;
        }
    }

    private static void AssertClose(JsonElement expected, float[] actual, string label, float tolerance = Tolerance)
    {
        Assert.Equal(expected.GetArrayLength(), actual.Length);
        int i = 0;
        foreach (JsonElement value in expected.EnumerateArray())
        {
            Assert.True(Math.Abs(value.GetDouble() - actual[i]) <= tolerance, $"{label}[{i}]: {actual[i]} vs {value.GetDouble()}.");
            i++;
        }
    }

    private static JsonDocument Load() =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "movieroll_schedule_reference.json")));
}
