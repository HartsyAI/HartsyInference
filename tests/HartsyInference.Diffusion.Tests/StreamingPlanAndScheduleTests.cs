using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Streaming;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary><see cref="ChunkPlanner"/> spans for bounded and unbounded plans, and <see cref="SelfForcingFlowSchedule"/>'s input
/// checks. The boundaries and the schedule values themselves are checked against upstream in
/// <see cref="SanaWmStreamingUpstreamTests"/>.</summary>
public sealed class StreamingPlanAndScheduleTests
{
    [Theory]
    [InlineData(ChunkSplitStrategy.FirstChunkPlusOne, new[] { 0, 4, 7, 10, 13 })]
    [InlineData(ChunkSplitStrategy.FirstFrameAlone, new[] { 0, 1, 4, 7, 10 })]
    [InlineData(ChunkSplitStrategy.Uniform, new[] { 0, 3, 6, 9, 12 })]
    public void Unbounded_SpansRunOnWithoutEnd(ChunkSplitStrategy strategy, int[] starts)
    {
        ChunkPlanner planner = new(3, strategy);
        Assert.Null(planner.TotalFrames);
        Assert.Null(planner.ChunkCount);
        Assert.Equal(starts[1], planner.FirstChunkFrames);
        for (int i = 0; i + 1 < starts.Length; i++)
        {
            ChunkSpan span = planner.Span(i);
            Assert.Equal((i, starts[i], starts[i + 1]), (span.Index, span.Start, span.End));
            Assert.False(planner.IsLast(i));
        }

        ChunkSpan far = planner.Span(10_000);
        Assert.Equal(3, far.Length);
        Assert.Throws<ArgumentOutOfRangeException>(() => planner.Span(-1));
    }

    [Theory]
    [InlineData(10, new[] { 0, 4, 7, 10 })]
    [InlineData(31, new[] { 0, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31 })]
    [InlineData(11, new[] { 0, 5, 8, 11 })]
    [InlineData(3, new[] { 0, 3 })]
    public void Bounded_SpansFollowTheBoundaries(int total, int[] bounds)
    {
        ChunkPlanner planner = new(3, ChunkSplitStrategy.FirstAbsorbsRemainder, total);
        Assert.Equal(bounds, ChunkPlanner.Boundaries(total, 3, ChunkSplitStrategy.FirstAbsorbsRemainder));
        Assert.Equal(bounds.Length - 1, planner.ChunkCount);
        for (int i = 0; i + 1 < bounds.Length; i++)
        {
            ChunkSpan span = planner.Span(i);
            Assert.Equal((bounds[i], bounds[i + 1]), (span.Start, span.End));
            Assert.Equal(i == bounds.Length - 2, planner.IsLast(i));
        }

        Assert.Throws<ArgumentOutOfRangeException>(() => planner.Span(bounds.Length - 1));
    }

    [Fact]
    public void FirstAbsorbsRemainder_NeedsATotalOfAtLeastOneChunk()
    {
        Assert.Throws<ArgumentException>(() => new ChunkPlanner(3, ChunkSplitStrategy.FirstAbsorbsRemainder));
        Assert.Throws<ArgumentException>(() => new ChunkPlanner(3, ChunkSplitStrategy.FirstAbsorbsRemainder, 2));
        Assert.Throws<ArgumentException>(() => ChunkPlanner.Boundaries(2, 3, ChunkSplitStrategy.FirstAbsorbsRemainder));
    }

    [Theory]
    [InlineData(new[] { 0 })]
    [InlineData(new[] { 1000, 500 })]
    [InlineData(new[] { 1000, 1000, 0 })]
    [InlineData(new[] { 960, 1000, 0 })]
    [InlineData(new[] { 1001, 0 })]
    [InlineData(new[] { 1000, -5, 0 })]
    public void Schedule_RefusesMalformedTimesteps(int[] timesteps) =>
        Assert.Throws<ArgumentException>(() => new SelfForcingFlowSchedule(timesteps));

    [Fact]
    public void Schedule_ParsesAndBoundsItsSteps()
    {
        Assert.Throws<FormatException>(() => SelfForcingFlowSchedule.Parse("1000,9x0,0"));
        Assert.Throws<FormatException>(() => SelfForcingFlowSchedule.Parse("1000,-1,0"));
        SelfForcingFlowSchedule schedule = SelfForcingFlowSchedule.Parse(" 1000, 960 ,889,727,0 ");
        Assert.Equal(4, schedule.Steps);
        Assert.Equal(0f, schedule.Sigma(4));
        Assert.Throws<ArgumentOutOfRangeException>(() => schedule.Timestep(4));
        Assert.Throws<ArgumentOutOfRangeException>(() => schedule.Sigma(5));

        float[] destination = new float[3];
        schedule.FrameTimesteps(2, [true, false, false], destination);
        Assert.Equal([0f, 889f, 889f], destination);
        Assert.Throws<ArgumentException>(() => schedule.FrameTimesteps(0, [false, false], destination));
    }

    [Fact]
    public void Step_LeavesCleanFramesBitExact_AndRoundsTheRestToBf16()
    {
        SelfForcingFlowSchedule schedule = SelfForcingFlowSchedule.Parse("1000,960,889,727,0");
        const int channels = 2, frames = 3, positions = 4;
        float[] latents = new float[channels * frames * positions];
        float[] velocity = new float[latents.Length];
        for (int i = 0; i < latents.Length; i++)
        {
            latents[i] = 0.123456789f * (i + 1);
            velocity[i] = -0.987654321f * (i % 5 + 1);
        }

        float[] before = (float[])latents.Clone();
        schedule.Step(latents, velocity, 0, channels, [false, true, false], roundToBf16: true);
        for (int c = 0; c < channels; c++)
        {
            for (int f = 0; f < frames; f++)
            {
                for (int p = 0; p < positions; p++)
                {
                    int i = (c * frames + f) * positions + p;
                    if (f == 1)
                    {
                        Assert.Equal(before[i], latents[i]);
                        continue;
                    }

                    float exact = before[i] + 0.04f * -velocity[i];
                    Assert.Equal(0u, BitConverter.SingleToUInt32Bits(latents[i]) & 0xFFFFu);
                    Assert.True(MathF.Abs(latents[i] - exact) <= MathF.Abs(exact) / 128f, $"element {i}: {latents[i]} vs {exact}");
                }
            }
        }

        Assert.Throws<ArgumentException>(() => schedule.Step(latents, velocity.AsSpan(1).ToArray(), 0, channels, [false, false, false],
            roundToBf16: false));
    }
}
