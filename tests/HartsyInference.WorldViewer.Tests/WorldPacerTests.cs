namespace HartsyInference.WorldViewer.Tests;

public sealed class WorldPacerTests
{
    private static TimeSpan AtTick(int tick, int perSecond = 240) => TimeSpan.FromTicks(TimeSpan.TicksPerSecond * tick / perSecond);

    [Fact]
    public void Plays_one_frame_per_period_when_the_buffer_is_full()
    {
        WorldPacer pacer = new(fps: 24, startFrames: 24, resumeFrames: 6);

        int shown = 0;
        for (int tick = 0; tick < 240; tick++)
        {
            if (pacer.Step(AtTick(tick), buffered: 1000, streamEnded: false) == WorldPaceStep.Show)
            {
                shown++;
            }
        }

        Assert.Equal(24, shown);
    }

    [Fact]
    public void Waits_for_the_start_buffer_before_the_first_frame()
    {
        WorldPacer pacer = new(fps: 16, startFrames: 16, resumeFrames: 4);

        Assert.Equal(WorldPaceStep.Buffering, pacer.Step(TimeSpan.Zero, buffered: 15, streamEnded: false));
        Assert.False(pacer.Started);
        Assert.Equal(WorldPaceStep.Show, pacer.Step(TimeSpan.Zero, buffered: 16, streamEnded: false));
        Assert.True(pacer.Started);
    }

    [Fact]
    public void An_underrun_rebuffers_and_resumes_at_the_resume_threshold()
    {
        WorldPacer pacer = new(fps: 16, startFrames: 16, resumeFrames: 4);
        pacer.Step(TimeSpan.Zero, buffered: 16, streamEnded: false);

        Assert.Equal(WorldPaceStep.Buffering, pacer.Step(TimeSpan.FromSeconds(0.1), buffered: 0, streamEnded: false));
        Assert.False(pacer.Playing);
        Assert.Equal(WorldPaceStep.Buffering, pacer.Step(TimeSpan.FromSeconds(0.2), buffered: 3, streamEnded: false));
        Assert.Equal(WorldPaceStep.Show, pacer.Step(TimeSpan.FromSeconds(0.3), buffered: 4, streamEnded: false));
        Assert.True(pacer.Playing);
    }

    [Fact]
    public void A_short_stream_plays_what_it_has_and_then_finishes()
    {
        WorldPacer pacer = new(fps: 16, startFrames: 16, resumeFrames: 4);

        Assert.Equal(WorldPaceStep.Show, pacer.Step(TimeSpan.Zero, buffered: 3, streamEnded: true));
        Assert.Equal(WorldPaceStep.Finished, pacer.Step(TimeSpan.FromSeconds(1), buffered: 0, streamEnded: true));
    }

    [Fact]
    public void A_stall_is_rebased_instead_of_replayed_as_a_burst()
    {
        WorldPacer pacer = new(fps: 16, startFrames: 1, resumeFrames: 1);
        pacer.Step(TimeSpan.Zero, buffered: 1000, streamEnded: false);

        TimeSpan stalled = TimeSpan.FromSeconds(10);
        int shownInHalfSecond = 0;
        for (int tick = 0; tick < 120; tick++)
        {
            if (pacer.Step(stalled + AtTick(tick), buffered: 1000, streamEnded: false) == WorldPaceStep.Show)
            {
                shownInHalfSecond++;
            }
        }

        Assert.InRange(shownInHalfSecond, 7, 9);
    }

    [Fact]
    public void Rejects_a_resume_threshold_above_the_start_threshold()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new WorldPacer(fps: 16, startFrames: 2, resumeFrames: 4));
    }
}
