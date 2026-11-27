using HartsyInference.Engine.Worlds;

namespace HartsyInference.WorldViewer.Tests;

public sealed class WorldSteeringTests
{
    [Fact]
    public void On_change_scheme_sends_the_held_set_only_when_it_changes()
    {
        ScriptedWorldSession session = new(_ => TestFrames.Run(0, 0));
        WorldSteering steering = new(session, SanaWmControlScheme.Instance);

        Assert.True(steering.Press("W"));
        Assert.False(steering.Press("w"));
        Assert.True(steering.Press("A"));
        Assert.True(steering.Release("W"));
        steering.ReleaseAll();

        Assert.Equal(["w", "wa", "a", "none"], session.Actions);
        Assert.Empty(steering.HeldKeys());
    }

    [Fact]
    public void Keys_the_scheme_does_not_declare_change_nothing()
    {
        ScriptedWorldSession session = new(_ => TestFrames.Run(0, 0));
        WorldSteering steering = new(session, SanaWmControlScheme.Instance);

        Assert.False(steering.Press("Escape"));
        Assert.False(steering.Release("Space"));
        Assert.Empty(session.Actions);
    }

    [Fact]
    public void A_null_scheme_tracks_nothing()
    {
        ScriptedWorldSession session = new(_ => TestFrames.Run(0, 0));
        WorldSteering steering = new(session, null);

        Assert.False(steering.Press("W"));
        Assert.Equal("none", steering.CurrentAction());
        Assert.Empty(session.Actions);
    }

    [Fact]
    public void Per_step_scheme_sends_nothing_on_key_changes_and_reports_the_current_action()
    {
        PerStepWorldSession session = new();
        WorldSteering steering = new(session, DiamondControlScheme.Instance);

        Assert.True(steering.Press("Space"));
        Assert.Equal("fire", steering.CurrentAction());
        Assert.Empty(session.Taken);
    }

    [Fact]
    public void Closing_stops_all_sending()
    {
        ScriptedWorldSession session = new(_ => TestFrames.Run(0, 0));
        WorldSteering steering = new(session, SanaWmControlScheme.Instance);

        steering.Press("W");
        steering.Close();

        Assert.False(steering.Press("A"));
        steering.ReleaseAll();
        Assert.Equal(["w"], session.Actions);
    }
}
