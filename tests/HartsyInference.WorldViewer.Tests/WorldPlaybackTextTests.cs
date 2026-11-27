using HartsyInference.Engine.Worlds;

namespace HartsyInference.WorldViewer.Tests;

public sealed class WorldPlaybackTextTests
{
    [Fact]
    public void Summary_names_the_held_keys_the_legend_and_the_note()
    {
        WorldPlaybackStatus status = new(WorldPlaybackPhase.Playing, 16, 24, 32, 40, 64, 15.8, null);

        string text = WorldPlaybackText.Summary(status, ["W", "A"], SanaWmControlScheme.Instance);

        Assert.Contains("Playing", text);
        Assert.Contains("Buffered 1.5 s (24/32 frames)", text);
        Assert.Contains("Delivered 15.8 fps against 16 to play", text);
        Assert.Contains("Held: W, A", text);
        Assert.Contains("W forward", text);
        Assert.Contains("moves the camera, not a character", text);
    }

    [Fact]
    public void A_model_with_no_live_control_says_the_clip_plays_when_the_run_completes()
    {
        WorldPlaybackStatus status = new(WorldPlaybackPhase.Generating, 16, 0, 32, 0, 0, 0, null);

        string text = WorldPlaybackText.Summary(status, [], null);

        Assert.Contains("No live control for this model", text);
        Assert.DoesNotContain("Held:", text);
        Assert.Equal("HartsyInference world viewer", WorldPlaybackText.Title(null));
    }

    [Fact]
    public void A_failed_run_shows_its_error()
    {
        WorldPlaybackStatus status = new(WorldPlaybackPhase.Failed, 16, 0, 32, 4, 4, 0, "out of memory");

        Assert.Contains("Failed: out of memory", WorldPlaybackText.Summary(status, [], null));
    }
}
