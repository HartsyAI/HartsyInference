using HartsyInference.Engine.Recipes;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the video defaults taken from SwarmUI core (RECIPE_CONTRACT.md ledger), so a value that drifts from
/// its cited source fails here instead of silently changing every unset request.</summary>
public sealed class VideoRecipeSwarmDefaultsTests
{
    [Theory]
    [InlineData("wan-vace", 50, 6.0f, 704, 480, 81, 24)]
    [InlineData("ltx-video", 50, 3.0f, 768, 512, 97, 24)]
    [InlineData("kandinsky5-video", 50, 1.0f, 640, 640, 73, 24)]
    [InlineData("hunyuan-video", 20, 6.0f, 720, 720, 73, 24)]
    public void Defaults_MatchTheLedger(string familyId, int steps, float cfg, int width, int height, int frames, int fps)
    {
        VideoDefaults defaults = VideoRecipeRegistry.Resolve(familyId)!.Defaults;

        Assert.Equal((steps, cfg, width, height, frames, fps, 2),
            (defaults.Steps, defaults.CfgScale, defaults.Width, defaults.Height, defaults.Frames, defaults.Fps, defaults.Revision));
    }
}
