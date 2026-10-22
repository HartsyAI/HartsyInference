using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Resolution helpers for tests that drive an image pipeline directly instead of through the image service.</summary>
internal static class TestImageRecipes
{
    /// <summary>Resolves <paramref name="request"/> against <paramref name="defaults"/> exactly as the image service
    /// does, returning the request and the settings a pipeline runs with.</summary>
    internal static (ImageRequest Request, ImageEffectiveSettings Settings) Resolve(ImageDefaults defaults, ImageRequest request)
    {
        ImageRequest resolved = defaults.Apply(request);
        return (resolved, ImageEffectiveSettings.Of(resolved));
    }

    /// <summary>Settings for tests that only exercise request mapping, not a generation.</summary>
    internal static ImageEffectiveSettings Settings(int width = 1024, int height = 1024, int steps = 20, float cfgScale = 7f) =>
        new ImageEffectiveSettings { Width = width, Height = height, Steps = steps, CfgScale = cfgScale };
}
