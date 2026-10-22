using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Planning inputs for tests that plan against hand-written defaults rather than a registered recipe.</summary>
internal static class TestVideoRecipes
{
    /// <summary>A grid that accepts every count, for families no recipe registers.</summary>
    internal static FrameGrid AnyCount { get; } = FrameGrid.StepPlusOne(1, FrameNormalization.Reject);

    /// <summary>A declaration for <paramref name="familyId"/> with the registered family's real frame grid when it
    /// has one, so planning normalizes frames exactly as it does in production.</summary>
    internal static VideoRecipeDeclaration Declare(string familyId, VideoDefaults defaults, VideoFeatures features) =>
        new VideoRecipeDeclaration
        {
            FamilyId = familyId,
            Defaults = defaults,
            Features = features,
            Constraints = VideoRecipeRegistry.Resolve(familyId)?.Constraints ?? new VideoConstraints { Frames = AnyCount },
        };

    /// <summary>A plan identity for tests that build a <see cref="VideoPlan"/> by hand.</summary>
    internal static RecipeIdentity Identity(string familyId) => new RecipeIdentity { FamilyId = familyId, Revision = 1 };

    /// <summary>Planned settings for tests that drive a pipeline directly with an explicit geometry and schedule.</summary>
    internal static VideoEffectiveSettings Settings(int width = 832, int height = 480, int frames = 33, int fps = 16,
        int steps = 20, float cfgScale = 5f) => new VideoEffectiveSettings
    {
        Width = width, Height = height, Frames = frames, Fps = fps, Steps = steps, CfgScale = cfgScale, Seed = 1,
        ReferenceSizing = VideoReferenceSizing.Native, LockedFields = VideoLockedFields.None,
    };

    /// <summary>Empty provenance for hand-built plans whose tests do not inspect it.</summary>
    internal static IReadOnlyDictionary<string, ResolvedParameter> NoProvenance { get; } =
        new Dictionary<string, ResolvedParameter>(StringComparer.Ordinal);
}
