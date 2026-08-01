using HartsyInference.Core.Exceptions;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>MiniMax-H3 ("Hailuo 03") registration seam; weights are unreleased so <see cref="Construct"/> always throws. See <c>docs/Research/MINIMAX_H3.md</c>.</summary>
public sealed class MiniMaxH3Recipe : IVideoRecipe
{
    // ModelScope publishes as MiniMax/*, HuggingFace as MiniMaxAI/*, and the checkpoint may say "Hailuo 03".
    private static readonly string[] _familyIds = { "minimax-h3", "minimax-hailuo-03", "hailuo-03", "hailuo03" };

    /// <inheritdoc/>
    public string Name => "minimax-h3";

    /// <inheritdoc/>
    public bool Matches(string familyId)
    {
        foreach (string id in _familyIds)
        {
            if (string.Equals(familyId, id, StringComparison.OrdinalIgnoreCase))
                return true;
        }
        return false;
    }

    /// <summary>Placeholders except <c>Fps</c>; MiniMax publish only "2K" plus an aspect ratio, and the sampler is undisclosed.</summary>
    public VideoDefaults Defaults { get; } =
        new VideoDefaults { Steps = 30, CfgScale = 6.0f, Width = 1920, Height = 1080, Frames = 120, Fps = 24 };

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        throw new UnsupportedModelException(
            "MiniMax-H3 weights are not released yet (checked 2026-08-01: the ModelScope repo 'MiniMax/MiniMax-H3' "
            + "does not resolve, there is no HuggingFace mirror, and no technical report or config has been published). "
            + "ComfyUI's MiniMax H3 nodes are cloud API clients and contain no architecture to port. This recipe is a "
            + "registration seam only. See docs/Research/MINIMAX_H3.md for the verified capability contract and the "
            + "day-one bring-up checklist.");
    }
}
