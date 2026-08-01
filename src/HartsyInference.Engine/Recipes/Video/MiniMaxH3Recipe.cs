using HartsyInference.Core.Exceptions;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>MiniMax-H3 ("Hailuo 03") recipe seam — registered so that a supplied H3 checkpoint reports why it cannot
/// run instead of surfacing as an unknown architecture. Verified 2026-08-01: <see cref="Construct"/> is reached only
/// when a checkpoint path is given (<c>hartsy video -m minimax-h3 --model-path ...</c>); with no path the catalog's
/// generic "no checkpoint found" message wins first, exactly as it would for an unregistered id. H3 is an
/// omni-modal video model (text/image/video/audio in,
/// 2K video plus a jointly generated stereo soundtrack out) announced 2026-07-31 with weights promised but not
/// published; there is no checkpoint, no config, and no technical report to implement against, and ComfyUI's H3
/// nodes are cloud API clients that carry no architecture. <see cref="Construct"/> therefore always throws.
/// The bring-up checklist and the verified capability contract live in <c>docs/Research/MINIMAX_H3.md</c>.</summary>
public sealed class MiniMaxH3Recipe : IVideoRecipe
{
    /// <summary>Weight-release aliases to watch: MiniMax publish to ModelScope as <c>MiniMax/*</c> but to
    /// HuggingFace as <c>MiniMaxAI/*</c>, and the checkpoint's architecture string may say "Hailuo 03" instead
    /// of "H3". <see cref="Matches"/> accepts all of them so the sniffer isn't keyed to one guess.</summary>
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

    /// <summary>Only <c>Fps = 24</c> is a documented H3 figure. Everything else is a PLACEHOLDER and must be replaced
    /// when weights ship: MiniMax's API exposes the opaque string "2K" plus an aspect ratio and never states pixel
    /// dimensions ("2K" is ambiguous — DCI 2048x1080 vs QHD 2560x1440), the sampler and guidance convention are
    /// undisclosed so steps/CFG are the generic fallbacks, and <c>Frames</c> is 5s x 24fps arithmetic off the API's
    /// minimum duration, not a vendor number. Note the unit mismatch the implementer inherits: H3's API takes
    /// duration in SECONDS (integer, 5-15) while <c>VideoRequest</c> carries a frame count.</summary>
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
            + "day-one bring-up checklist. Note that H3 generates a stereo soundtrack with every clip, so widening the "
            + "frame-only IVideoRecipePipeline contract (the existing TODO(E-IMG-4/5) that drops LTX-2.3's audio) is a "
            + "prerequisite for this model, not a follow-up.");
    }
}
