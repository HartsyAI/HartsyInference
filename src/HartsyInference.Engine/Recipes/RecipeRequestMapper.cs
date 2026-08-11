using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Recipes;

/// <summary>Shared mapping from the native <see cref="ImageRequest"/> onto the diffusion package's request records, so every architecture recipe clamps seeds and reads the resolved size identically.</summary>
internal static class RecipeRequestMapper
{
    /// <summary>Folds a native 64-bit seed into the diffusion pipelines' 31-bit seed space; a negative seed maps to <c>null</c>, meaning "pick one at random".</summary>
    public static int? MapSeed(long seed) => seed < 0 ? null : (int?)(int)(seed & 0x7FFFFFFF);

    /// <summary>Maps a zero/negative CLIP-skip to <c>null</c> so the text encoder's standard final layer is used.</summary>
    public static int? MapClipSkip(int? clipSkip) => clipSkip is null or <= 0 ? null : clipSkip;

    /// <summary>The concrete pixel size for a pipeline that has to snap or clamp it; <c>ImagesService</c> fills width/height from the family's <see cref="ImageDefaults"/> before <c>Generate</c>, so the coalesce here only covers a directly-driven pipeline.</summary>
    public static (int Width, int Height) Size(ImageRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        return (request.Width ?? ImageDefaults.Standard.Width!.Value, request.Height ?? ImageDefaults.Standard.Height!.Value);
    }

    /// <summary>Builds the text-to-image core (prompt, size, seed, steps and guidance) from an already defaults-resolved request; families that also carry a scheduler, CLIP-skip, init noise or img2img fields layer them on with a record <c>with</c> expression.</summary>
    public static TextToImageRequest ToTextToImage(ImageRequest request, string negative)
    {
        ArgumentNullException.ThrowIfNull(request);
        return new TextToImageRequest
        {
            Prompt = request.Prompt,
            NegativePrompt = negative,
            Width = request.Width,
            Height = request.Height,
            Steps = request.Steps,
            CfgScale = request.CfgScale,
            CfgRescale = request.CfgRescale,
            Tcfg = request.Tcfg,
            Seed = MapSeed(request.Seed),
        };
    }
}
