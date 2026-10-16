using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Recipes;

/// <summary>Shared mapping from the native <see cref="ImageRequest"/> onto the diffusion package's request records, so every architecture recipe clamps seeds and reads the resolved size identically.</summary>
internal static class RecipeRequestMapper
{
    /// <summary>Folds a native 64-bit seed into the diffusion pipelines' 31-bit seed space; a negative seed maps to <c>null</c>, meaning "pick one at random".</summary>
    public static int? MapSeed(long seed) => seed < 0 ? null : (int?)(int)(seed & 0x7FFFFFFF);

    /// <summary>Maps a zero/negative CLIP-skip to <c>null</c> so the text encoder's standard final layer is used.</summary>
    public static int? MapClipSkip(int? clipSkip) => clipSkip is null or <= 0 ? null : clipSkip;

    /// <summary>Narrows a static sigma shift (SwarmUI's <c>ModelSamplingSD3</c>/<c>ModelSamplingAuraFlow</c> <c>shift</c>) to
    /// the diffusion request's float; null keeps the family's own shift. The narrowed value must be finite and above
    /// zero, so overflow, underflow to zero and a non-positive shift are refused here by the request field's name
    /// rather than later by a scheduler argument.</summary>
    public static float? MapSigmaShift(double? sigmaShift)
    {
        if (sigmaShift is not double shift)
        {
            return null;
        }
        float narrowed = (float)shift;
        if (!float.IsFinite(narrowed) || narrowed <= 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(ImageRequest.SigmaShift), shift,
                "SigmaShift must be a finite number above zero within float range.");
        }
        return narrowed;
    }

    /// <summary>Narrows a Flux-style sigma shift for Flux.2, FLUX.3 and Krea 2, where SwarmUI feeds the value to
    /// <c>ModelSamplingFlux</c> as <c>max_shift</c> (Krea 2 also as <c>base_shift</c>), so it is mu and the scheduler
    /// shift is exp(mu). Zero is a real setting there (shift 1, SwarmUI's minimum), so it is accepted; a negative mu, a
    /// non-finite value and a mu whose exp overflows float are refused by the request field's name.</summary>
    public static float? MapFluxMaxShift(double? sigmaShift)
    {
        if (sigmaShift is not double shift)
        {
            return null;
        }
        float mu = (float)shift;
        if (!float.IsFinite(mu) || mu < 0f || !float.IsFinite(MathF.Exp(mu)))
        {
            throw new ArgumentOutOfRangeException(nameof(ImageRequest.SigmaShift), shift,
                "SigmaShift on Flux.2 and Krea 2 is mu (ModelSamplingFlux max_shift): it must be finite, zero or above, and small enough that exp(mu) fits a float.");
        }
        return mu;
    }

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
            SeamlessTiling = request.SeamlessTiling,
            VariationSeed = request.VariationSeed?.Seed ?? -1,
            VariationSeedStrength = request.VariationSeed?.Strength ?? 0,
            Seed = MapSeed(request.Seed),
            SigmaShift = MapSigmaShift(request.SigmaShift),
            // Routed through the resolver rather than read raw, so an unavailable sampler is refused by name here —
            // before the checkpoint loads — instead of deep inside the pipeline, or silently dropped.
            Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
        };
    }
}
