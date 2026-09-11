using HartsyInference.Core.Logging;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Features;

/// <summary>Cuts the subject out of a finished generation: RMBG-1.4's matte becomes <see cref="ImageResult.Alpha"/> and
/// the generated RGB is left exactly as it was, so the caller composites the cutout against whatever it likes. Mirrors
/// where the ComfyUI backend puts its <c>SwarmRemBg</c> node — last, over the final pixels, after every refine pass.
/// <para>Deliberately NOT an <see cref="Recipes.ImageFeatures"/> bit: this is pixel post-processing that works for any
/// family, so no recipe has to declare it.</para></summary>
public static class BackgroundRemovalStage
{
    /// <summary>Returns <paramref name="result"/> carrying an alpha plane when the request asked for a cutout, else
    /// unchanged. The RGB bytes are never rewritten — a partial-coverage edge must be composited once by the consumer,
    /// and pre-blending it here would darken the same pixels twice.</summary>
    public static ImageResult Apply(InferenceEngine engine, ImageRequest request, ImageResult result)
    {
        if (request.RemoveBackground != true)
        {
            return result;
        }
        ModelSpec spec = ModelResolver.Resolve("rmbg", null, Modality.Vision);
        ImageData image = new ImageData { Rgb = result.Rgb, Width = result.Width, Height = result.Height };
        byte[] alpha = engine.VisionInternal.Matte(spec, image);
        Logs.Verbose($"[Features][RemoveBackground] Matted {result.Width}x{result.Height} with RMBG-1.4.");
        return result with { Alpha = alpha };
    }
}
