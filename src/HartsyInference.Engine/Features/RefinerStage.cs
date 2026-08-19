using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Registry;

namespace HartsyInference.Engine.Features;

/// <summary>The generic second-stage refiner: ANY image model refining ANY base result, the way the ComfyUI backend's
/// PostApply method works — run the base in full, then run the refiner as img2img over the base's pixels at
/// <c>Control</c> strength, optionally upscaled first (hires-fix). Because the hand-off is PIXELS, the
/// cross-architecture problem Comfy solves with an explicit decode→re-encode (<c>modelMustReencode</c>) is
/// structural here: the refiner pipeline's own img2img path VAE-encodes the pixels with its own encoder, whatever
/// family it is.
/// <para>The one case NOT handled here is the classic SDXL-refiner-on-SDXL pair, which
/// <see cref="Recipes.Image.SdxlRecipePipeline"/> keeps internally — its dedicated
/// <see cref="SdxlRefinerPipeline"/> carries the aesthetic-score conditioning and true mid-loop StepSwap that only
/// exist for that pair. <see cref="IsSdxlInternalPair"/> is the single routing decision both sides consult, so the
/// pass can never run twice or zero times.</para></summary>
public static class RefinerStage
{
    /// <summary>Whether this request's refiner is the classic SDXL pair the SDXL recipe handles internally.</summary>
    public static bool IsSdxlInternalPair(ImageRequest request, string baseFamilyId)
    {
        if (request.Refiner is null || !string.Equals(baseFamilyId, "sdxl", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }
        string family = ResolveRefinerFamily(request.Refiner);
        return family is "sdxl" or "sdxl-refiner";
    }

    /// <summary>The refiner model's family id: the host-provided <see cref="Refiner.FamilyId"/> when present, else
    /// the engine's own header sniff (classic single-file architectures only).</summary>
    public static string ResolveRefinerFamily(Refiner refiner)
    {
        if (!string.IsNullOrWhiteSpace(refiner.FamilyId))
        {
            return refiner.FamilyId;
        }
        try
        {
            ModelArchitecture arch = PipelineFactory.DetectArchitecture(refiner.Model);
            return arch switch
            {
                ModelArchitecture.Sdxl => "sdxl",
                ModelArchitecture.SdxlRefiner => "sdxl-refiner",
                ModelArchitecture.StableDiffusion15 => "sd15",
                ModelArchitecture.StableDiffusion3 => "sd3",
                ModelArchitecture.Flux1 => "flux1",
                ModelArchitecture.Flux2 => "flux2",
                ModelArchitecture.AuraFlow => "auraflow",
                ModelArchitecture.Chroma => "chroma",
                _ => arch.ToString().ToLowerInvariant(),
            };
        }
        catch (Exception ex)
        {
            Logs.Warning($"[Features][Refiner] Could not sniff the refiner model's architecture: {ex.Message}");
            return "unknown";
        }
    }

    /// <summary>Runs the generic refiner pass over <paramref name="baseResult"/> when the request asks for one, else
    /// returns it unchanged. <paramref name="request"/> must be the defaults-resolved, segment-stripped base request
    /// (its prompt is what conditions the refine).</summary>
    internal static ImageResult Apply(
        InferenceEngine engine, ModelSpec baseSpec, ImageRequest request, ImageResult baseResult,
        IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        RefinerResolver.RefinerSpec? spec = RefinerResolver.Resolve(
            request.Refiner,
            request.Steps ?? 20,
            request.CfgScale ?? 7f);
        if (spec is null)
        {
            return baseResult;
        }
        string baseFamily = baseSpec.Catalog?.Id ?? baseSpec.Requested ?? "";
        if (IsSdxlInternalPair(request, baseFamily))
        {
            return baseResult; // SdxlRecipePipeline already consumed it (aesthetic conditioning + true StepSwap).
        }
        string refinerFamily = ResolveRefinerFamily(request.Refiner!);
        if (refinerFamily is "unknown")
        {
            throw new NotSupportedException(
                $"The refiner model '{Path.GetFileName(spec.Model)}' has no resolvable architecture — the host did "
                + "not supply a family id and the engine's header sniff did not recognize it.");
        }

        ModelSpec refinerSpec = new ModelSpec
        {
            Requested = refinerFamily,
            Modality = Modality.Image,
            LocalPath = spec.Model,
            Catalog = new CatalogEntry
            {
                Id = refinerFamily,
                Modality = Modality.Image,
                DisplayName = Path.GetFileNameWithoutExtension(spec.Model),
                Architecture = refinerFamily,
                Status = ModelStatus.Verified,
            },
        };

        // The refiner model must be able to consume an init image at a strength — that IS the PostApply hand-off.
        ImageFeatures refinerFeatures = engine.SupportedFeatures(refinerSpec);
        if ((refinerFeatures & (ImageFeatures.Img2Img | ImageFeatures.RefEdit)) == 0)
        {
            throw new NotSupportedException(
                $"'{Path.GetFileName(spec.Model)}' ({refinerFamily}) can't be used as a refiner: its recipe has no "
                + "img2img path to consume the base image with.");
        }

        if (!string.Equals(spec.Method, "PostApply", StringComparison.OrdinalIgnoreCase))
        {
            // True StepSwap needs a shared latent space mid-loop, which only the SDXL internal pair has. The pixel
            // hand-off is the honest general equivalent (it is also exactly what Comfy's re-encode fallback does
            // when the latent spaces differ).
            Logs.Info($"[Features][Refiner] Method '{spec.Method}' runs as PostApply on the generic path — "
                + "mid-loop model swap requires the SDXL base + SDXL-refiner pair.");
        }

        // Optional hires-fix upscale, snapped to the refiner family's dimension granularity.
        byte[] rgb = baseResult.Rgb;
        int width = baseResult.Width;
        int height = baseResult.Height;
        if (spec.Upscale is > 0f and not 1f)
        {
            int granularity = DimensionGranularity(refinerFamily);
            int newW = Math.Max(granularity, (int)Math.Round(width * spec.Upscale / granularity) * granularity);
            int newH = Math.Max(granularity, (int)Math.Round(height * spec.Upscale / granularity) * granularity);
            if (newW != width || newH != height)
            {
                rgb = FeatureImaging.ResizeRgb24(new ImageData { Rgb = rgb, Width = width, Height = height }, newW, newH);
                width = newW;
                height = newH;
            }
        }

        ImageRequest refineRequest = request with
        {
            Width = width,
            Height = height,
            Steps = spec.Steps,
            CfgScale = spec.CfgScale,
            Refiner = null,          // no recursion
            Inpaint = null,          // the base pass already honoured masks; refining re-runs the whole canvas
            VariationSeed = null,    // variation shaped the base noise; re-blending would double-apply it
            Img2Img = new Img2Img
            {
                InitImage = new ImageData { Rgb = rgb, Width = width, Height = height },
                Creativity = spec.Strength,
                Mode = Img2ImgMode.Denoise,
            },
            Components = string.IsNullOrWhiteSpace(spec.Vae)
                ? request.Components
                : (request.Components ?? new ComponentOverrides()) with { Vae = spec.Vae },
        };

        Logs.Info($"[Features][Refiner] Generic refine: base '{baseFamily}' {baseResult.Width}x{baseResult.Height} → "
            + $"'{refinerFamily}' at {width}x{height}, strength={spec.Strength:F2}, steps={spec.Steps}, cfg={spec.CfgScale}.");
        IRecipePipeline refinerPipeline = engine.GetOrConstructRecipe(refinerSpec, refineRequest, alsoKeepPath: baseSpec.LocalPath);
        ImageRequest resolvedRefine = engine.DefaultsFor(refinerSpec, refinerPipeline).Apply(refineRequest) with
        {
            // Apply() may re-fill from family defaults; the refine geometry and budget are already decided above.
            Width = width,
            Height = height,
            Steps = spec.Steps,
            CfgScale = spec.CfgScale,
        };
        ImageResult refined = engine.GenerateWithVramCleanup(() => refinerPipeline.Generate(resolvedRefine, progress, cancel));
        Dictionary<string, string> meta = new Dictionary<string, string>(refined.Meta, StringComparer.OrdinalIgnoreCase)
        {
            ["refiner_model"] = Path.GetFileName(spec.Model),
            ["refiner_family"] = refinerFamily,
            ["refiner_strength"] = spec.Strength.ToString("F2"),
        };
        return refined with { Meta = meta };
    }

    /// <summary>The width/height multiple a family's pipeline requires. 16 satisfies every /8 and /16 family;
    /// HunyuanImage's 32× VAE is the one stricter case.</summary>
    private static int DimensionGranularity(string familyId) =>
        string.Equals(familyId, "hunyuan-image", StringComparison.OrdinalIgnoreCase) ? 32 : 16;
}
