using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Image-generation service: detects the checkpoint's architecture, resolves its recipe from the
/// <see cref="RecipeRegistry"/>, constructs (and caches) the pipeline, and generates. Every composition object the
/// request sets is mapped to an <see cref="ImageFeatures"/> bit and checked against the resolved recipe's
/// <see cref="IArchitectureRecipe.Supports"/>, so an unwired feature is rejected by name rather than silently ignored.
/// Every tunable the caller left null is filled from the resolved recipe's <see cref="ImageDefaults"/> before the
/// pipeline is driven, so each family runs at its creator's recommended settings unless the caller overrides them.</summary>
public sealed class ImagesService : IImagesService
{
    private readonly InferenceEngine _engine;
    /// <summary>Tier 3.2 segment refinement's CLIPSeg backend — a service-owned cache (mirrors
    /// <c>VisionService</c>'s own <c>ClipSegSegmenter</c> instance) so a prompt with no <c>&lt;segment:&gt;</c>
    /// parts never loads it, and a repeat segment query on the same generation reuses the loaded weights.</summary>
    private readonly Vision.ClipSegSegmenter _clipSeg = new();

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal ImagesService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<ImageResult> GenerateAsync(ModelSpec spec, ImageRequest request, IProgress<StepPreview>? progress = null, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        RejectUnsupported(spec, request);
        return Task.Run(
            () =>
            {
                // A generic-refiner request keeps the refiner checkpoint cached across generations too — see
                // EvictOtherCheckpointPipelines' alsoKeepPath doc for the ping-pong this prevents.
                string? keepRefiner = request.Refiner?.Model;
                IRecipePipeline pipeline = _engine.GetOrConstructRecipe(spec, request, alsoKeepPath: keepRefiner);
                ImageRequest resolved = _engine.DefaultsFor(spec, pipeline).Apply(request);

                // Base-prompt tag-leak fix: <segment:>/<clear:> text must not reach the BASE (full-canvas) pass's
                // text encoder — a segment's own sub-prompt is meant for its own later masked denoise only. Gated
                // on hasSegments so a segment-free prompt takes the identity path (basePass IS resolved, not a
                // re-serialized copy) — zero behavior change for every request without a <segment:>/<clear:> tag.
                bool hasSegments = SegmentRefinement.HasSegmentParts(resolved.Prompt);
                ImageRequest basePass = hasSegments
                    ? resolved with { Prompt = SegmentRefinement.StripSegmentText(resolved.Prompt) }
                    : resolved;

                // basePass first: the crop is scaled to the resolution the model will actually run at, so it has to see
                // the defaults-filled request rather than the caller's.
                InpaintOnlyMasked.Plan? cropPlan = InpaintOnlyMasked.Prepare(basePass);
                ImageResult result = cropPlan is null
                    ? _engine.GenerateWithVramCleanup(() => pipeline.Generate(basePass, progress, cancel))
                    : InpaintOnlyMasked.Composite(
                        _engine.GenerateWithVramCleanup(() => pipeline.Generate(InpaintOnlyMasked.Apply(basePass, cropPlan), progress, cancel)),
                        cropPlan);

                // Generic refiner (any model over any base, PostApply semantics): pixels from the base pass become
                // the refiner family's img2img init. Runs BEFORE segment refinement, matching Comfy's stage order.
                // The classic SDXL-on-SDXL pair is skipped here — SdxlRecipePipeline keeps that internally, with
                // RefinerStage.IsSdxlInternalPair as the single routing decision both sides consult.
                result = RefinerStage.Apply(_engine, spec, basePass, result, progress, cancel);

                // Tier 3.2: <segment:X> runs AFTER pixels exist (it needs to segment the decoded image), so it
                // composes on top of whatever the ordinary inpaint-only-masked path above already produced —
                // a request can combine a top-level Inpaint.ShrinkGrow crop with segment refinement. Passes the
                // ORIGINAL resolved (tags intact), not basePass — SegmentRefinement.Apply re-parses resolved.Prompt
                // itself to find each segment's own sub-prompt/geometry.
                if (hasSegments)
                {
                    result = SegmentRefinement.Apply(result, resolved, pipeline, _engine.Backend, _clipSeg, progress, cancel);
                }
                return result;
            },
            cancel);
    }

    /// <summary>Which bit an init image asks for, given the caller's <see cref="Img2ImgMode"/> and what the family can
    /// actually do. Under <see cref="Img2ImgMode.Auto"/> a family that offers only reference editing gets
    /// <see cref="ImageFeatures.RefEdit"/>, one that offers only classic img2img gets <see cref="ImageFeatures.Img2Img"/>,
    /// and one offering both prefers classic — an <c>Init Image</c> plus a <c>Creativity</c> value conventionally means
    /// a strength-based denoise. An explicit mode is honoured as written so the refusal names the mode the caller asked
    /// for rather than silently doing the other thing.</summary>
    private static ImageFeatures Img2ImgBit(Img2Img img2img, ImageFeatures supported) => img2img.Mode switch
    {
        Img2ImgMode.Denoise => ImageFeatures.Img2Img,
        Img2ImgMode.Reference => ImageFeatures.RefEdit,
        _ => (supported & ImageFeatures.Img2Img) != 0 ? ImageFeatures.Img2Img
            : (supported & ImageFeatures.RefEdit) != 0 ? ImageFeatures.RefEdit
            : ImageFeatures.Img2Img,
    };

    /// <summary>The features <paramref name="request"/> asks for, one bit per composition object actually set.
    /// <paramref name="supported"/> only disambiguates the init-image mode; it never widens what is requested.</summary>
    private static ImageFeatures RequestedFeatures(ImageRequest request, ImageFeatures supported)
    {
        ImageFeatures features = ImageFeatures.None;
        if (request.Loras is { Entries.Count: > 0 })
        {
            features |= ImageFeatures.Lora;
        }
        if (request.ControlNets is { Count: > 0 })
        {
            features |= ImageFeatures.ControlNet;
        }
        if (request.IpAdapter is not null)
        {
            features |= ImageFeatures.IpAdapter;
        }
        if (request.Refiner is not null)
        {
            features |= ImageFeatures.Refiner;
        }
        if (request.Img2Img is not null)
        {
            features |= Img2ImgBit(request.Img2Img, supported);
        }
        if (request.Inpaint is not null)
        {
            features |= ImageFeatures.Inpaint;
        }
        if (request.Regional is not null)
        {
            features |= ImageFeatures.Regional;
        }
        if (request.VariationSeed is not null)
        {
            features |= ImageFeatures.VariationSeed;
        }
        if (!string.IsNullOrEmpty(request.SeamlessTiling) && request.SeamlessTiling != "false")
        {
            features |= ImageFeatures.SeamlessTiling;
        }
        return features;
    }

    /// <summary>Throws naming the family and the exact features it cannot apply; features the recipe declares pass through.</summary>
    private void RejectUnsupported(ModelSpec spec, ImageRequest request)
    {
        ImageFeatures supported = _engine.SupportedFeatures(spec);
        ImageFeatures requested = RequestedFeatures(request, supported);
        if (requested == ImageFeatures.None)
        {
            return;
        }
        ImageFeatures missing = requested & ~supported;
        if (missing != ImageFeatures.None)
        {
            throw new NotSupportedException(
                $"Model family '{InferenceEngine.FamilyIdFor(spec)}' does not support: {missing}.");
        }
    }
}
