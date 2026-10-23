using System.Globalization;
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Fibo 2.0 editing: an instruction edit of one image, a masked edit (white = regenerate, the region greyed out before it
/// is encoded) and a multi-reference edit of up to four images referred to as <c>&lt;image_1&gt;</c>..<c>&lt;image_4&gt;</c>. The
/// edit-turbo build only edits. Both variants use the same path; they differ in their defaults (4 steps, no guidance).</summary>
public sealed partial class Fibo2RecipePipeline
{
    /// <summary>The reference policy: four images, refused beyond that. Only the count is read; Fibo 2.0 sizes its own copies.</summary>
    private static readonly ReferenceImagePolicy EditPolicy = new()
    {
        MaxReferences = Fibo2EditImages.MaxImages,
        LatentMultiple = 16,
        Overflow = ReferenceOverflow.Refuse,
    };

    /// <summary>Components the edit needs (vision tower, VAE encoder); the recipe sets it, and a pipeline built without it refuses edits.</summary>
    internal Fibo2EditSupport? EditSupport { get; init; }

    /// <summary>Whether <paramref name="request"/> goes down the edit path: it carries an init image, reference images or a mask, or
    /// the build is edit-only (so a text-to-image request reaches the refusal there rather than the sampler).</summary>
    private bool IsEdit(ImageRequest request) =>
        request.Img2Img is not null || request.Inpaint is not null || request.ReferenceImages is { Count: > 0 }
        || _variant.Is(Fibo2Variants.EditTurbo);

    private ImageResult GenerateEdit(ImageRequest request, ImageEffectiveSettings settings, IProgress<StepPreview>? progress,
        CancellationToken cancel)
    {
        if (request.Img2Img is { Mode: Img2ImgMode.Denoise })
            throw new NotSupportedException("Fibo 2.0 edits an image by conditioning on it; strength-based img2img denoising is not implemented. "
                + "Send the init image in Reference mode.");
        if (!IsDefaultSampler(request.Sampler) || !IsDefaultSchedule(request.Scheduler))
            throw new NotSupportedException("Fibo 2.0 samples only with its built-in flow-matching Euler; drop the sampler and schedule selection.");

        IReadOnlyList<ImageData> sources = ReferenceImageResolver.Collect(request.Img2Img?.InitImage, request.ReferenceImages, EditPolicy, "Fibo 2.0");
        if (sources.Count == 0)
        {
            throw new NotSupportedException($"{_variant.Variant.DisplayName} only edits images: send a source image (the init image in Reference "
                + "mode, or a reference image) and an edit instruction. It has no text-to-image mode.");
        }
        if (request.Inpaint is not null && sources.Count != 1)
            throw new NotSupportedException($"A Fibo 2.0 masked edit takes exactly one image; {sources.Count} were supplied.");
        Fibo2EditSupport support = EditSupport
            ?? throw new InvalidOperationException("This Fibo 2.0 pipeline was built without its editing components.");

        float guidance = settings.CfgScale;
        bool guided = CfgHelper.IsGuidanceActive(guidance);
        if (guided && !string.IsNullOrWhiteSpace(request.NegativePrompt))
            throw new NotSupportedException("Fibo 2.0 does not map negative prompts yet: its unconditional branch is the fixed null caption.");
        if (!guided && !string.IsNullOrWhiteSpace(request.NegativePrompt))
            Logs.Debug("[Fibo2] Negative prompt ignored: guidance is off, so the unconditional branch is never run.");

        // Result size: the one asked for, else the size trained for the first image's aspect ratio (ComfyUI and diffusers default).
        (int trainedW, int trainedH) = Fibo2EditImages.TrainedSize(sources[0].Width, sources[0].Height);
        int width = Math.Max(16, (request.Width ?? trainedW) / 16 * 16);
        int height = Math.Max(16, (request.Height ?? trainedH) / 16 * 16);

        ImageData[] images = [.. sources];
        if (request.Inpaint is not null)
        {
            byte[] mask = MaskResolver.ResolveBytes(request.Inpaint, images[0].Width, images[0].Height)!;
            images[0] = Fibo2EditImages.GreyOut(images[0], mask);
        }

        using Fibo2EditImages.Prepared prepared = Fibo2EditImages.Prepare(images, width, height, support.Processor, support.VisionFactor);
        string editJson = Fibo2EditPrompt.Build(request.Prompt, images.Length);
        int[] conditional = Fibo2EditPrompt.Tokenize(_tokenizer, editJson, prepared.PadCounts);
        int[] unconditional = Fibo2PromptTemplate.Tokenize(_tokenizer, Fibo2PromptTemplate.NullCaption);
        Fibo2EditInputs edit = new()
        {
            ConditionalTokens = conditional,
            VisionImages = prepared.Vision,
            ReferencePixels = prepared.Pixels,
            VisionEncoder = support.Encoder,
            VaeEncoder = support.VaeEncoder,
        };

        int steps = settings.Steps;
        float shift = Fibo2Recipe.SchedulerShift(request.SigmaShift ?? _defaults.SigmaShift);
        int? seed = RecipeRequestMapper.MapSeed(request.Seed);
        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel, totalSteps: steps);
        (byte[] rgb, int outWidth, int outHeight, int usedSeed) = _pipeline.GenerateEdit(edit, unconditional, width, height, steps, guidance,
            shift, seed, bridge);
        return new ImageResult
        {
            Rgb = rgb,
            Width = outWidth,
            Height = outHeight,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "fibo2",
                ["variant"] = _variant.Variant.Id,
                ["size"] = string.Create(CultureInfo.InvariantCulture, $"{outWidth}x{outHeight}"),
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                ["references"] = images.Length.ToString(CultureInfo.InvariantCulture),
                ["masked"] = (request.Inpaint is not null).ToString(CultureInfo.InvariantCulture),
            },
        };
    }
}
