using System.Globalization;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Memory;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed MOSS-Image pipeline driven against the native <see cref="ImageRequest"/>: snaps the size to the
/// 16-pixel latent grid, turns regional prompt parts into the model's layout JSON, renders both prompts through Qwen3.5's
/// thinking chat template, reads the skip-layer guidance recipe from the request extras (<c>slg.*</c>, the reference's
/// layer 22 on the unconditional branch by default) and calls <see cref="MossImagePipeline.Generate"/>. Owns every
/// disposable the recipe created.</summary>
public sealed class MossImageRecipePipeline(MossImagePipeline pipeline, MossImagePromptTokenizer tokenizer, MossImageConfig config,
    ResolvedModelVariant variant, IReadOnlyList<IDisposable> owned) : IRecipePipeline
{
    private int _disposed;

    /// <inheritdoc/>
    public ImageDefaults? VariantDefaults => MossImageRecipe.FamilyDefaults;

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancel.ThrowIfCancellationRequested();
        ImageDefaults defaults = MossImageRecipe.FamilyDefaults;
        int steps = request.Steps ?? defaults.Steps;
        float cfg = request.CfgScale ?? defaults.CfgScale;
        int unit = config.VaeDownscaleFactor;
        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        int width = Math.Max(unit, reqWidth / unit * unit);
        int height = Math.Max(unit, reqHeight / unit * unit);
        if (width != reqWidth || height != reqHeight)
            Logs.Info($"[MossImageRecipePipeline] Snapped {reqWidth}x{reqHeight} → {width}x{height} (multiple of {unit}).");
        if (request.Img2Img is { Mode: Img2ImgMode.Reference })
        {
            throw new NotSupportedException("MOSS-Image has no reference-image conditioning: send the init image in Auto or "
                + "Denoise mode to run image-to-image on it.");
        }

        string prompt = LayoutPromptBuilder.BuildSceneLayout(PromptTagFlattening.Flatten(request.Prompt), width, height);
        WeightedTokenSequence positive = tokenizer.Tokenize(prompt);
        WeightedTokenSequence negative = tokenizer.Tokenize(PromptTagFlattening.Flatten(request.NegativePrompt ?? ""));
        MossImageGuidance guidance = GuidanceFrom(request.Extra);

        using Img2ImgResolver.Img2ImgSpec? img2img = RecipeImg2ImgBinder.Resolve(request, width, height);
        TextToImageRequest inner = RecipeImg2ImgBinder.Apply(
            new TextToImageRequest
            {
                SigmaShift = RecipeRequestMapper.MapSigmaShift(request.SigmaShift),
                SeamlessTiling = request.SeamlessTiling,
                VariationSeed = request.VariationSeed?.Seed ?? -1,
                VariationSeedStrength = request.VariationSeed?.Strength ?? 0,
                Prompt = request.Prompt,
                NegativePrompt = request.NegativePrompt ?? "",
                Width = width,
                Height = height,
                Steps = steps,
                CfgScale = cfg,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
                Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
            },
            img2img);

        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel, totalSteps: steps);
        (byte[] rgb, int outW, int outH, int seed) = pipeline.Generate(positive, negative, inner, guidance, bridge);
        return new ImageResult
        {
            Rgb = rgb,
            Width = outW,
            Height = outH,
            Seed = seed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "moss-image",
                ["variant"] = variant.Variant.Id,
                ["size"] = string.Create(CultureInfo.InvariantCulture, $"{outW}x{outH}"),
                ["seed"] = seed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                ["cfg"] = cfg.ToString(CultureInfo.InvariantCulture),
                ["uncond_skip_layers"] = string.Join(",", guidance.UncondSkipLayers),
            },
        };
    }

    /// <summary>The guidance recipe the request's <c>slg.*</c> extras describe, each absent key keeping the reference's
    /// value (<see cref="MossImageGuidance.Reference"/>: layer 22 skipped on the unconditional branch at every step).</summary>
    public static MossImageGuidance GuidanceFrom(IReadOnlyDictionary<string, object>? extra)
    {
        MossImageGuidance reference = MossImageGuidance.Reference;
        return new MossImageGuidance
        {
            UncondSkipLayers = RequestExtras.IntList(extra, RequestExtras.SlgUncondSkipLayers) ?? reference.UncondSkipLayers,
            SlgLayers = RequestExtras.IntList(extra, RequestExtras.SlgLayers) ?? reference.SlgLayers,
            SlgScale = (float)RequestExtras.Number(extra, RequestExtras.SlgScale, reference.SlgScale),
            Start = (float)RequestExtras.Number(extra, RequestExtras.SlgStart, reference.Start),
            Stop = (float)RequestExtras.Number(extra, RequestExtras.SlgStop, reference.Stop),
        };
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        new CompositeDisposable([.. Enumerable.Reverse(owned)]).Dispose();
    }
}
