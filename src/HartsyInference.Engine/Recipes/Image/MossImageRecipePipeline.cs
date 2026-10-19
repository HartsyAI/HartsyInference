using System.Globalization;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Memory;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed MOSS-Image pipeline driven against the native <see cref="ImageRequest"/>: snaps the size to the
/// 16-pixel latent grid, turns regional prompt parts into the model's layout JSON, renders both prompts through Qwen3.5's
/// thinking chat template, reads the skip-layer guidance recipe from the request extras (<c>slg.*</c>, the reference's
/// layer 22 on the unconditional branch by default) and calls <see cref="MossImagePipeline.Generate"/>. Reference images
/// (an edit: <see cref="ImageRequest.ReferenceImages"/>, and the init image in <see cref="Img2ImgMode.Reference"/>) follow
/// the Ostris edit node: each is shown to the encoder's vision tower behind a <c>Picture N:</c> block in both prompts and
/// to the DiT as reference tokens (<c>reference.*</c> extras). Owns every disposable the recipe created.</summary>
public sealed class MossImageRecipePipeline(MossImagePipeline pipeline, MossImagePromptTokenizer tokenizer, MossImageConfig config,
    ResolvedModelVariant variant, IReadOnlyList<IDisposable> owned) : IRecipePipeline
{
    /// <summary>Training defaults of ai-toolkit's MOSS-Image edit models: the VAE copy fit to 1 MP, the vision copy to 384².</summary>
    public const int DefaultReferencePixels = 1024 * 1024, DefaultVisionPixels = 384 * 384;

    private int _disposed;

    /// <inheritdoc/>
    public ImageDefaults? VariantDefaults => MossImageRecipe.FamilyDefaults;

    /// <summary>The encoder that reads reference images, or null when it has no vision tower (text-to-image only).</summary>
    public IMultimodalFinalHiddenStateEncoder? ReferenceEncoder { get; init; }

    /// <summary>The <c>Picture N:</c> placeholder renderer for the encoder's tokenizer; null with no vision tower.</summary>
    public VisionPlaceholderPrompt? Placeholders { get; init; }

    /// <summary>Whether a LoRA is merged; references without one are logged, since the base model was not trained on them.</summary>
    public bool HasLora { get; init; }

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

        bool initIsReference = request.Img2Img is { Mode: Img2ImgMode.Reference };
        if (initIsReference && request.Inpaint is not null)
        {
            throw new InvalidOperationException("MOSS-Image cannot inpaint an init image sent in Reference mode: it conditions on "
                + "the image instead of denoising it, so the mask would be ignored. Use Auto or Denoise mode to inpaint.");
        }
        ReferenceImagePolicy policy = ReferencePolicy(request.Extra);
        IReadOnlyList<ImageData> ordered = ReferenceImageResolver.Collect(initIsReference ? request.Img2Img!.InitImage : null,
            request.ReferenceImages, policy, "MOSS-Image");
        if (ordered.Count > 0 && (ReferenceEncoder is not { HasVision: true } || Placeholders is null))
        {
            throw new NotSupportedException("MOSS-Image reference images need a Qwen3.5 encoder with its vision tower: Comfy-Org's "
                + "qwen3.5_9b_* files carry it, OpenMOSS's text_encoder/ and the GGUFs do not. Pass one as the encoder, or give "
                + "the tower separately (the TextEncoderVision component, e.g. Qwen/Qwen3.5-9B's model-00004-of-00004.safetensors).");
        }
        using ResolvedReferences? references = ordered.Count == 0 ? null
            : ReferenceImageResolver.Resolve(ordered, policy, ReferenceEncoder!.VisionInputSize);

        string prompt = LayoutPromptBuilder.BuildSceneLayout(PromptTagFlattening.Flatten(request.Prompt), width, height);
        string negativePrompt = PromptTagFlattening.Flatten(request.NegativePrompt ?? "");
        WeightedTokenSequence positive, negative;
        ReferenceImageInputs? inputs = null;
        if (references is null)
        {
            positive = tokenizer.Tokenize(prompt);
            negative = tokenizer.Tokenize(negativePrompt);
        }
        else
        {
            int[] padCounts = new int[references.Count];
            int pads = 0;
            for (int i = 0; i < padCounts.Length; i++)
            {
                Tensor vision = references.Vision[i];
                padCounts[i] = ReferenceEncoder!.CountImageTokens((int)vision.Shape[3], (int)vision.Shape[2]);
                pads += padCounts[i];
            }
            string blocks = Placeholders!.Render(padCounts);
            positive = tokenizer.Tokenize(prompt, blocks, pads, Placeholders.ImagePadId);
            negative = tokenizer.Tokenize(negativePrompt, blocks, pads, Placeholders.ImagePadId);
            bool kvCache = RequestExtras.Bool(request.Extra, RequestExtras.ReferenceKvCache, true);
            inputs = new ReferenceImageInputs
            {
                Latent = references.Latent,
                Vision = references.Vision,
                Options = new ReferenceConditioningOptions { Mode = kvCache ? ReferenceAttentionMode.CachedKv : ReferenceAttentionMode.Joint },
            };
            if (!HasLora)
                Logs.Warning("[MossImageRecipePipeline] Reference images without a LoRA: the base model was not trained to follow them.");
        }
        MossImageGuidance guidance = GuidanceFrom(request.Extra);

        using Img2ImgResolver.Img2ImgSpec? img2img = initIsReference ? null : RecipeImg2ImgBinder.Resolve(request, width, height);
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
        (byte[] rgb, int outW, int outH, int seed) = pipeline.Generate(positive, negative, inner, guidance, bridge, inputs);
        Dictionary<string, string> meta = new(StringComparer.OrdinalIgnoreCase)
        {
            ["arch"] = "moss-image",
            ["variant"] = variant.Variant.Id,
            ["size"] = string.Create(CultureInfo.InvariantCulture, $"{outW}x{outH}"),
            ["seed"] = seed.ToString(CultureInfo.InvariantCulture),
            ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
            ["cfg"] = cfg.ToString(CultureInfo.InvariantCulture),
            ["uncond_skip_layers"] = string.Join(",", guidance.UncondSkipLayers),
        };
        if (inputs is not null)
        {
            meta["references"] = inputs.Count.ToString(CultureInfo.InvariantCulture);
            meta["reference_attention"] = inputs.Options.Mode == ReferenceAttentionMode.CachedKv ? "kv_cache" : "joint";
        }
        return new ImageResult { Rgb = rgb, Width = outW, Height = outH, Seed = seed, Meta = meta };
    }

    /// <summary>The reference policy the request's <c>reference.*</c> extras describe: up to three images, the VAE copy fit
    /// down to <c>reference.max_pixels</c> (1 MP; 0 keeps its size) and snapped to the 16-pixel token grid, the vision copy
    /// fit down to <c>reference.vision_pixels</c> (384²), more images than three refused.</summary>
    public static ReferenceImagePolicy ReferencePolicy(IReadOnlyDictionary<string, object>? extra) => new()
    {
        MaxReferences = MossImageConfig.MaxReferenceImages,
        LatentTargetArea = Math.Max(0, (int)RequestExtras.Number(extra, RequestExtras.ReferenceMaxPixels, DefaultReferencePixels)),
        LatentMultiple = 16,
        VisionTargetArea = Math.Max(1, (int)RequestExtras.Number(extra, RequestExtras.ReferenceVisionPixels, DefaultVisionPixels)),
        Overflow = ReferenceOverflow.Refuse,
    };

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
