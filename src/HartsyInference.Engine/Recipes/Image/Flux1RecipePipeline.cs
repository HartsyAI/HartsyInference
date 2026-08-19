using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Flux.1 pipeline driven against the native <see cref="ImageRequest"/>. <see cref="FluxPipeline"/> owns the CLIP-L + T5-XXL encoders, so this tokenizes the prompt with both (CLIP-L for the pooled EOS vector, T5-XXL per-token plus its attention mask) and calls <see cref="FluxPipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>FluxLoader.Generate</c> vanilla text-to-image drive path.</summary>
public sealed class Flux1RecipePipeline : IRecipePipeline
{
    private readonly FluxPipeline _pipeline;
    private readonly ClipTokenizer _clipTokenizer;
    private readonly T5Tokenizer _t5Tokenizer;
    private readonly bool _isDev;
    private readonly List<SafeTensorsLoader> _loaders;
    private readonly MergedLoraStack? _loraStack;
    private readonly IBackend _backend;

    /// <summary>Wraps the constructed Flux.1 pipeline plus its tokenizers and merged LoRA stack, taking ownership of every disposable. <paramref name="isDev"/> selects the step fallback and whether the embedded distilled guidance is applied.</summary>
    public Flux1RecipePipeline(FluxPipeline pipeline, ClipTokenizer clipTokenizer, T5Tokenizer t5Tokenizer, bool isDev, List<SafeTensorsLoader> loaders, MergedLoraStack? loraStack, IBackend backend)
    {
        _pipeline = pipeline;
        _clipTokenizer = clipTokenizer;
        _t5Tokenizer = t5Tokenizer;
        _isDev = isDev;
        _loaders = loaders;
        _loraStack = loraStack;
        _backend = backend;
    }

    /// <summary>A Schnell checkpoint (no guidance embedding) is a 4-step distilled model, so it resolves against <see cref="Flux1Recipe.SchnellDefaults"/> rather than Dev's 28 steps.</summary>
    public ImageDefaults? VariantDefaults => _isDev ? Flux1Recipe.FamilyDefaults : Flux1Recipe.SchnellDefaults;

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        int steps = request.Steps ?? (_isDev ? Flux1Recipe.FamilyDefaults.Steps : Flux1Recipe.SchnellDefaults.Steps);
        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);

        // TODO(E-IMG-4/5): Flux guidance came from the FluxGuidanceScale T2IParam (default 3.5 for Dev, ignored by
        // Schnell) — defaulted here. True-CFG (trueCfgScale + negative prompt) and Tools/Kontext are deferred, so
        // NegativePrompt / CfgScale are not mapped for the base path.
        float guidance = _isDev ? 3.5f : 0f;

        // Prompt images on Flux mean Redux (redux.stylemodel); a real IP-Adapter checkpoint is SD15/SDXL-only, and
        // prompt images with NEITHER selected must refuse rather than silently drop.
        string? styleModel = RequestExtras.String(request.Extra, RequestExtras.ReduxStyleModel);
        if (RequestExtras.String(request.Extra, RequestExtras.IpAdapterModel) is not null)
        {
            throw new NotSupportedException(
                "IP-Adapter checkpoints are not supported on Flux — the image-prompt slot on Flux drives FLUX.1 Redux. "
                + "Select a Redux style model instead, or use an SD15/SDXL model for IP-Adapter.");
        }
        if (styleModel is null && request.IpAdapter?.PromptImages is { Count: > 0 })
        {
            throw new NotSupportedException(
                "Prompt images on Flux require a FLUX.1 Redux style model (redux.stylemodel) — none was selected, "
                + "and silently ignoring the images would misrepresent the output.");
        }

        int[] clipTokens = _clipTokenizer.Encode(prompt);
        int eosPos = ClipTokenizer.FindEosPosition(clipTokens);
        int[] t5Tokens = _t5Tokenizer.Encode(prompt);
        int[] t5Mask = T5Tokenizer.CreateAttentionMask(t5Tokens);

        // FLUX.1 Canny/Depth: the host already ran the edge/depth annotator (it needs host-app image types this
        // package can't reference) and handed back the finished map under this key. Absent on a genuine Tools
        // checkpoint surfaces as GenerateFromTokens' own "requires a control image" error, not a silent fallback.
        ImageData? toolsControlImageData = RequestExtras.Image(request.Extra, RequestExtras.FluxToolsControlImage);
        Tensor? controlImage = toolsControlImageData is null
            ? null
            : FeatureImaging.RgbToTensorMinusOneOne(FeatureImaging.ResizeRgb24(toolsControlImageData, reqWidth, reqHeight), reqWidth, reqHeight);

        FluxControlNetResolver.ResolvedSpec? controlNets = null;
        RegionalPlan? regionalPlan = null;
        try
        {
            controlNets = FluxControlNetResolver.Resolve(
                request.ControlNets, reqWidth, reqHeight,
                static message => Logs.Info($"[Features][ControlNet] {message}"));

            // Redux runs before the DiT preload; the resolver frees its encoder weights afterward. Sharding excludes
            // redux only when applyStart > 0 (logged unsharded fallback pipeline-side); applyStart == 0 composes.
            using ReduxResolver.ReduxSpec? redux = ReduxResolver.ResolveAsync(
                request.IpAdapter,
                styleModel,
                request.Components?.ClipVision,
                RequestExtras.Number(request.Extra, RequestExtras.ReduxMultiply, 1.0),
                RequestExtras.Number(request.Extra, RequestExtras.ReduxMerge, 1.0),
                RequestExtras.Number(request.Extra, RequestExtras.ReduxApplyStart, 0.0),
                _backend,
                static message => Logs.Info($"[Features][Redux] {message}"),
                cancel).GetAwaiter().GetResult();

            using Img2ImgResolver.Img2ImgSpec? img2img = RecipeImg2ImgBinder.Resolve(request, reqWidth, reqHeight);

            // Variation noise only applies to the text-to-image path — an init latent replaces the seeded noise
            // outright, and building it anyway would hand the pipeline a tensor it never disposes.
            Tensor? variationNoise = img2img is null
                ? VariationSeedResolver.Resolve(
                    request.VariationSeed, reqWidth, reqHeight, request.Seed, VariationSeedResolver.FluxLatentChannels)
                : null;
            if (img2img is not null && request.VariationSeed is not null)
            {
                Logs.Warning("[Flux1RecipePipeline] Variation seed is ignored on the img2img path — the init latent replaces the seeded noise.");
            }

            TextToImageRequest inner = RecipeImg2ImgBinder.Apply(
                new TextToImageRequest
                {
                    SeamlessTiling = request.SeamlessTiling,
                    Prompt = prompt,
                    Width = request.Width,
                    Height = request.Height,
                    Steps = steps,
                    Seed = RecipeRequestMapper.MapSeed(request.Seed),
                    InitialNoise = variationNoise,
                },
                img2img);

            Action<GenerationProgress> bridge = p =>
            {
                cancel.ThrowIfCancellationRequested();
                progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
            };

            regionalPlan = BuildRegionalPlan(prompt, reqWidth, reqHeight, steps);

            (byte[] rgb, int width, int height, int usedSeed) = _pipeline.GenerateFromTokens(
                clipTokens, eosPos, t5Tokens, t5Mask, inner,
                guidanceScale: guidance,
                onProgress: bridge,
                controlImage: controlImage,
                regionalPlan: regionalPlan,
                fluxControlNets: controlNets?.Conditionings,
                reduxImageEmbeds: redux?.Embeds,
                reduxApplyStartFraction: redux?.ApplyStart ?? 0f);

            return new ImageResult
            {
                Rgb = rgb,
                Width = width,
                Height = height,
                Seed = usedSeed,
                Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["arch"] = "flux1",
                    ["size"] = $"{width}x{height}",
                    ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                    ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                },
            };
        }
        finally
        {
            RegionalPromptResolver.DisposeRegions(regionalPlan);
            controlNets?.Dispose();
            controlImage?.Dispose();
        }
    }

    /// <summary>Builds a regional-conditioning plan when the prompt carries <c>&lt;region:&gt;</c>/<c>&lt;object:&gt;</c>
    /// parts, null otherwise. <see cref="RegionalPlan.BaseCond"/> is a required field on the resolver's signature but
    /// is never read by <see cref="FluxPipeline.GenerateFromTokens"/>'s regional path (confirmed by inspection: the
    /// pipeline builds its actual background stream from its own internally-computed T5 embeddings, not this field)
    /// — a throwaway placeholder, disposed immediately rather than kept alive for the plan's lifetime.</summary>
    private RegionalPlan? BuildRegionalPlan(string prompt, int width, int height, int steps)
    {
        if (!RegionalPromptResolver.HasRegionParts(prompt))
        {
            return null;
        }
        using Tensor baseCondPlaceholder = new Tensor(new TensorShape(1), DType.F32);
        return RegionalPromptResolver.Resolve(prompt, baseCondPlaceholder, width, height, steps, encodeRegion: text =>
        {
            int[] tokens = _t5Tokenizer.Encode(text);
            int[] mask = T5Tokenizer.CreateAttentionMask(tokens);
            return _pipeline.EncodeRegionText(tokens, mask);
        });
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _clipTokenizer.Dispose();
        _t5Tokenizer.Dispose();
        // The LoRA stack owns the merged tensors the transformer references, so it outlives them by exactly this much.
        _loraStack?.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
