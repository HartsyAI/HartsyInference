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
using HartsyInference.ModelAssets.Tokenizers;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Flux.1 pipeline driven against the native <see cref="ImageRequest"/>. <see cref="FluxPipeline"/> owns the CLIP-L + T5-XXL encoders, so this tokenizes the prompt with both (CLIP-L for the pooled EOS vector, T5-XXL per-token plus its attention mask) and calls <see cref="FluxPipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>FluxLoader.Generate</c> vanilla text-to-image drive path. Wraps the constructed Flux.1 pipeline plus its tokenizers and merged LoRA stack, taking ownership of every disposable.</summary>
/// <param name="isDev">Selects the step fallback and whether the embedded distilled guidance is applied.</param>
/// <param name="loaders">Whatever keeps the weights alive: the open checkpoint — either container, plus any copies the backend needed widened — and one loader per component resolved as a separate file.</param>
public sealed class Flux1RecipePipeline(FluxPipeline pipeline, ClipTokenizer clipTokenizer, T5Tokenizer t5Tokenizer, bool isDev,
    List<IDisposable> loaders, MergedLoraStack? loraStack, IBackend backend) : IRecipePipeline
{
    private readonly FluxPipeline _pipeline = pipeline;
    private readonly ClipTokenizer _clipTokenizer = clipTokenizer;
    private readonly T5Tokenizer _t5Tokenizer = t5Tokenizer;
    private readonly bool _isDev = isDev;
    private readonly List<IDisposable> _loaders = loaders;
    private readonly MergedLoraStack? _loraStack = loraStack;
    private readonly IBackend _backend = backend;

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

        // Declaring ComfyBlend is what stops ImagesService collapsing `(word:N)`, so the recipe owns the grammar
        // now. Only the T5 arm is blendable: CLIP-L contributes its POOLED vector and its hidden states are
        // discarded, and ComfyUI's blend rewrites hidden states only.
        bool hasRegionParts = RegionalPromptResolver.HasRegionParts(prompt);
        if (hasRegionParts && RegionalPromptWeightSplit.BaseTextCarriesWeight(prompt))
        {
            throw new NotSupportedException(
                "Flux.1 cannot weight the base prompt and a region in the same request: the base encode covers "
                + "the region tags too, so the two sets of weights would land on the same conditioning rows. "
                + "Move the emphasis inside the region, or drop the region tags.");
        }
        string baseText = RegionalPromptWeightSplit.BaseText(prompt, hasRegionParts);
        int[] clipTokens = _clipTokenizer.Encode(PromptWeighting.Join(PromptWeighting.Parse(baseText)));
        int eosPos = ClipTokenizer.FindEosPosition(clipTokens);
        (int[] t5Tokens, float[]? t5Weights) = T5WeightedConditioning.Tokenize(_t5Tokenizer, baseText);
        int[] t5Mask = T5Tokenizer.CreateAttentionMask(t5Tokens);
        int[] emptyT5 = T5WeightedConditioning.EmptyTokens(_t5Tokenizer);

        // FLUX.1 Canny/Depth: the host already ran the edge/depth annotator (it needs host-app image types this
        // package can't reference) and handed back the finished map under this key. Absent on a genuine Tools
        // checkpoint surfaces as GenerateFromTokens' own "requires a control image" error, not a silent fallback.
        ImageData? toolsControlImageData = RequestExtras.Image(request.Extra, RequestExtras.FluxToolsControlImage);
        Tensor? controlImage = toolsControlImageData is null ? null
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

            // Variation blending happens inside the pipeline's own TakeOrCreateNoise now (from the two request
            // fields below) — on img2img too, matching ComfyUI, where var_seed applies to the sampler noise
            // regardless of the latent's source.
            TextToImageRequest inner = RecipeImg2ImgBinder.Apply(
                new TextToImageRequest
                {
                    SeamlessTiling = request.SeamlessTiling,
                    VariationSeed = request.VariationSeed?.Seed ?? -1,
                    VariationSeedStrength = request.VariationSeed?.Strength ?? 0,
                    Prompt = prompt,
                    Width = request.Width,
                    Height = request.Height,
                    Steps = steps,
                    Seed = RecipeRequestMapper.MapSeed(request.Seed),
                    // Routed through the resolver rather than read raw, so an unavailable sampler is refused by name here —
                    // before the checkpoint loads — instead of deep inside the pipeline, or silently dropped.
                    Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
                },
                img2img);

            Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel);

            regionalPlan = BuildRegionalPlan(prompt, reqWidth, reqHeight, steps);

            (byte[] rgb, int width, int height, int usedSeed) = _pipeline.GenerateFromTokens(
                clipTokens, eosPos, t5Tokens, t5Mask, inner,
                guidanceScale: guidance,
                onProgress: bridge,
                controlImage: controlImage,
                regionalPlan: regionalPlan,
                fluxControlNets: controlNets?.Conditionings,
                reduxImageEmbeds: redux?.Embeds,
                reduxApplyStartFraction: redux?.ApplyStart ?? 0f,
                promptWeights: t5Weights,
                emptyTokenIdsT5: emptyT5,
                emptyAttentionMaskT5: T5Tokenizer.CreateAttentionMask(emptyT5));

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

    /// <summary>Builds a regional-conditioning plan when the prompt carries <c>&lt;region:&gt;</c>/<c>&lt;object:&gt;</c> parts, null otherwise. <see cref="RegionalPlan.BaseCond"/> is a required field on the resolver's signature but is never read by <see cref="FluxPipeline.GenerateFromTokens"/>'s regional path (confirmed by inspection: the pipeline builds its actual background stream from its own internally-computed T5 embeddings, not this field) — a throwaway placeholder, disposed immediately rather than kept alive for the plan's lifetime.</summary>
    private RegionalPlan? BuildRegionalPlan(string prompt, int width, int height, int steps)
    {
        if (!RegionalPromptResolver.HasRegionParts(prompt))
        {
            return null;
        }
        using Tensor baseCondPlaceholder = new Tensor(new TensorShape(1), DType.F32);
        return RegionalPromptResolver.Resolve(prompt, baseCondPlaceholder, width, height, steps, encodeRegion: text =>
        {
            // Each region is its own leaf carrying its own emphasis, as encode_leaves does per region.
            (int[] tokens, float[]? weights) =
                T5WeightedConditioning.Tokenize(_t5Tokenizer, PromptTagFlattening.Flatten(text));
            int[] mask = T5Tokenizer.CreateAttentionMask(tokens);
            int[] empty = T5WeightedConditioning.EmptyTokens(_t5Tokenizer);
            return _pipeline.EncodeRegionText(
                tokens, mask, weights, empty, T5Tokenizer.CreateAttentionMask(empty));
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
        foreach (IDisposable loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
