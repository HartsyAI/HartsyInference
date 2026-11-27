using System.Globalization;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
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

/// <summary>A constructed Flux.2 pipeline driven against the native <see cref="ImageRequest"/>. <see cref="Flux2Pipeline"/> owns the text encoder, so this only produces the token ids — the embedded Qwen3 chat template for Klein, or the spliced Mistral tekken conditioning ids for Dev — and calls <see cref="Flux2Pipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>Flux2Loader.Generate</c> text-to-image drive path. Wraps the constructed Flux.2 pipeline plus its tokenizer, taking ownership of every disposable. Exactly one of <paramref name="qwenTokenizer"/> (Klein) / <paramref name="mistralTokenizer"/> (Dev) is non-null. <paramref name="checkpoint"/> holds whatever keeps the transformer's weights alive — the checkpoint's memory map, which pass-through tensors still point into, and any copies the backend needed widened.</summary>
public sealed class Flux2RecipePipeline(Flux2Pipeline pipeline, Flux2Config config, Qwen3Tokenizer? qwenTokenizer, ErnieTokenizer? mistralTokenizer,
    string mistralSystemPrompt, LlamaStyleEncoder encoder, List<IDisposable> loaders, IDisposable? checkpoint = null,
    MergedLoraStack? loraStack = null) : IRecipePipeline
{
    private readonly Flux2Pipeline _pipeline = pipeline;
    private readonly Flux2Config _config = config;
    private readonly Qwen3Tokenizer? _qwenTokenizer = qwenTokenizer;
    private readonly ErnieTokenizer? _mistralTokenizer = mistralTokenizer;
    private readonly Flux2PromptTokenizer _tokenizer = new(config, qwenTokenizer, mistralTokenizer, mistralSystemPrompt);
    private readonly LlamaStyleEncoder _encoder = encoder;
    private readonly List<IDisposable> _loaders = loaders;
    private int _disposed;
    private readonly IDisposable? _checkpoint = checkpoint;

    private readonly MergedLoraStack? _loraStack = loraStack;

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, ImageEffectiveSettings settings, IProgress<StepPreview>? progress,
        CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        int steps = settings.Steps;
        // Flux.2 rounds image dims down to a multiple of 16 (VAE 8× × 2×2 patch).
        (int reqWidth, int reqHeight) = (settings.Width, settings.Height);
        int width = (reqWidth / 16) * 16;
        int height = (reqHeight / 16) * 16;
        // Klein has no guidance embedding; Dev uses guidance ~3.5 (BFL distillation target).
        float guidance = _config.GuidanceEmbed ? 3.5f : 0f;

        // TODO(E-IMG-4/5): img2img, NegativePrompt mapping, and user component overrides are deferred.
        // <alternate:>/<fromto[N]:> now survive ImagesService, because this recipe declares
        // ImageFeatures.PromptScheduling. Everything OUTSIDE the schedule builder therefore needs them collapsed to
        // their step-0 value first, or the literal tag text is tokenized as prose — the same split SdxlRecipePipeline
        // makes for the same reason. Idempotent when nothing schedules, and a no-op on the weight tags ImagesService
        // already turned into parens.
        string flatPrompt = PromptTagFlattening.Flatten(prompt);
        WeightedTokenSequence tokens = _tokenizer.Tokenize(flatPrompt);
        WeightedTokenSequence? promptWeights = tokens.IsUniformlyUnweighted ? null : tokens;
        // Each distinct step-text goes through the SAME Tokenize, so a branch carrying its own emphasis is weighted
        // per branch. Null when the prompt schedules nothing, which keeps an ordinary request on the single-encode
        // path; a schedule that collapses to one variant also lands there, on the flattened ids above.
        ScheduledPrompt? promptSchedule = ScheduledPrompt.TryBuild(prompt, steps, _tokenizer.Tokenize);

        // Resolved at the 16-rounded size Flux2Pipeline validates against.
        using Img2ImgResolver.Img2ImgSpec? img2img = RecipeImg2ImgBinder.Resolve(request, width, height);
        TextToImageRequest inner = RecipeImg2ImgBinder.Apply(
            new TextToImageRequest
            {
                SigmaShift = RecipeRequestMapper.MapFluxMaxShift(request.SigmaShift),
                SeamlessTiling = request.SeamlessTiling,
                VariationSeed = request.VariationSeed?.Seed ?? -1,
                VariationSeedStrength = request.VariationSeed?.Strength ?? 0,
                Prompt = prompt,
                Width = width,
                Height = height,
                Steps = steps,
                CfgScale = settings.CfgScale,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
                // Routed through the resolver rather than read raw, so an unavailable sampler is refused by name here —
                // before the checkpoint loads — instead of deep inside the pipeline, or silently dropped.
                Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
            },
            img2img);

        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel);

        RegionalPlan? regionalPlan = null;
        try
        {
            regionalPlan = BuildRegionalPlan(prompt, width, height, steps);

            (byte[] rgb, int outW, int outH, int usedSeed) = _pipeline.GenerateFromTokens(
                tokens.Tokens, inner, guidanceScale: guidance, onProgress: bridge, regionalPlan: regionalPlan,
                promptWeights: promptWeights, promptSchedule: promptSchedule);

            return new ImageResult
            {
                Rgb = rgb,
                Width = outW,
                Height = outH,
                Seed = usedSeed,
                Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["arch"] = "flux2",
                    ["size"] = $"{outW}x{outH}",
                    ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                    ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                },
            };
        }
        finally
        {
            RegionalPromptResolver.DisposeRegions(regionalPlan);
        }
    }

    /// <summary>Builds a regional-conditioning plan when the prompt carries <c>&lt;region:&gt;</c>/<c>&lt;object:&gt;</c> parts, null otherwise (Tier 3.7). Mirrors <c>Flux1RecipePipeline.BuildRegionalPlan</c>: each region's text is tokenized the SAME way the base prompt was (Klein's chat template vs. Dev's Mistral splice) and encoded through <see cref="Flux2Pipeline.EncodeRegionText"/> — the same text-encoder instance + hidden-layer taps the base prompt uses. <see cref="RegionalPlan.BaseCond"/> is a required field on the resolver's signature that <see cref="Flux2Pipeline.GenerateFromTokens"/>'s regional path never reads (same as Flux.1 — confirmed by inspection: the pipeline builds its own background stream from the base <c>textEmbeddings</c>) — a throwaway placeholder tensor satisfies it.</summary>
    private RegionalPlan? BuildRegionalPlan(string prompt, int width, int height, int steps)
    {
        if (!RegionalPromptResolver.HasRegionParts(prompt))
        {
            return null;
        }
        using Tensor baseCondPlaceholder = new Tensor(new TensorShape(1), DType.F32);
        return RegionalPromptResolver.Resolve(prompt, baseCondPlaceholder, width, height, steps, encodeRegion: text =>
        {
            // Regions are not weighted yet (C.2 work): strip the emphasis rather than let the parens and digits
            // reach the encoder as prose, which is what SwarmUI does for anything it is not weighting.
            return _pipeline.EncodeRegionText(_tokenizer.Tokenize(PromptWeighting.Join(PromptWeighting.Parse(text))).Tokens);
        });
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        // Every handle is released even if one throws; the LoRA stack goes last, it owns the merged weights.
        new CompositeDisposable([_pipeline, _qwenTokenizer, _mistralTokenizer, _encoder,
            .. _loaders, _checkpoint, _loraStack]).Dispose();
    }
}
