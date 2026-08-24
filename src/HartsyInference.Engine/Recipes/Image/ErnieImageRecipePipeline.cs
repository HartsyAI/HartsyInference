using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Globalization;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed ERNIE-Image pipeline driven against the native <see cref="ImageRequest"/>. <see cref="ErnieImagePipeline"/> owns the Ministral-3-3B forward and self-manages its GPU preload/free per generation, so this only tokenizes (raw prompt, no chat template — BOS prepended, EOS appended, no padding; the pipeline assembles the sequence) and calls <see cref="ErnieImagePipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>ErnieImageLoader.Generate</c> drive path. Wraps the constructed ERNIE-Image pipeline plus its tokenizer and text stack, taking ownership of every disposable.</summary>
public sealed class ErnieImageRecipePipeline(ErnieImagePipeline pipeline, ErnieTokenizer tokenizer, ErnieImageLlamaTextEncoder textEncoder,
    LlamaStyleEncoder llama, ErnieImageTransformer transformer, List<SafeTensorsLoader> loaders, MergedLoraStack? loraStack = null) : IRecipePipeline
{
    private readonly ErnieImagePipeline _pipeline = pipeline;
    private readonly ErnieTokenizer _tokenizer = tokenizer;
    private readonly ErnieImageLlamaTextEncoder _textEncoder = textEncoder;
    private readonly LlamaStyleEncoder _llama = llama;
    private readonly ErnieImageTransformer _transformer = transformer;
    private readonly List<SafeTensorsLoader> _loaders = loaders;
    private readonly MergedLoraStack? _loraStack = loraStack;

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? ErnieImageRecipe.FamilyDefaults.Steps;
        float cfg = request.CfgScale ?? ErnieImageRecipe.FamilyDefaults.CfgScale;

        // TODO(E-IMG-4/5): img2img/inpaint, LoRA, ControlNet, IP-Adapter, refiner, regional prompting and
        // ImageRequest.Components overrides are deferred — text-to-image only.
        // The negative is only consumed when cfg > 1, but it tokenizes cheaply either way.
        int[] promptTokens = _tokenizer.Encode(prompt);
        int[] negTokens = _tokenizer.Encode(negative);

        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        using Img2ImgResolver.Img2ImgSpec? img2img = RecipeImg2ImgBinder.Resolve(request, reqWidth, reqHeight);
        TextToImageRequest inner = RecipeImg2ImgBinder.Apply(
            new TextToImageRequest
            {
                SeamlessTiling = request.SeamlessTiling,
                    VariationSeed = request.VariationSeed?.Seed ?? -1,
                    VariationSeedStrength = request.VariationSeed?.Strength ?? 0,
                Prompt = prompt,
                NegativePrompt = negative,
                Width = request.Width,
                Height = request.Height,
                Steps = steps,
                CfgScale = cfg,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
                // Routed through the resolver rather than read raw, so an unavailable sampler is refused by name here —
                // before the checkpoint loads — instead of deep inside the pipeline, or silently dropped.
                Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
            },
            img2img);

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
        };

        (byte[] rgb, int outW, int outH, int usedSeed) = _pipeline.GenerateFromTokens(
            promptTokens, negTokens, promptTokens.Length, negTokens.Length, inner, bridge);

        return new ImageResult
        {
            Rgb = rgb,
            Width = outW,
            Height = outH,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "ernie-image",
                ["size"] = $"{outW}x{outH}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                ["cfg"] = cfg.ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _tokenizer.Dispose();
        _textEncoder.Dispose();
        _llama.Dispose();
        _transformer.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
