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

/// <summary>A constructed HiDream-I1 pipeline driven against the native <see cref="ImageRequest"/>. <see cref="HiDreamPipeline"/> owns all four encoder forwards, so this only tokenizes: one CLIP BPE pass reused for both CLIP-L and CLIP-G (the reference feeds identical ids to both), a T5 pass with its attention mask, and a Llama-3.1 pass — then calls <see cref="HiDreamPipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>HiDreamLoader.Generate</c> drive path.</summary>
public sealed class HiDreamRecipePipeline : IRecipePipeline
{
    private readonly HiDreamPipeline _pipeline;
    private readonly ClipTokenizer _clipTokenizer;
    private readonly T5Tokenizer _t5Tokenizer;
    private readonly LlamaTokenizer _llamaTokenizer;
    private readonly T5TextEncoder _t5;
    private readonly LlamaStyleEncoder _llama;
    private readonly HiDreamTransformer _transformer;
    private readonly List<SafeTensorsLoader> _loaders;

    private readonly MergedLoraStack? _loraStack;

    /// <summary>Wraps the constructed HiDream pipeline plus its tokenizers and heavyweight components, taking ownership of every disposable.</summary>
    public HiDreamRecipePipeline(HiDreamPipeline pipeline, ClipTokenizer clipTokenizer, T5Tokenizer t5Tokenizer, LlamaTokenizer llamaTokenizer, T5TextEncoder t5, LlamaStyleEncoder llama, HiDreamTransformer transformer, List<SafeTensorsLoader> loaders, MergedLoraStack? loraStack = null)
    {
        _loraStack = loraStack;
        _pipeline = pipeline;
        _clipTokenizer = clipTokenizer;
        _t5Tokenizer = t5Tokenizer;
        _llamaTokenizer = llamaTokenizer;
        _t5 = t5;
        _llama = llama;
        _transformer = transformer;
        _loaders = loaders;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? HiDreamRecipe.FamilyDefaults.Steps;
        float cfg = request.CfgScale ?? HiDreamRecipe.FamilyDefaults.CfgScale;

        // TODO(E-IMG-4/5): LoRA, ControlNet, IP-Adapter, refiner, regional prompting, weighted
        // conditioning and ImageRequest.Components overrides are deferred — text-to-image only.
        int[] clipTokens = _clipTokenizer.Encode(prompt);
        int[] negClipTokens = _clipTokenizer.Encode(negative);
        int eosPos = ClipTokenizer.FindEosPosition(clipTokens);
        int negEosPos = ClipTokenizer.FindEosPosition(negClipTokens);

        // Always tokenize the negative: the pipeline's parameters are non-optional and it decides internally
        // whether to run the negative pass (cfg > 1).
        int[] t5Tokens = _t5Tokenizer.Encode(prompt);
        int[] negT5Tokens = _t5Tokenizer.Encode(negative);
        int[] t5Mask = T5Tokenizer.CreateAttentionMask(t5Tokens);
        int[] negT5Mask = T5Tokenizer.CreateAttentionMask(negT5Tokens);

        int[] llamaTokens = _llamaTokenizer.Encode(prompt);
        int[] negLlamaTokens = _llamaTokenizer.Encode(negative);

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
            clipTokens, negClipTokens,
            clipTokens, negClipTokens,
            eosPos, negEosPos,
            eosPos, negEosPos,
            t5Tokens, negT5Tokens,
            t5Mask, negT5Mask,
            llamaTokens, negLlamaTokens,
            inner, bridge);

        return new ImageResult
        {
            Rgb = rgb,
            Width = outW,
            Height = outH,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "hidream",
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
        _clipTokenizer.Dispose();
        _t5Tokenizer.Dispose();
        _llamaTokenizer.Dispose();
        _t5.Dispose();
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
