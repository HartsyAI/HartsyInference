using System.Globalization;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Pipelines;
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

    /// <summary>Wraps the constructed Flux.1 pipeline plus its tokenizers and merged LoRA stack, taking ownership of every disposable. <paramref name="isDev"/> selects the step fallback and whether the embedded distilled guidance is applied.</summary>
    public Flux1RecipePipeline(FluxPipeline pipeline, ClipTokenizer clipTokenizer, T5Tokenizer t5Tokenizer, bool isDev, List<SafeTensorsLoader> loaders, MergedLoraStack? loraStack)
    {
        _pipeline = pipeline;
        _clipTokenizer = clipTokenizer;
        _t5Tokenizer = t5Tokenizer;
        _isDev = isDev;
        _loaders = loaders;
        _loraStack = loraStack;
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
        // Schnell) — defaulted here. True-CFG (trueCfgScale + negative prompt), img2img/inpaint, Tools/Kontext/Redux/
        // ControlNet are deferred, so NegativePrompt / CfgScale are not mapped for the base path.
        float guidance = _isDev ? 3.5f : 0f;

        int[] clipTokens = _clipTokenizer.Encode(prompt);
        int eosPos = ClipTokenizer.FindEosPosition(clipTokens);
        int[] t5Tokens = _t5Tokenizer.Encode(prompt);
        int[] t5Mask = T5Tokenizer.CreateAttentionMask(t5Tokens);

        FluxControlNetResolver.ResolvedSpec? controlNets = null;
        try
        {
            controlNets = FluxControlNetResolver.Resolve(
                request.ControlNets, reqWidth, reqHeight,
                static message => Logs.Info($"[Features][ControlNet] {message}"));

            Tensor? variationNoise = VariationSeedResolver.Resolve(
                request.VariationSeed, reqWidth, reqHeight, request.Seed, VariationSeedResolver.FluxLatentChannels);

            TextToImageRequest inner = new TextToImageRequest
            {
                Prompt = prompt,
                Width = request.Width,
                Height = request.Height,
                Steps = steps,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
                InitialNoise = variationNoise,
            };

            Action<GenerationProgress> bridge = p =>
            {
                cancel.ThrowIfCancellationRequested();
                progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
            };

            (byte[] rgb, int width, int height, int usedSeed) = _pipeline.GenerateFromTokens(
                clipTokens, eosPos, t5Tokens, t5Mask, inner,
                guidanceScale: guidance,
                onProgress: bridge,
                fluxControlNets: controlNets?.Conditionings);

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
            controlNets?.Dispose();
        }
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
