using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Globalization;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed HunyuanImage 2.1 pipeline driven against the native <see cref="ImageRequest"/>. <see cref="HunyuanImagePipeline"/> owns the Qwen2.5-VL-7B forward (including its own TE⇄DiT residency swap and embedding cache), so this only produces the padded chat-template token ids + attention masks and calls <see cref="HunyuanImagePipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>HunyuanImageLoader.Generate</c>.</summary>
public sealed class HunyuanImageRecipePipeline : IRecipePipeline
{
    /// <summary>HunyuanImage's guidance default when the caller does not supply one (SwarmUI loader fallback).</summary>
    private const float DefaultCfg = 3.5f;

    private readonly HunyuanImagePipeline _pipeline;
    private readonly Qwen2Tokenizer _tokenizer;
    private readonly LlamaStyleEncoder _llama;
    private readonly HunyuanImageQwenTextEncoder _qwenEncoder;
    private readonly HunyuanImageTransformer _transformer;
    private readonly HunyuanImageVaeDecoder _vae;
    private readonly List<SafeTensorsLoader> _loaders;
    private readonly IDisposable? _ggufHandle;
    private readonly MergedLoraStack? _loraStack;

    /// <summary>Wraps the constructed HunyuanImage pipeline plus its components, taking ownership of every disposable.</summary>
    public HunyuanImageRecipePipeline(HunyuanImagePipeline pipeline, Qwen2Tokenizer tokenizer, LlamaStyleEncoder llama, HunyuanImageQwenTextEncoder qwenEncoder, HunyuanImageTransformer transformer, HunyuanImageVaeDecoder vae, List<SafeTensorsLoader> loaders, IDisposable? ggufHandle, MergedLoraStack? loraStack = null)
    {
        _loraStack = loraStack;
        _pipeline = pipeline;
        _tokenizer = tokenizer;
        _llama = llama;
        _qwenEncoder = qwenEncoder;
        _transformer = transformer;
        _vae = vae;
        _loaders = loaders;
        _ggufHandle = ggufHandle;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? HunyuanImageRecipe.FamilyDefaults.Steps;
        float cfg = request.CfgScale ?? HunyuanImageRecipe.FamilyDefaults.CfgScale;
        // 32× VAE + unit patches → the image dims must be a multiple of 32.
        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        int width = (reqWidth / 32) * 32;
        int height = (reqHeight / 32) * 32;

        // TODO(E-IMG-4/5): img2img/inpaint, LoRA, ControlNet, regional prompting and the ByT5 glyph branch are
        // deferred — text-to-image only.
        (int[] ids, int[] mask) = TokenizePadded(_tokenizer, prompt);
        bool useCfg = cfg > 1.0f;
        // An empty negative encodes to exactly the 34-token template, which the encoder's prefix-drop rejects —
        // give it one real token.
        if (useCfg && string.IsNullOrWhiteSpace(negative))
        {
            negative = ".";
        }
        int[]? negIds = null;
        int[]? negMask = null;
        if (useCfg)
        {
            (negIds, negMask) = TokenizePadded(_tokenizer, negative);
        }

        // Resolved at the same width/height the inner request carries — HunyuanImage rejects sizes that are not a
        // multiple of 32 rather than snapping, so these are already the dimensions the pipeline validates against.
        using Img2ImgResolver.Img2ImgSpec? img2img = RecipeImg2ImgBinder.Resolve(request, width, height);
        TextToImageRequest inner = RecipeImg2ImgBinder.Apply(
            new TextToImageRequest
            {
                SeamlessTiling = request.SeamlessTiling,
                    VariationSeed = request.VariationSeed?.Seed ?? -1,
                    VariationSeedStrength = request.VariationSeed?.Strength ?? 0,
                Prompt = prompt,
                Width = width,
                Height = height,
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
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = steps });
        };

        (byte[] rgb, int outW, int outH, int usedSeed) = _pipeline.GenerateFromTokens(
            ids, mask, negIds, negMask, inner, onProgress: bridge);

        return new ImageResult
        {
            Rgb = rgb,
            Width = outW,
            Height = outH,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "hunyuan-image",
                ["size"] = $"{outW}x{outH}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                ["cfg"] = cfg.ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <summary>Chat-template encode padded to the fixed 1034-token window with a matching attention mask (diffusers <c>_get_qwen_prompt_embeds</c>).</summary>
    private static (int[] ids, int[] mask) TokenizePadded(Qwen2Tokenizer tokenizer, string prompt)
    {
        int[] raw = tokenizer.EncodeChat(prompt, systemPrompt: HunyuanImageQwenTextEncoder.SystemPrompt, addGenerationPrompt: false);
        int realLen = Math.Min(raw.Length, HunyuanImageQwenTextEncoder.PaddedLength);
        int[] ids = Qwen2Tokenizer.PadToLength(raw, HunyuanImageQwenTextEncoder.PaddedLength);
        int[] mask = new int[HunyuanImageQwenTextEncoder.PaddedLength];
        for (int i = 0; i < realLen; i++)
        {
            mask[i] = 1;
        }
        return (ids, mask);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _qwenEncoder.Dispose();
        _llama.Dispose();
        _transformer.Dispose();
        _vae.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
        _ggufHandle?.Dispose();
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
