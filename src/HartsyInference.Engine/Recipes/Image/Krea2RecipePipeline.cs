using System.Globalization;
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Krea 2 pipeline driven against the native <see cref="ImageRequest"/>. <see cref="Krea2Pipeline"/> owns the Qwen3-VL-4B forward (it taps 12 decoder layers itself), so this only builds the templated token ids — byte-identical to Qwen-Image's template — plus the prefix-drop indices and calls <see cref="Krea2Pipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>Krea2Loader.Generate</c>.</summary>
public sealed class Krea2RecipePipeline : IRecipePipeline
{
    /// <summary>Krea 2's prompt template is byte-identical to Qwen-Image's — same system prompt, same prefix-drop design.</summary>
    private const string Krea2SystemPrompt =
        "system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, " +
        "spatial relationships of the objects and background:";

    /// <summary>The templated sequence is truncated at 512 tokens.</summary>
    private const int MaxTokens = 512;

    private readonly Krea2Pipeline _pipeline;
    private readonly Qwen3Tokenizer _tokenizer;
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly Krea2Transformer _transformer;
    private readonly QwenImageVaeDecoder _vae;
    private readonly bool _isTurbo;
    private readonly List<SafeTensorsLoader> _loaders;

    /// <summary>Wraps the constructed Krea 2 pipeline plus its components, taking ownership of every disposable.</summary>
    public Krea2RecipePipeline(Krea2Pipeline pipeline, Qwen3Tokenizer tokenizer, LlamaStyleEncoder textEncoder, Krea2Transformer transformer, QwenImageVaeDecoder vae, bool isTurbo, List<SafeTensorsLoader> loaders)
    {
        _pipeline = pipeline;
        _tokenizer = tokenizer;
        _textEncoder = textEncoder;
        _transformer = transformer;
        _vae = vae;
        _isTurbo = isTurbo;
        _loaders = loaders;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        // Turbo: 8 steps, guidance off. Base: 28 steps, CFG 4.5.
        int steps = request.Steps <= 0 ? (_isTurbo ? 8 : 28) : request.Steps;
        float cfg = _isTurbo ? 1.0f : (request.CfgScale <= 0 ? 4.5f : request.CfgScale);
        bool useCfg = cfg > 1.0f;

        // Width/height must be multiples of 16 (2×2 patchify × 8× VAE), clamped to 128–4096.
        int width = Math.Clamp(request.Width / 16 * 16, 128, 4096);
        int height = Math.Clamp(request.Height / 16 * 16, 128, 4096);
        if (width != request.Width || height != request.Height)
        {
            Logs.Info($"[Krea2RecipePipeline] Snapped {request.Width}x{request.Height} → {width}x{height} (multiple of 16, 128–4096).");
        }

        // TODO(E-IMG-4/5): img2img/inpaint, LoRA, ControlNet, IP-Adapter and regional prompting are deferred.
        (int[] promptTokens, int promptDrop) = EncodeWithTemplate(_tokenizer, prompt);
        (int[] negTokens, int negDrop) = EncodeWithTemplate(_tokenizer, negative);

        TextToImageRequest inner = new TextToImageRequest
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = width,
            Height = height,
            Steps = steps,
            CfgScale = cfg,
            Seed = request.Seed < 0 ? null : (int?)(int)(request.Seed & 0x7FFFFFFF),
        };

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = steps });
        };

        (byte[] rgb, int outW, int outH, int usedSeed) = _pipeline.GenerateFromTokens(
            promptTokens, useCfg ? negTokens : null, inner, bridge,
            promptDropIndex: promptDrop, negativeDropIndex: negDrop);

        return new ImageResult
        {
            Rgb = rgb,
            Width = outW,
            Height = outH,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "krea2",
                ["variant"] = _isTurbo ? "turbo" : "base",
                ["size"] = $"{outW}x{outH}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                ["cfg"] = cfg.ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <summary>Builds the Krea 2 templated token sequence plus the prefix-drop index (the leading system-block + user-header positions whose hidden states the pipeline discards — Krea 2's <c>prompt_template_encode_start_idx</c>).</summary>
    private static (int[] tokens, int dropIndex) EncodeWithTemplate(Qwen3Tokenizer tokenizer, string prompt)
    {
        List<int> ids = new List<int>(64);
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw(Krea2SystemPrompt));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("user\n"));
        int dropIndex = ids.Count;
        ids.AddRange(tokenizer.EncodeRaw(prompt));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("assistant\n"));
        if (ids.Count > MaxTokens)
        {
            ids.RemoveRange(MaxTokens, ids.Count - MaxTokens);
        }
        return (ids.ToArray(), dropIndex);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _tokenizer.Dispose();
        _textEncoder.Dispose();
        _transformer.Dispose();
        _vae.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
