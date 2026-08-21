using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae.Mage;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Drives a constructed Mage-Flow pipeline against the native <see cref="ImageRequest"/>: builds the
/// chat-templated Qwen3-VL token ids + prefix-drop index (Qwen-Image/Krea2 template — flagged as a parity risk vs
/// Mage-Flow's own <c>chat_template.json</c>), calls <see cref="MageFlowPipeline.GenerateFromTokens"/>, and converts
/// the returned <c>[1,3,H,W]</c> F32 image (in [-1,1]) to an RGB <see cref="ImageResult"/>.</summary>
public sealed unsafe class MageFlowRecipePipeline : IRecipePipeline
{
    // NOTE(parity): Mage-Flow applies its own tokenizer.chat_template. Until that template is transcribed, this
    // reuses the Qwen-Image/Krea2 system prompt + prefix-drop layout (same Qwen3-VL family). Verify on GPU.
    private const string SystemPrompt =
        "system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, " +
        "spatial relationships of the objects and background:";
    private const int MaxTokens = 512;

    private readonly MageFlowPipeline _pipeline;
    private readonly Qwen3Tokenizer _tokenizer;
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly QwenImageTransformer _transformer;
    private readonly MageVaeDecoder _vae;
    private readonly MageVaeEncoder? _vaeEncoder;
    private readonly bool _isTurbo;
    private readonly List<SafeTensorsLoader> _loaders;
    private readonly IDisposable? _ggufHandle;
    private readonly MergedLoraStack? _loraStack;
    private int _disposed;

    public MageFlowRecipePipeline(MageFlowPipeline pipeline, Qwen3Tokenizer tokenizer, LlamaStyleEncoder textEncoder,
        QwenImageTransformer transformer, MageVaeDecoder vae, MageVaeEncoder? vaeEncoder, bool isTurbo,
        List<SafeTensorsLoader> loaders, IDisposable? ggufHandle, MergedLoraStack? loraStack = null)
    {
        _loraStack = loraStack;
        _pipeline = pipeline; _tokenizer = tokenizer; _textEncoder = textEncoder; _transformer = transformer;
        _vae = vae; _vaeEncoder = vaeEncoder; _isTurbo = isTurbo; _loaders = loaders; _ggufHandle = ggufHandle;
    }

    public ImageDefaults? VariantDefaults => _isTurbo ? MageFlowRecipe.TurboDefaults : MageFlowRecipe.FamilyDefaults;

    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? (_isTurbo ? MageFlowRecipe.TurboDefaults.Steps : MageFlowRecipe.FamilyDefaults.Steps);
        float cfg = _isTurbo ? MageFlowRecipe.TurboDefaults.CfgScale : (request.CfgScale ?? MageFlowRecipe.FamilyDefaults.CfgScale);
        bool useCfg = cfg > 1.0f;

        (int reqW, int reqH) = RecipeRequestMapper.Size(request);
        int width = Math.Clamp(reqW / 16 * 16, 128, 4096);   // MageVAE /16, patch 1 → multiple of 16
        int height = Math.Clamp(reqH / 16 * 16, 128, 4096);
        int seed = RecipeRequestMapper.MapSeed(request.Seed) ?? Random.Shared.Next(int.MaxValue);

        (int[] promptTokens, int promptDrop) = EncodeWithTemplate(_tokenizer, prompt);
        (int[] negTokens, int negDrop) = EncodeWithTemplate(_tokenizer, negative);

        // Whether a checkpoint carries the encoder half is a property of the file, not of the family, so the recipe's
        // Supports bit (read before construction, to route the request) cannot express it. Refuse loudly here instead:
        // dropping the init image silently is the one outcome that looks like a working generation but is not.
        if (request.Img2Img is not null && _vaeEncoder is null)
        {
            throw new NotSupportedException(
                "This Mage-Flow checkpoint carries no VAE encoder weights, so it cannot condition on an init image. "
                + "Use an edit-capable Mage-Flow checkpoint, or remove the init image.");
        }

        // Edit: the init image (if any) is VAE-encoded to in-context reference latents. Resolved at the *snapped* size
        // so the source matches what the pipeline computes. The mask arg is null because Mage-Flow conditions on a
        // reference image and has no masked-inpaint path — its Supports omits Inpaint, so a mask is refused upstream.
        using Img2ImgResolver.Img2ImgSpec? editSpec = Img2ImgResolver.Resolve(request.Img2Img, null, width, height);

        progress?.Report(new StepPreview { Step = 0, TotalSteps = steps });
        Tensor image = _pipeline.GenerateFromTokens(promptTokens, promptDrop, useCfg ? negTokens : null, negDrop,
            width, height, steps, cfg, seed, editSpec?.SourceTensor, request.SeamlessTiling,
            request.VariationSeed?.Seed ?? -1, request.VariationSeed?.Strength ?? 0,
            // Mage-Flow takes primitives rather than a TextToImageRequest, so the sampler selection is threaded
            // explicitly. Validated by the resolver, which refuses an unavailable name instead of silently
            // substituting Euler.
            SamplingParamResolver.ResolveSchedulerName(request));
        progress?.Report(new StepPreview { Step = steps, TotalSteps = steps });

        byte[] rgb = ToRgbBytes(image, out int outW, out int outH);
        image.Dispose();

        return new ImageResult
        {
            Rgb = rgb, Width = outW, Height = outH, Seed = seed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "mage-flow",
                ["variant"] = _isTurbo ? "turbo" : "base",
                ["size"] = $"{outW}x{outH}",
                ["seed"] = seed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                ["cfg"] = cfg.ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    // [1,3,H,W] F32 in [-1,1] → interleaved RGB byte[] (H·W·3). Truncating .byte() cast (mage_flow parity).
    private static byte[] ToRgbBytes(Tensor image, out int W, out int H)
    {
        int c = (int)image.Shape[1]; H = (int)image.Shape[2]; W = (int)image.Shape[3];
        int hw = H * W;
        byte[] rgb = new byte[hw * 3];
        float* p = (float*)image.DataPointer;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
            {
                int pix = y * W + x;
                for (int ch = 0; ch < 3; ch++)
                {
                    float v = p[(long)ch * hw + pix];
                    v = v < -1f ? -1f : v > 1f ? 1f : v;
                    rgb[pix * 3 + ch] = (byte)(127.5f * (v + 1f));   // truncation, matching torch .byte()
                }
            }
        return rgb;
    }

    private static (int[] tokens, int dropIndex) EncodeWithTemplate(Qwen3Tokenizer tokenizer, string prompt)
    {
        List<int> ids = new(64) { Qwen3Tokenizer.ImStartId };
        ids.AddRange(tokenizer.EncodeRaw(SystemPrompt));
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
        if (ids.Count > MaxTokens) ids.RemoveRange(MaxTokens, ids.Count - MaxTokens);
        return (ids.ToArray(), dropIndex);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _pipeline.Dispose();
        _vae.Dispose();
        _vaeEncoder?.Dispose();
        _transformer.Dispose();
        foreach (SafeTensorsLoader loader in _loaders) loader.Dispose();
        _ggufHandle?.Dispose();
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
