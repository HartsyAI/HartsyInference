using System.Globalization;
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Ideogram 4 pipeline driven against the native <see cref="ImageRequest"/>. <see cref="Ideogram4Pipeline"/> owns the Qwen3-VL forward, so this only chat-templates and trims the prompt tokens, snaps the resolution to Ideogram's 16-pixel grid, and maps <see cref="ImageRequest.Steps"/> onto the nearest official sampler preset (the preset carries the per-step asymmetric-CFG guidance schedule, so CfgScale and the negative prompt are ignored by design). Mirrors the SwarmUI backend's <c>Ideogram4Loader.Generate</c> drive path.</summary>
public sealed class Ideogram4RecipePipeline : IRecipePipeline
{
    private readonly Ideogram4Pipeline _pipeline;
    private readonly Qwen3Tokenizer _tokenizer;
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly Ideogram4Transformer _conditional;
    private readonly Ideogram4Transformer _unconditional;
    private readonly List<SafeTensorsLoader> _loaders;

    /// <summary>Wraps the constructed Ideogram 4 pipeline plus its tokenizer and both transformers, taking ownership of every disposable.</summary>
    public Ideogram4RecipePipeline(Ideogram4Pipeline pipeline, Qwen3Tokenizer tokenizer, LlamaStyleEncoder textEncoder, Ideogram4Transformer conditional, Ideogram4Transformer unconditional, List<SafeTensorsLoader> loaders)
    {
        _pipeline = pipeline;
        _tokenizer = tokenizer;
        _textEncoder = textEncoder;
        _conditional = conditional;
        _unconditional = unconditional;
        _loaders = loaders;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;

        // Width/height must be multiples of 16 (2x2 patchify x 8x VAE), range 256-2048.
        int snappedW = Math.Clamp(request.Width / 16 * 16, 256, 2048);
        int snappedH = Math.Clamp(request.Height / 16 * 16, 256, 2048);
        if (snappedW != request.Width || snappedH != request.Height)
        {
            Logs.Info($"[Ideogram4] Snapped resolution {request.Width}x{request.Height} -> {snappedW}x{snappedH}.");
        }

        // Steps pick the nearest official preset; the preset owns the guidance schedule (gw~7 main + gw~3 polish)
        // and the logit-normal mu/std, so CfgScale and the negative prompt are intentionally not consumed.
        int steps = request.Steps <= 0 ? 20 : request.Steps;
        Ideogram4SamplerPreset preset = steps <= 14 ? Ideogram4SamplerPreset.Turbo12
            : steps >= 40 ? Ideogram4SamplerPreset.Quality48
            : Ideogram4SamplerPreset.Default20;
        if (steps != preset.NumSteps)
        {
            Logs.Info($"[Ideogram4] Steps={steps} mapped to official preset {preset.Name} ({preset.NumSteps} steps — Ideogram 4 uses fixed preset schedules).");
        }
        if (!string.IsNullOrWhiteSpace(request.NegativePrompt))
        {
            Logs.Info("[Ideogram4] Negative prompt is ignored — Ideogram 4's asymmetric CFG runs the unconditional branch with zeroed text features.");
        }

        // TODO(E-IMG-5): the SwarmUI host wrapped a plain prompt into Ideogram's structured JSON caption shape
        // (optionally via an LLM "magic prompt" expansion). That is a host-side prompt helper and is NOT ported —
        // the prompt is fed verbatim. Ideogram 4 was trained only on structured JSON captions, so a bare prompt
        // frequently trips its built-in safety filter (grey placeholder output); supply a structured caption.
        Logs.Warning("[Ideogram4] Prompt fed verbatim — supply a structured JSON caption for reliable results (magic-prompt expansion is host-side and not wired here).");

        // includeThinkBlock:false — Ideogram's encoder is Qwen3-VL-8B-Instruct, whose chat template ends the
        // generation prompt at "<|im_start|>assistant\n" with no <think> block. EncodeChat right-pads to maxLength
        // with BOS; feeding ~2048 mostly-pad tokens would dilute conditioning and multiply attention cost.
        int[] padded = _tokenizer.EncodeChat(prompt, includeThinkBlock: false);
        int[] promptTokens = TrimRightPad(padded, Qwen3Tokenizer.BosTokenId);

        // TODO(E-IMG-4): img2img/inpaint, LoRA, ControlNet, IP-Adapter, refiner, regional prompting and
        // ImageRequest.Components overrides are deferred — text-to-image only.
        TextToImageRequest inner = new TextToImageRequest
        {
            Prompt = prompt,
            NegativePrompt = "",
            Width = snappedW,
            Height = snappedH,
            Steps = preset.NumSteps,
            CfgScale = 7.0f,
            Seed = request.Seed < 0 ? null : (int?)(int)(request.Seed & 0x7FFFFFFF),
        };

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
        };

        (byte[] rgb, int outW, int outH, int usedSeed) = _pipeline.GenerateFromTokens(promptTokens, inner, preset, bridge);

        return new ImageResult
        {
            Rgb = rgb,
            Width = outW,
            Height = outH,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "ideogram4",
                ["size"] = $"{outW}x{outH}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = preset.NumSteps.ToString(CultureInfo.InvariantCulture),
                ["preset"] = preset.Name,
            },
        };
    }

    /// <summary>Strips the trailing run of <paramref name="padId"/> tokens <see cref="Qwen3Tokenizer.EncodeChat"/> right-pads with, keeping everything up to and including the last real token.</summary>
    private static int[] TrimRightPad(int[] tokens, int padId)
    {
        int end = tokens.Length;
        while (end > 1 && tokens[end - 1] == padId)
        {
            end--;
        }
        return end == tokens.Length ? tokens : tokens[..end];
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _tokenizer.Dispose();
        _textEncoder.Dispose();
        _conditional.Dispose();
        _unconditional.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
