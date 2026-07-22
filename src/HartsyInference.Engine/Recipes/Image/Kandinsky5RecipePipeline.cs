using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Kandinsky 5 pipeline driven against the native <see cref="ImageRequest"/>. <see cref="Kandinsky5Pipeline"/> takes only pre-computed embeddings, so this owns the dual text stack: it wraps the prompt in Kandinsky's fixed "promt engineer" ChatML template, runs Qwen2.5-VL-7B for the last hidden state and drops the template prefix, frees those weights, then takes the CLIP-L pooled embedding — the two inputs the reference <c>encode_prompt</c> produces. Ported from the diffusers reference, not from a SwarmUI loader (none exists); UNVERIFIED against real weights.</summary>
public sealed unsafe class Kandinsky5RecipePipeline : IRecipePipeline
{
    private readonly Kandinsky5Pipeline _pipeline;
    private readonly LlamaStyleEncoder _qwen;
    private readonly ClipTextEncoder _clipL;
    private readonly Qwen2Tokenizer _qwenTokenizer;
    private readonly ClipTokenizer _clipTokenizer;
    private readonly IBackend _backend;
    private readonly Kandinsky5Transformer _transformer;
    private readonly List<SafeTensorsLoader> _loaders;

    /// <summary>Wraps the constructed Kandinsky 5 pipeline plus its dual text stack, taking ownership of every disposable.</summary>
    public Kandinsky5RecipePipeline(Kandinsky5Pipeline pipeline, LlamaStyleEncoder qwen, ClipTextEncoder clipL, Qwen2Tokenizer qwenTokenizer, ClipTokenizer clipTokenizer, IBackend backend, Kandinsky5Transformer transformer, List<SafeTensorsLoader> loaders)
    {
        _pipeline = pipeline;
        _qwen = qwen;
        _clipL = clipL;
        _qwenTokenizer = qwenTokenizer;
        _clipTokenizer = clipTokenizer;
        _backend = backend;
        _transformer = transformer;
        _loaders = loaders;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? Kandinsky5Recipe.FamilyDefaults.Steps;
        float cfg = request.CfgScale ?? Kandinsky5Recipe.FamilyDefaults.CfgScale;
        bool useCfg = cfg > 1.0f;

        // TODO(E-IMG-4/5): img2img/inpaint, LoRA, ControlNet, IP-Adapter, refiner, regional prompting and
        // ImageRequest.Components overrides are deferred — text-to-image only.
        Tensor? qwenEmbeds = null;
        Tensor? negQwenEmbeds = null;
        Tensor? clipPooled = null;
        Tensor? negClipPooled = null;
        try
        {
            // Qwen2.5-VL is ~16 GB: bulk-upload, encode both branches, then free before the denoise loop preloads
            // the transformer (the ZImage/OmniGen2 preload -> encode -> free pattern).
            _backend.PreloadWeights(_qwen.EnumerateWeights());
            qwenEmbeds = Kandinsky5TextEncoding.EncodeQwen(_backend, _qwen, _qwenTokenizer, prompt);
            if (useCfg)
            {
                negQwenEmbeds = Kandinsky5TextEncoding.EncodeQwen(_backend, _qwen, _qwenTokenizer, negative);
            }
            _backend.FreeWeights(_qwen.EnumerateWeights());

            clipPooled = Kandinsky5TextEncoding.EncodeClipPooled(_backend, _clipL, _clipTokenizer, prompt);
            if (useCfg)
            {
                negClipPooled = Kandinsky5TextEncoding.EncodeClipPooled(_backend, _clipL, _clipTokenizer, negative);
            }

            TextToImageRequest inner = new TextToImageRequest
            {
                Prompt = prompt,
                NegativePrompt = negative,
                Width = request.Width,
                Height = request.Height,
                Steps = steps,
                CfgScale = cfg,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
            };

            Action<GenerationProgress> bridge = p =>
            {
                cancel.ThrowIfCancellationRequested();
                progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
            };

            (byte[] rgb, int outW, int outH, int usedSeed) = _pipeline.GenerateFromEmbeddings(
                qwenEmbeds, clipPooled, negQwenEmbeds, negClipPooled, inner, bridge);

            return new ImageResult
            {
                Rgb = rgb,
                Width = outW,
                Height = outH,
                Seed = usedSeed,
                Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["arch"] = "kandinsky5",
                    ["size"] = $"{outW}x{outH}",
                    ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                    ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                    ["cfg"] = cfg.ToString(CultureInfo.InvariantCulture),
                },
            };
        }
        finally
        {
            qwenEmbeds?.Dispose();
            negQwenEmbeds?.Dispose();
            clipPooled?.Dispose();
            negClipPooled?.Dispose();
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _qwen.Dispose();
        _qwenTokenizer.Dispose();
        _clipTokenizer.Dispose();
        _transformer.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
