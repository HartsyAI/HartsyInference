using System.Globalization;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Tokenizers;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Iris-Z pipeline driven against the native <see cref="ImageRequest"/>. The task comes from
/// <see cref="RequestExtras.IrisZTask"/>: <c>image</c> and <c>rgbd</c> generate from the prompt, <c>depth</c> estimates
/// the init image's depth, and <c>paint</c> generates from an init image that is a depth map. Any depth produced is
/// returned in <see cref="ImageResult.Depth"/>. Owns every disposable it is handed.</summary>
public sealed class IrisZRecipePipeline(IrisZPipeline pipeline, Flux2PromptTokenizer tokenizer, float skyCeiling, string variantId,
    Qwen3Tokenizer? qwenTokenizer, ErnieTokenizer? mistralTokenizer, LlamaStyleEncoder encoder, List<IDisposable> loaders,
    IDisposable? checkpoint = null, MergedLoraStack? loraStack = null) : IRecipePipeline
{
    /// <summary>Below this many pixels a photo's depth is estimated at about 512², above at about 1024².</summary>
    private const int SmallImagePixels = 500_000;

    private readonly IrisZPipeline _pipeline = pipeline;
    private readonly Flux2PromptTokenizer _tokenizer = tokenizer;
    private readonly float _skyCeiling = skyCeiling;
    private readonly string _variantId = variantId;
    private readonly Qwen3Tokenizer? _qwenTokenizer = qwenTokenizer;
    private readonly ErnieTokenizer? _mistralTokenizer = mistralTokenizer;
    private readonly LlamaStyleEncoder _encoder = encoder;
    private readonly List<IDisposable> _loaders = loaders;
    private readonly IDisposable? _checkpoint = checkpoint;
    private readonly MergedLoraStack? _loraStack = loraStack;
    private int _disposed;

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, ImageEffectiveSettings settings, IProgress<StepPreview>? progress,
        CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        ImageData? init = request.Img2Img?.InitImage;
        string task = (RequestExtras.String(request.Extra, RequestExtras.IrisZTask) ?? (init is null ? "image" : "depth"))
            .Trim().ToLowerInvariant();
        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel);
        switch (task)
        {
            case "image":
            case "rgbd":
                if (init is not null)
                {
                    throw new NotSupportedException(
                        $"The Iris-Z '{task}' task generates from the prompt alone and takes no init image; image-to-image is not "
                        + $"part of the model. Use {RequestExtras.IrisZTask}=depth to estimate the image's depth, or =paint to "
                        + "generate from a depth map.");
                }
                return TextToImage(request, settings, task == "rgbd", bridge);
            case "depth":
                return ImageToDepth(request, init ?? throw MissingInit(task, "the photo to estimate depth for"), bridge);
            case "paint":
                return DepthToImage(request, settings, init ?? throw MissingInit(task, "the depth map to generate from"), bridge);
            default:
                throw new NotSupportedException(
                    $"Unknown Iris-Z task '{task}' in {RequestExtras.IrisZTask}; expected image, rgbd, depth or paint.");
        }
    }

    private ImageResult TextToImage(ImageRequest request, ImageEffectiveSettings settings, bool withDepth,
        Action<GenerationProgress> bridge)
    {
        int width = settings.Width / 16 * 16, height = settings.Height / 16 * 16;
        IrisZDepthOutput depth = !withDepth ? IrisZDepthOutput.None
            : RequestExtras.Bool(request.Extra, RequestExtras.IrisZJointDepth, false) ? IrisZDepthOutput.Joint : IrisZDepthOutput.Estimated;
        (IrisZPrompt? prompt, IrisZPrompt? empty) = Prompts(request.Prompt);
        IrisZImage image = _pipeline.TextToImage(prompt, empty, Inner(request, settings, width, height), depth, bridge);
        return Result(image.Rgb, image.LogDepth, image.Width, image.Height, image.Seed, withDepth ? "rgbd" : "image", settings.Steps);
    }

    private ImageResult DepthToImage(ImageRequest request, ImageEffectiveSettings settings, ImageData depthMap,
        Action<GenerationProgress> bridge)
    {
        string format = (RequestExtras.String(request.Extra, RequestExtras.IrisZInputDepthFormat) ?? "depth").Trim().ToLowerInvariant();
        if (format is not ("depth" or "disparity"))
        {
            throw new NotSupportedException(
                $"Iris-Z reads a depth map as 'depth' (bright is far) or 'disparity' (bright is near); '{format}' "
                + "(the reference's colour-coded maps included) is not supported.");
        }
        int resolution = Resolution(request, fallback: 1024);
        (int width, int height) = IrisZDepth.EnlargedSize(depthMap.Width, depthMap.Height, resolution);
        float[] gray = IrisZDepth.ResizeBilinear(Luma(depthMap), depthMap.Width, depthMap.Height, width, height);
        float[] encoded = IrisZDepth.EncodeLogDepth(IrisZDepth.LogDepthFromGray(gray, format == "disparity"), [], _skyCeiling);
        (IrisZPrompt? prompt, IrisZPrompt? empty) = Prompts(request.Prompt);
        IrisZImage image = _pipeline.DepthToImage(encoded, prompt, empty, Inner(request, settings, width, height), bridge);
        return Result(image.Rgb, null, image.Width, image.Height, image.Seed, "paint", settings.Steps);
    }

    private ImageResult ImageToDepth(ImageRequest request, ImageData photo, Action<GenerationProgress> bridge)
    {
        // Depth runs its own four-step Euler loop with clean-estimate averaging, which no named sampler reproduces.
        FlowMatchSampling.ThrowIfSamplerSelected(SamplingParamResolver.ResolveSchedulerName(request), "Iris-Z image to depth");
        int steps = (int)RequestExtras.Number(request.Extra, RequestExtras.IrisZDepthSteps, IrisZPipeline.DepthSteps);
        int seedCount = (int)RequestExtras.Number(request.Extra, RequestExtras.IrisZDepthSeeds, 1);
        if (steps < 1 || seedCount < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(request),
                $"{RequestExtras.IrisZDepthSteps} and {RequestExtras.IrisZDepthSeeds} must be at least 1; got {steps} and {seedCount}.");
        }
        int resolution = Resolution(request, fallback: (long)photo.Width * photo.Height < SmallImagePixels ? 512 : 1024);
        (int width, int height) = IrisZDepth.ChooseResolution(photo.Width, photo.Height, resolution);
        int seed = RecipeRequestMapper.MapSeed(request.Seed) ?? SeedGenerator.RandomSeed();
        int[] seeds = new int[seedCount];
        for (int i = 0; i < seedCount; i++) seeds[i] = unchecked(seed + i);
        float[] logDepth;
        using (Tensor image = FeatureImaging.RgbToTensorMinusOneOne(FeatureImaging.ResizeRgb24Lanczos(photo, width, height), width, height))
        {
            logDepth = _pipeline.ImageToDepth(image, steps, IrisZPipeline.DepthAverageLast, seeds, bridge);
        }
        // Back at the photo's own size: depth bilinear, the sky mask nearest, as the reference resizes them.
        bool[] sky = ResizeNearest(IrisZDepth.SkyMask(logDepth, _skyCeiling), width, height, photo.Width, photo.Height);
        float[] depth = IrisZDepth.ResizeBilinear(logDepth, width, height, photo.Width, photo.Height);
        string format = (RequestExtras.String(request.Extra, RequestExtras.IrisZDepthFormat) ?? "colours").Trim().ToLowerInvariant();
        byte[] rgb = format switch
        {
            "colours" or "colors" => IrisZDepth.Colourize(depth, sky),
            "depth" => IrisZDepth.ToGray(depth, disparity: false),
            "disparity" => IrisZDepth.ToGray(depth, disparity: true),
            _ => throw new NotSupportedException(
                $"Unknown {RequestExtras.IrisZDepthFormat} '{format}'; expected colours, depth or disparity."),
        };
        return Result(rgb, depth, photo.Width, photo.Height, seed, "depth", steps);
    }

    private static TextToImageRequest Inner(ImageRequest request, ImageEffectiveSettings settings, int width, int height) => new()
    {
        Prompt = request.Prompt,
        Width = width,
        Height = height,
        Steps = settings.Steps,
        CfgScale = settings.CfgScale,
        Seed = RecipeRequestMapper.MapSeed(request.Seed),
        SigmaShift = RecipeRequestMapper.MapFluxMaxShift(request.SigmaShift),
        // Resolved rather than read raw, so an unavailable sampler is refused by name before sampling starts.
        Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
    };

    /// <summary>The prompt and the empty prompt, tokenized; both null for a blank prompt, which runs unguided on the
    /// release's empty-prompt tensor and never loads the text encoder.</summary>
    private (IrisZPrompt? Prompt, IrisZPrompt? Empty) Prompts(string? prompt)
    {
        string flat = PromptTagFlattening.Flatten(prompt ?? "");
        if (string.IsNullOrWhiteSpace(flat))
            return (null, null);
        return (Tokenized(flat), Tokenized(""));
    }

    private IrisZPrompt Tokenized(string prompt)
    {
        (WeightedTokenSequence tokens, int validLength) = _tokenizer.TokenizeWithLength(prompt);
        return new IrisZPrompt(tokens.Tokens, validLength < tokens.Tokens.Length ? validLength : 0,
            tokens.IsUniformlyUnweighted ? null : tokens);
    }

    private static int Resolution(ImageRequest request, int fallback)
    {
        int resolution = (int)RequestExtras.Number(request.Extra, RequestExtras.IrisZResolution, fallback);
        if (resolution is not (512 or 1024))
        {
            throw new ArgumentOutOfRangeException(nameof(request), resolution,
                $"{RequestExtras.IrisZResolution} is 512 or 1024, the sizes Iris-Z was trained at.");
        }
        return resolution;
    }

    private ImageResult Result(byte[] rgb, float[]? depth, int width, int height, int seed, string task, int steps) => new()
    {
        Rgb = rgb,
        Depth = depth,
        Width = width,
        Height = height,
        Seed = seed,
        Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["arch"] = "flux2",
            ["variant"] = _variantId,
            [DepthImage.TaskMeta] = task,
            [DepthImage.SkyCeilingMeta] = _skyCeiling.ToString(CultureInfo.InvariantCulture),
            ["size"] = $"{width}x{height}",
            ["seed"] = seed.ToString(CultureInfo.InvariantCulture),
            ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
        },
    };

    private static NotSupportedException MissingInit(string task, string what) =>
        new($"The Iris-Z '{task}' task needs an init image: {what}.");

    /// <summary>ITU-R 601 luma in <c>[0, 1]</c>, the grey conversion the reference applies to a colour depth map.</summary>
    private static float[] Luma(ImageData image)
    {
        float[] gray = new float[image.Width * image.Height];
        ReadOnlySpan<byte> rgb = image.Rgb;
        for (int i = 0; i < gray.Length; i++)
            gray[i] = ((rgb[i * 3] * 19595 + rgb[i * 3 + 1] * 38470 + rgb[i * 3 + 2] * 7471 + 0x8000) >> 16) / 255f;
        return gray;
    }

    private static bool[] ResizeNearest(bool[] mask, int width, int height, int newWidth, int newHeight)
    {
        if (newWidth == width && newHeight == height)
            return mask;
        bool[] resized = new bool[newWidth * newHeight];
        for (int y = 0; y < newHeight; y++)
        {
            int sourceY = Math.Min((int)((y + 0.5) * height / newHeight), height - 1);
            for (int x = 0; x < newWidth; x++)
                resized[y * newWidth + x] = mask[sourceY * width + Math.Min((int)((x + 0.5) * width / newWidth), width - 1)];
        }
        return resized;
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
