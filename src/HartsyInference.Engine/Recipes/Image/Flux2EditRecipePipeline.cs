using System.Globalization;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Tokenizers;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Open Vision Banana pipeline driven against the native <see cref="ImageRequest"/>. The prompt names the
/// task and reaches the text encoder <b>verbatim</b> (no emphasis, prompt-tag or region parsing: the task prompts carry
/// parentheses, brackets and braces that grammar would rewrite). One photo, the init image in <see cref="Img2ImgMode.Reference"/>
/// or the first of <see cref="ImageRequest.ReferenceImages"/>, is the reference the answer is painted for; without one it is
/// plain text-to-image. Guidance is true CFG against the empty prompt, as diffusers' Klein pipeline runs it. Owns every
/// disposable it is handed.</summary>
public sealed class Flux2EditRecipePipeline(Flux2EditPipeline pipeline, Qwen3Tokenizer tokenizer, string variantId,
    LlamaStyleEncoder encoder, List<IDisposable> loaders, IDisposable? checkpoint = null, MergedLoraStack? loraStack = null) : IRecipePipeline
{
    /// <summary>One reference photo. Only the count is a policy here: the size and crop follow diffusers' Klein pipeline in
    /// <see cref="ReferenceSize"/> and <see cref="ToReferenceTensor"/>, not <see cref="ReferenceImageResolver"/>'s rounding.</summary>
    public static ReferenceImagePolicy ReferencePolicy { get; } = new()
    {
        MaxReferences = 1,
        LatentMultiple = 16,
        Overflow = ReferenceOverflow.Refuse,
    };

    /// <summary>The pixel area diffusers caps an input photo at before the VAE.</summary>
    public const int ReferenceArea = 1024 * 1024;

    /// <summary>The size diffusers' <c>Flux2KleinPipeline</c> feeds the VAE: a photo above one megapixel is scaled to that area
    /// (<c>int(side * sqrt(area / (w*h)))</c>, truncating), then each side is floored to a multiple of 16.</summary>
    public static (int Width, int Height) ReferenceSize(int width, int height)
    {
        if ((long)width * height > ReferenceArea)
        {
            double scale = Math.Sqrt(ReferenceArea / ((double)width * height));
            (width, height) = ((int)(width * scale), (int)(height * scale));
        }
        return (width / 16 * 16, height / 16 * 16);
    }

    /// <summary>The photo as the VAE reads it: area-capped with Lanczos, then cover-cropped to the 16-multiple size, RGB in [-1, 1].</summary>
    public static Tensor ToReferenceTensor(ImageData photo)
    {
        ArgumentNullException.ThrowIfNull(photo);
        (int width, int height) = ReferenceSize(photo.Width, photo.Height);
        if (width < 16 || height < 16)
            throw new ArgumentException($"{Family} needs a photo of at least 16 pixels a side; got {photo.Width}x{photo.Height}.", nameof(photo));
        ImageData image = photo;
        if ((long)photo.Width * photo.Height > ReferenceArea)
        {
            double scale = Math.Sqrt(ReferenceArea / ((double)photo.Width * photo.Height));
            int scaledW = (int)(photo.Width * scale), scaledH = (int)(photo.Height * scale);
            image = new ImageData { Rgb = FeatureImaging.ResizeRgb24Lanczos(photo, scaledW, scaledH), Width = scaledW, Height = scaledH };
        }
        image = FeatureImaging.CoverCrop(image, width, height);
        return FeatureImaging.RgbToTensorMinusOneOne(image.Rgb, width, height);
    }

    private const string Family = "Open Vision Banana";

    private readonly Flux2EditPipeline _pipeline = pipeline;
    private readonly Qwen3Tokenizer _tokenizer = tokenizer;
    private readonly string _variantId = variantId;
    private readonly LlamaStyleEncoder _encoder = encoder;
    private readonly List<IDisposable> _loaders = loaders;
    private readonly IDisposable? _checkpoint = checkpoint;
    private readonly MergedLoraStack? _loraStack = loraStack;
    private int _disposed;

    /// <summary>The prompt's token ids exactly as Qwen3's chat template renders them (thinking off, as diffusers'
    /// <c>enable_thinking=False</c>), right-padded to 512, and the real length the encoder masks to. No weighting grammar runs.</summary>
    public IrisZPrompt Tokenize(string prompt)
    {
        (int[] ids, int realLength) = _tokenizer.EncodeChatWithLength(prompt);
        return new IrisZPrompt(ids, realLength < ids.Length ? realLength : 0);
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, ImageEffectiveSettings settings, IProgress<StepPreview>? progress,
        CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        if (!string.IsNullOrWhiteSpace(request.NegativePrompt))
        {
            throw new NotSupportedException(
                $"{Family} guides against the empty prompt, as it was trained and as diffusers runs it; a negative prompt is not part of the model.");
        }
        if (request.Img2Img is { Mode: Img2ImgMode.Denoise })
        {
            throw new NotSupportedException(
                $"{Family} has no strength-based image-to-image. Pass the photo as a reference ({nameof(Img2ImgMode)}.{nameof(Img2ImgMode.Reference)} "
                + $"or {nameof(ImageRequest.ReferenceImages)}); it paints the answer for that photo.");
        }
        if (string.IsNullOrWhiteSpace(request.Prompt))
            throw new ArgumentException($"{Family} needs a task prompt.", nameof(request));

        IReadOnlyList<ImageData> ordered = ReferenceImageResolver.Collect(
            request.Img2Img is { Mode: Img2ImgMode.Reference } ? request.Img2Img.InitImage : null,
            request.ReferenceImages, ReferencePolicy, Family);
        List<Tensor> references = new(ordered.Count);
        try
        {
            foreach (ImageData photo in ordered) references.Add(ToReferenceTensor(photo));
            (int width, int height) = references.Count == 0
                ? (settings.Width / 16 * 16, settings.Height / 16 * 16)
                : ((int)references[0].Shape[3], (int)references[0].Shape[2]);
            IrisZPrompt prompt = Tokenize(request.Prompt);
            IrisZPrompt? negative = CfgHelper.IsGuidanceActive(settings.CfgScale) ? Tokenize("") : null;
            TextToImageRequest inner = new()
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
            (byte[] rgb, int outW, int outH, int usedSeed) = _pipeline.Edit(prompt, negative, references, inner,
                RecipeProgressAdapter.Create(progress, cancel));
            return new ImageResult
            {
                Rgb = rgb,
                Width = outW,
                Height = outH,
                Seed = usedSeed,
                Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["arch"] = "flux2",
                    ["variant"] = _variantId,
                    ["size"] = $"{outW}x{outH}",
                    ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                    ["steps"] = settings.Steps.ToString(CultureInfo.InvariantCulture),
                    ["cfg"] = settings.CfgScale.ToString(CultureInfo.InvariantCulture),
                    ["references"] = references.Count.ToString(CultureInfo.InvariantCulture),
                },
            };
        }
        finally
        {
            foreach (Tensor reference in references) reference.Dispose();
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        // Every handle is released even if one throws; the LoRA stack goes last, it owns the merged weights.
        new CompositeDisposable([_pipeline, _tokenizer, _encoder, .. _loaders, _checkpoint, _loraStack]).Dispose();
    }
}
