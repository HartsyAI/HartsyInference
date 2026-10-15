using System.Globalization;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Memory;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed FLUX.3 Klein pipeline driven against the native <see cref="ImageRequest"/>: renders the
/// Qwen3-VL chat template (no think block) right-padded to a multiple of 80, turns regional prompt parts into the
/// model's layout table, sizes reference images to about one megapixel in <c>&lt;ref_image_N&gt;</c> order (the init
/// image first when <see cref="Img2ImgMode.Reference"/> asks for it), and calls <see cref="Flux3Pipeline.Generate"/>.
/// Owns every disposable the recipe created.</summary>
public sealed class Flux3RecipePipeline(Flux3Pipeline pipeline, Qwen3Tokenizer tokenizer, Flux3Config config,
    ResolvedModelVariant variant, IReadOnlyList<IDisposable> owned) : IRecipePipeline
{
    /// <summary>ComfyUI <c>ReferenceMultiLatent</c>'s default reference size: one megapixel.</summary>
    private const int ReferenceArea = 1024 * 1024;

    private readonly Flux3Pipeline _pipeline = pipeline;
    private readonly Qwen3Tokenizer _tokenizer = tokenizer;
    private readonly Flux3Config _config = config;
    private readonly ResolvedModelVariant _variant = variant;
    private readonly IReadOnlyList<IDisposable> _owned = owned;
    private int _disposed;

    /// <inheritdoc/>
    public ImageDefaults? VariantDefaults => _config.FlowMap ? Flux3Recipe.DistilledDefaults : Flux3Recipe.FamilyDefaults;

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancel.ThrowIfCancellationRequested();
        ImageDefaults defaults = VariantDefaults!;
        int steps = request.Steps ?? defaults.Steps;
        float cfg = _config.FlowMap ? 1f : request.CfgScale ?? defaults.CfgScale;
        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        int width = Math.Max(_config.VaeDownscaleFactor, reqWidth / _config.VaeDownscaleFactor * _config.VaeDownscaleFactor);
        int height = Math.Max(_config.VaeDownscaleFactor, reqHeight / _config.VaeDownscaleFactor * _config.VaeDownscaleFactor);

        Img2ImgMode mode = request.Img2Img?.Mode ?? Img2ImgMode.Auto;
        bool initIsReference = mode == Img2ImgMode.Reference && request.Img2Img is not null;
        List<ImageData> references = CollectReferences(request, initIsReference);

        (string prompt, string? layoutTable) = LayoutPromptBuilder.Split(PromptTagFlattening.Flatten(request.Prompt));
        Flux3PromptTokens promptTokens = Tokenize(prompt, layoutTable);
        Flux3PromptTokens? negativeTokens = cfg > 1f
            ? Tokenize(PromptTagFlattening.Flatten(request.NegativePrompt ?? ""), null) : null;

        using Img2ImgResolver.Img2ImgSpec? img2img = initIsReference ? null : RecipeImg2ImgBinder.Resolve(request, width, height);
        TextToImageRequest inner = RecipeImg2ImgBinder.Apply(
            new TextToImageRequest
            {
                SigmaShift = RecipeRequestMapper.MapFluxMaxShift(request.SigmaShift),
                SeamlessTiling = request.SeamlessTiling,
                VariationSeed = request.VariationSeed?.Seed ?? -1,
                VariationSeedStrength = request.VariationSeed?.Strength ?? 0,
                Prompt = request.Prompt,
                NegativePrompt = request.NegativePrompt ?? "",
                Width = width,
                Height = height,
                Steps = steps,
                CfgScale = cfg,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
                Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
            },
            img2img);

        List<Tensor> referenceTensors = new List<Tensor>(references.Count);
        try
        {
            foreach (ImageData reference in references)
                referenceTensors.Add(ToReferenceTensor(reference));
            Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel, totalSteps: steps);
            (byte[] rgb, int outW, int outH, int seed) = _pipeline.Generate(promptTokens, negativeTokens, inner, referenceTensors, bridge);
            return new ImageResult
            {
                Rgb = rgb,
                Width = outW,
                Height = outH,
                Seed = seed,
                Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["arch"] = "flux3",
                    ["variant"] = _variant.Variant.Id,
                    ["size"] = string.Create(CultureInfo.InvariantCulture, $"{outW}x{outH}"),
                    ["seed"] = seed.ToString(CultureInfo.InvariantCulture),
                    ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                    ["references"] = referenceTensors.Count.ToString(CultureInfo.InvariantCulture),
                },
            };
        }
        finally
        {
            foreach (Tensor tensor in referenceTensors) tensor.Dispose();
        }
    }

    /// <summary>The init image (when it is a reference) followed by every reference, refusing more than the model reads.</summary>
    private List<ImageData> CollectReferences(ImageRequest request, bool initIsReference)
    {
        List<ImageData> references = new List<ImageData>(_config.MaxReferenceImages);
        if (initIsReference)
            references.Add(request.Img2Img!.InitImage);
        if (request.ReferenceImages is not null)
        {
            foreach (ImageData reference in request.ReferenceImages)
            {
                if (reference is not null) references.Add(reference);
            }
        }
        if (references.Count > _config.MaxReferenceImages)
        {
            throw new InvalidOperationException(
                $"FLUX.3 conditions on at most {_config.MaxReferenceImages} reference images; {references.Count} were supplied.");
        }
        return references;
    }

    /// <summary>The Qwen3-VL chat template (no think block, no system turn) around the prompt, the layout table appended
    /// as unweighted text, right-padded to a multiple of 80. The valid length ends at the first pad id, where ComfyUI's
    /// attention mask ends.</summary>
    private Flux3PromptTokens Tokenize(string prompt, string? layoutTable)
    {
        List<WeightedSpan> spans = [.. PromptWeighting.Parse(prompt)];
        if (layoutTable is not null)
            spans.Add(new WeightedSpan(string.Concat(" ", layoutTable), 1f));
        WeightedTokenSequence sequence;
        if (!PromptWeighting.HasWeights(spans))
        {
            int[] ids = _tokenizer.EncodeChatUnpadded(PromptWeighting.Join(spans), systemPrompt: null, includeThinkBlock: false);
            float[] ones = new float[ids.Length];
            Array.Fill(ones, 1f);
            sequence = new WeightedTokenSequence(ids, ones) { UniformWeight = 1f };
        }
        else
        {
            (int[] prefix, int[] suffix) = _tokenizer.ChatTemplateIds(includeThinkBlock: false);
            sequence = WeightedTokenBuilder.Build(spans, _tokenizer.EncodeRaw, prefix, suffix);
        }
        int valid = Array.IndexOf(sequence.Tokens, _tokenizer.PadTokenId);
        valid = valid < 0 ? sequence.Tokens.Length : valid;
        if (valid == 0)
            throw new InvalidOperationException("FLUX.3 prompt tokenized to nothing before the first pad token.");
        return new Flux3PromptTokens(sequence.PadToMultiple(_config.TextPadMultiple, _tokenizer.PadTokenId), valid);
    }

    /// <summary>ReferenceMultiLatent's sizing: aspect-preserving rescale to one megapixel, then the VAE's centre crop to
    /// multiples of 16, as RGB in [-1, 1].</summary>
    private Tensor ToReferenceTensor(ImageData reference)
    {
        (int scaledW, int scaledH) = FeatureImaging.ScaleToArea(reference.Width, reference.Height, ReferenceArea, multiple: 1);
        ImageData scaled = new ImageData { Rgb = FeatureImaging.ResizeRgb24(reference, scaledW, scaledH), Width = scaledW, Height = scaledH };
        int unit = _config.VaeDownscaleFactor;
        int cropW = scaledW / unit * unit;
        int cropH = scaledH / unit * unit;
        if (cropW == 0 || cropH == 0)
            throw new InvalidOperationException($"FLUX.3 reference {reference.Width}x{reference.Height} is too small to encode.");
        ImageData cropped = cropW == scaledW && cropH == scaledH
            ? scaled
            : FeatureImaging.CropRgb24(scaled, (scaledW % unit) / 2, (scaledH % unit) / 2, cropW, cropH);
        Logs.Debug($"[Flux3] reference {reference.Width}x{reference.Height} → {cropW}x{cropH}");
        return FeatureImaging.RgbToTensorMinusOneOne(cropped.Rgb, cropW, cropH);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        new CompositeDisposable([.. Enumerable.Reverse(_owned)]).Dispose();
    }
}
