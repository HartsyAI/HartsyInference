using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Z-Image pipeline driven against the native <see cref="ImageRequest"/>. Owns the Qwen3-4B encoder + tokenizer (the text-encoder forward lives outside <see cref="ZImagePipeline"/>): it encodes the prompt (and, for CFG, the negative) into caption embeddings, then runs <see cref="ZImagePipeline.GenerateFromEmbeddings"/>. Mirrors the SwarmUI backend's <c>ZImageLoader.Generate</c> drive path. Z-Image's non-standard CFG combine and mandatory velocity negation are encoded inside the pipeline; this just calls it with the right cfg + embeddings.</summary>
public sealed unsafe class ZImageRecipePipeline : IRecipePipeline
{
    /// <summary>Qwen3 right-pads EncodeChat output with BosTokenId (151643); the real length ends at the first such pad.</summary>
    private const int Qwen3PadTokenId = 151643;

    private readonly ZImagePipeline _pipeline;
    private readonly LlamaStyleEncoder _qwen;
    private readonly Qwen3Tokenizer _tokenizer;
    private readonly IBackend _backend;
    private readonly SafeTensorsLoader _checkpointLoader;
    private readonly SafeTensorsLoader _qwenLoader;
    private readonly SafeTensorsLoader _vaeLoader;

    /// <summary>Wraps the constructed Z-Image pipeline plus its text stack, taking ownership of every disposable.</summary>
    public ZImageRecipePipeline(ZImagePipeline pipeline, LlamaStyleEncoder qwen, Qwen3Tokenizer tokenizer, IBackend backend, SafeTensorsLoader checkpointLoader, SafeTensorsLoader qwenLoader, SafeTensorsLoader vaeLoader)
    {
        _pipeline = pipeline;
        _qwen = qwen;
        _tokenizer = tokenizer;
        _backend = backend;
        _checkpointLoader = checkpointLoader;
        _qwenLoader = qwenLoader;
        _vaeLoader = vaeLoader;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? ZImageRecipe.FamilyDefaults.Steps;
        // Z-Image CFG: <=0 (or 1) means Turbo (no CFG, single forward). >1 is Base and requires a negative encode.
        float cfg = request.CfgScale ?? ZImageRecipe.FamilyDefaults.CfgScale;
        bool needNegative = cfg > 1.0f;
        int penultimateIdx = _qwen.NumLayers - 1;

        // Bulk-upload the Qwen3 weights, encode, then free them — their ~8 GB is the headroom the VAE full-res decode needs.
        _backend.PreloadWeights(_qwen.EnumerateWeights());

        int[] tokenIds = _tokenizer.EncodeChat(prompt);
        int realLen = ComputeRealLength(tokenIds);
        Tensor encodedFull = _qwen.EncodeMultiLayer(_backend, new[] { tokenIds }, new[] { penultimateIdx });
        Tensor positiveEmbeddings = SliceFirstSeqF32(encodedFull, realLen);
        encodedFull.Dispose();

        Tensor? negativeEmbeddings = null;
        if (needNegative)
        {
            // Encode even an empty negative — the reference passes "" through the encoder, yielding the short but
            // valid unconditional embedding CFG needs.
            int[] negTokens = _tokenizer.EncodeChat(negative);
            int negRealLen = ComputeRealLength(negTokens);
            Tensor negEncodedFull = _qwen.EncodeMultiLayer(_backend, new[] { negTokens }, new[] { penultimateIdx });
            negativeEmbeddings = SliceFirstSeqF32(negEncodedFull, negRealLen);
            negEncodedFull.Dispose();
        }

        // Regional/object prompt parts, encoded via the SAME Qwen3 encoder + penultimate layer as the base
        // prompt above — must happen before FreeWeights below, while the encoder is still resident.
        (int reqW, int reqH) = RecipeRequestMapper.Size(request);
        RegionalPlan? regionalPlan = RegionalPromptResolver.HasRegionParts(prompt)
            ? RegionalPromptResolver.Resolve(prompt, positiveEmbeddings, reqW, reqH, steps, encodeRegion: text =>
            {
                int[] regionTokens = _tokenizer.EncodeChat(text);
                int regionRealLen = ComputeRealLength(regionTokens);
                Tensor regionEncodedFull = _qwen.EncodeMultiLayer(_backend, new[] { regionTokens }, new[] { penultimateIdx });
                Tensor regionEmbeddings = SliceFirstSeqF32(regionEncodedFull, regionRealLen);
                regionEncodedFull.Dispose();
                return regionEmbeddings;
            })
            : null;

        _backend.FreeWeights(_qwen.EnumerateWeights());

        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        using Img2ImgResolver.Img2ImgSpec? img2img = RecipeImg2ImgBinder.Resolve(request, reqWidth, reqHeight);
        TextToImageRequest inner = RecipeImg2ImgBinder.Apply(
            new TextToImageRequest
            {
                Prompt = prompt,
                NegativePrompt = negative,
                Width = request.Width,
                Height = request.Height,
                Steps = steps,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
            },
            img2img);

        try
        {
            Action<GenerationProgress> bridge = p =>
            {
                cancel.ThrowIfCancellationRequested();
                progress?.Report(new StepPreview { Step = p.Step, TotalSteps = steps });
            };

            (byte[] rgb, int width, int height, int usedSeed) = _pipeline.GenerateFromEmbeddings(
                positiveEmbeddings,
                inner,
                cfgScale: cfg,
                negativeCaptionEmbeddings: negativeEmbeddings,
                onProgress: bridge,
                regionalPlan: regionalPlan);

            return new ImageResult
            {
                Rgb = rgb,
                Width = width,
                Height = height,
                Seed = usedSeed,
                Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["arch"] = "zimage",
                    ["size"] = $"{width}x{height}",
                    ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                    ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                    ["cfg"] = cfg.ToString(CultureInfo.InvariantCulture),
                },
            };
        }
        finally
        {
            RegionalPromptResolver.DisposeRegions(regionalPlan);
            positiveEmbeddings.Dispose();
            negativeEmbeddings?.Dispose();
        }
    }

    /// <summary>The real token count: Qwen3's EncodeChat right-pads with <see cref="Qwen3PadTokenId"/>, so the length is the index of the first pad token (or the full array when there is none).</summary>
    private static int ComputeRealLength(int[] tokenIds)
    {
        for (int i = 0; i < tokenIds.Length; i++)
        {
            if (tokenIds[i] == Qwen3PadTokenId)
            {
                return i;
            }
        }
        return tokenIds.Length;
    }

    /// <summary>Slices a [batch, fullLen, hidden] F32 tensor down to [batch, realLen, hidden], copying the leading real tokens of each batch row (drops the Qwen3 right-padding before the transformer sees the caption embeddings).</summary>
    private static Tensor SliceFirstSeqF32(Tensor source, int realLen)
    {
        if (source.Shape.Rank != 3)
        {
            throw new ArgumentException($"Expected 3D tensor, got rank {source.Shape.Rank}.");
        }
        if (source.DType != DType.F32)
        {
            throw new ArgumentException($"SliceFirstSeqF32 expects F32, got {source.DType}.");
        }
        long batch = source.Shape[0];
        long fullLen = source.Shape[1];
        long hidden = source.Shape[2];
        if (realLen <= 0 || realLen > fullLen)
        {
            throw new ArgumentOutOfRangeException(nameof(realLen), $"realLen {realLen} out of range [1..{fullLen}].");
        }
        TensorShape outShape = new TensorShape(batch, realLen, hidden);
        Tensor result = new Tensor(outShape, source.DType);
        long elemSize = source.DType.SizeInBytes;
        long fullRowBytes = fullLen * hidden * elemSize;
        long sliceRowBytes = realLen * hidden * elemSize;
        byte* src = (byte*)source.DataPointer;
        byte* dst = (byte*)result.DataPointer;
        for (long b = 0; b < batch; b++)
        {
            Buffer.MemoryCopy(src + b * fullRowBytes, dst + b * sliceRowBytes, sliceRowBytes, sliceRowBytes);
        }
        return result;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _qwen.Dispose();
        _tokenizer.Dispose();
        _checkpointLoader.Dispose();
        _qwenLoader.Dispose();
        _vaeLoader.Dispose();
    }
}
