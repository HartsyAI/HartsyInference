using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
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

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Zeta-Chroma pipeline driven against the native <see cref="ImageRequest"/>. Owns the Qwen3-4B encoder + tokenizer (the text-encoder forward lives outside <see cref="ZetaChromaPipeline"/>): it encodes the prompt (and, for CFG, the negative) into caption embeddings, then runs <see cref="ZetaChromaPipeline.GenerateFromEmbeddings"/>. Mirrors the SwarmUI backend's <c>ZetaChromaLoader.Generate</c> drive path, with the Z-Image preload→encode→free staging around the ~8 GB encoder.</summary>
public sealed unsafe class ZetaChromaRecipePipeline : IRecipePipeline
{
    /// <summary>Qwen3 right-pads EncodeChat output with BosTokenId (151643); the real length ends at the first such pad.</summary>
    private const int Qwen3PadTokenId = 151643;

    private readonly ZetaChromaPipeline _pipeline;
    private readonly ZetaChromaConfig _config;
    private readonly LlamaStyleEncoder _qwen;
    private readonly Qwen3Tokenizer _tokenizer;
    private readonly IBackend _backend;
    private readonly SafeTensorsLoader _checkpointLoader;
    private readonly SafeTensorsLoader _qwenLoader;

    private readonly MergedLoraStack? _loraStack;

    /// <summary>Wraps the constructed Zeta-Chroma pipeline plus its text stack, taking ownership of every disposable.</summary>
    public ZetaChromaRecipePipeline(ZetaChromaPipeline pipeline, ZetaChromaConfig config, LlamaStyleEncoder qwen, Qwen3Tokenizer tokenizer, IBackend backend, SafeTensorsLoader checkpointLoader, SafeTensorsLoader qwenLoader, MergedLoraStack? loraStack = null)
    {
        _loraStack = loraStack;
        _pipeline = pipeline;
        _config = config;
        _qwen = qwen;
        _tokenizer = tokenizer;
        _backend = backend;
        _checkpointLoader = checkpointLoader;
        _qwenLoader = qwenLoader;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? _config.DefaultSteps;
        float cfg = request.CfgScale ?? _config.DefaultCfgScale;
        int penultimateIdx = _qwen.NumLayers - 1;

        // Bulk-upload the Qwen3 weights, encode, then free them — the same staging Z-Image uses for this encoder.
        _backend.PreloadWeights(_qwen.EnumerateWeights());

        int[] tokenIds = _tokenizer.EncodeChat(prompt);
        int realLen = ComputeRealLength(tokenIds);
        Tensor encodedFull = _qwen.EncodeMultiLayer(_backend, new[] { tokenIds }, new[] { penultimateIdx });
        Tensor positiveEmbeddings = SliceFirstSeqF32(encodedFull, realLen);
        encodedFull.Dispose();

        Tensor? negativeEmbeddings = null;
        if (cfg > 1.0f)
        {
            int[] negTokens = _tokenizer.EncodeChat(negative);
            int negRealLen = ComputeRealLength(negTokens);
            Tensor negEncodedFull = _qwen.EncodeMultiLayer(_backend, new[] { negTokens }, new[] { penultimateIdx });
            negativeEmbeddings = SliceFirstSeqF32(negEncodedFull, negRealLen);
            negEncodedFull.Dispose();
        }

        _backend.FreeWeights(_qwen.EnumerateWeights());

        // Pixel-space: no VAE encoder, the source pixels are noised directly at sigma[startStep].
        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        using Img2ImgResolver.Img2ImgSpec? img2img = RecipeImg2ImgBinder.Resolve(request, reqWidth, reqHeight);
        TextToImageRequest inner = RecipeImg2ImgBinder.Apply(
            new TextToImageRequest
            {
                SeamlessTiling = request.SeamlessTiling,
                    VariationSeed = request.VariationSeed?.Seed ?? -1,
                    VariationSeedStrength = request.VariationSeed?.Strength ?? 0,
                Prompt = prompt,
                NegativePrompt = negative,
                Width = request.Width,
                Height = request.Height,
                Steps = steps,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
                // Routed through the resolver rather than read raw, so an unavailable sampler is refused by name here —
                // before the checkpoint loads — instead of deep inside the pipeline, or silently dropped.
                Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
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
                onProgress: bridge);

            return new ImageResult
            {
                Rgb = rgb,
                Width = width,
                Height = height,
                Seed = usedSeed,
                Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["arch"] = "zeta-chroma",
                    ["size"] = $"{width}x{height}",
                    ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                    ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                    ["cfg"] = cfg.ToString(CultureInfo.InvariantCulture),
                },
            };
        }
        finally
        {
            positiveEmbeddings.Dispose();
            negativeEmbeddings?.Dispose();
        }
    }

    /// <summary>The real token count: the length is the index of the first <see cref="Qwen3PadTokenId"/> (or the full array when there is none).</summary>
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

    /// <summary>Slices a [batch, fullLen, hidden] F32 tensor down to [batch, realLen, hidden], dropping the Qwen3 right-padding before the transformer sees the caption embeddings.</summary>
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
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
