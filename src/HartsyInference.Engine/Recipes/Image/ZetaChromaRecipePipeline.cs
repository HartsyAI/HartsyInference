using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
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

/// <summary>A constructed Zeta-Chroma pipeline driven against the native <see cref="ImageRequest"/>. Owns the Qwen3-4B encoder + tokenizer (the text-encoder forward lives outside <see cref="ZetaChromaPipeline"/>): it encodes the prompt (and, for CFG, the negative) into caption embeddings, then runs <see cref="ZetaChromaPipeline.GenerateFromEmbeddings"/>. Mirrors the SwarmUI backend's <c>ZetaChromaLoader.Generate</c> drive path, with the Z-Image preload→encode→free staging around the ~8 GB encoder. Wraps the constructed Zeta-Chroma pipeline plus its text stack, taking ownership of every disposable.</summary>
public sealed unsafe class ZetaChromaRecipePipeline(ZetaChromaPipeline pipeline, ZetaChromaConfig config,
    LlamaStyleEncoder qwen, Qwen3Tokenizer tokenizer, IBackend backend, IDisposable checkpoint,
    IReadOnlyList<IDisposable> sideModelLoaders, MergedLoraStack? loraStack = null) : IRecipePipeline
{
    /// <summary>Qwen3 right-pads EncodeChat output with BosTokenId (151643); the real length ends at the first such pad.</summary>
    private const int Qwen3PadTokenId = 151643;

    private readonly ZetaChromaPipeline _pipeline = pipeline;
    private readonly ZetaChromaConfig _config = config;
    private readonly LlamaStyleEncoder _qwen = qwen;
    private readonly Qwen3Tokenizer _tokenizer = tokenizer;
    private readonly IBackend _backend = backend;
    private readonly IDisposable _checkpoint = checkpoint;
    private readonly IReadOnlyList<IDisposable> _sideModelLoaders = sideModelLoaders;

    private readonly MergedLoraStack? _loraStack = loraStack;

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

        Tensor positiveEmbeddings = EncodeWeighted(prompt, penultimateIdx);
        Tensor? negativeEmbeddings = cfg > 1.0f ? EncodeWeighted(negative, penultimateIdx) : null;

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
            Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel, totalSteps: steps);

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

    /// <summary>Encodes one prompt and applies its per-token weights. Nothing caches this conditioning, so the
    /// scale replaces the tensor outright rather than needing a per-request copy.</summary>
    /// <remarks>An unweighted prompt keeps <see cref="Qwen3Tokenizer.EncodeChat"/> verbatim, so its ids and its
    /// encoder shape are exactly what they were before weighting existed. The weighted build is padded back to
    /// the same window for the reason given at the call site.</remarks>
    private Tensor EncodeWeighted(string prompt, int layerIndex)
    {
        (int[] prefix, int[] suffix) = _tokenizer.ChatTemplateIds();
        WeightedTokenSequence sequence = TemplatedPromptTokens.Build(
            PromptTagFlattening.Flatten(prompt), t => _tokenizer.EncodeChat(t), _tokenizer.EncodeRaw, prefix, suffix)
            .Truncate(_tokenizer.MaxLength);
        // EncodeChat right-pads to the 256-token window; the weighted build produces the real tokens alone. Pad
        // it back so the encoder sees the SAME shape either way. Causal attention means a real token's hidden
        // state cannot depend on padding that follows it, so the short form would be correct in exact arithmetic
        // — but F16 attention at a different sequence length is free to differ in the last bit, and that would
        // make a weighted generation differ from its baseline for two reasons instead of one.
        int realLen = ComputeRealLength(sequence.Tokens);
        int[] tokens = PadToWindow(sequence.Tokens, _tokenizer.MaxLength);
        Tensor encodedFull = _qwen.EncodeMultiLayer(_backend, new[] { tokens }, new[] { layerIndex });
        Tensor embeddings = SliceFirstSeqF32(encodedFull, realLen);
        encodedFull.Dispose();
        // Right-aligned against the SLICED rows, which is why the slice happens first: the weights describe the
        // real tokens, and padding that is no longer there cannot shift them.
        if (CondTokenWeights.Apply(_backend, embeddings, null, sequence).Cond is Tensor scaled)
        {
            embeddings.Dispose();
            embeddings = scaled;
        }
        return embeddings;
    }

    /// <summary>Right-pads to the encoder's fixed window with <see cref="Qwen3PadTokenId"/>, the same padding
    /// <see cref="Qwen3Tokenizer.EncodeChat"/> applies. Already-padded input is returned unchanged.</summary>
    private static int[] PadToWindow(int[] tokens, int window)
    {
        if (tokens.Length >= window)
        {
            return tokens;
        }
        int[] padded = new int[window];
        tokens.CopyTo(padded, 0);
        Array.Fill(padded, Qwen3PadTokenId, tokens.Length, window - tokens.Length);
        return padded;
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
        _checkpoint.Dispose();
        foreach (IDisposable loader in _sideModelLoaders)
        {
            loader.Dispose();
        }
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
