using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
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

/// <summary>A constructed Anima pipeline driven against the native <see cref="ImageRequest"/>. Owns Anima's dual text stack: the Qwen-3 0.6B hidden states (LlmAdapter cross-attention K/V) plus a T5 SentencePiece tokenization of the same prompt (the adapter's main-stream <c>embed[t5_ids]</c> lookup), then calls <see cref="AnimaPipeline.GenerateFromEmbeddings"/>. Mirrors the SwarmUI backend's <c>AnimaLoader.Generate</c> drive path. Anima's Cosmos <c>t/1000</c> timestep normalization lives inside the pipeline.</summary>
public sealed unsafe class AnimaRecipePipeline : IRecipePipeline
{
    /// <summary>Qwen3 right-pads sequences with BosTokenId (151643); the real length ends at the first such pad.</summary>
    private const int Qwen3PadTokenId = 151643;

    private readonly AnimaPipeline _pipeline;
    private readonly LlamaStyleEncoder _qwen;
    private readonly Qwen3Tokenizer _tokenizer;
    private readonly T5Tokenizer _t5Tokenizer;
    private readonly AnimaTransformer _transformer;
    private readonly AnimaLlmAdapter _llmAdapter;
    private readonly IBackend _backend;
    private readonly IDisposable _checkpoint;
    private readonly IReadOnlyList<IDisposable> _sideModelLoaders;

    private readonly MergedLoraStack? _loraStack;

    /// <summary>Wraps the constructed Anima pipeline plus its dual text stack, taking ownership of every disposable.</summary>
    public AnimaRecipePipeline(AnimaPipeline pipeline, LlamaStyleEncoder qwen, Qwen3Tokenizer tokenizer, T5Tokenizer t5Tokenizer,
        AnimaTransformer transformer, AnimaLlmAdapter llmAdapter, IBackend backend,
        IDisposable checkpoint, IReadOnlyList<IDisposable> sideModelLoaders, MergedLoraStack? loraStack = null)
    {
        _loraStack = loraStack;
        _pipeline = pipeline;
        _qwen = qwen;
        _tokenizer = tokenizer;
        _t5Tokenizer = t5Tokenizer;
        _transformer = transformer;
        _llmAdapter = llmAdapter;
        _backend = backend;
        _checkpoint = checkpoint;
        _sideModelLoaders = sideModelLoaders;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? AnimaRecipe.FamilyDefaults.Steps;
        float cfg = request.CfgScale ?? AnimaRecipe.FamilyDefaults.CfgScale;


        // Plain (non-chat) tokenization: Anima's reference workflow uses CLIPLoader type="stable_diffusion", which is
        // Comfy's path for raw Qwen-3 text encoding (no chat template).
        // Declaring ComfyBlend is what stops ImagesService collapsing `(word:N)`, so the recipe owns the grammar
        // now. Only the Qwen-3 arm is blendable: the T5 side is an id lookup inside the adapter
        // (`embed[t5_ids]`), not an encoder output, so there is nothing there to interpolate — its text only
        // needs the grammar taken off, which PromptWeighting.Join does below.
        Tensor positiveEmbeddings = EncodeWeightedQwen(prompt);
        int[] positiveT5Ids = EncodeT5(_t5Tokenizer, StripGrammar(prompt));

        Tensor? negativeEmbeddings = null;
        int[]? negativeT5Ids = null;
        if (cfg > 1.0f)
        {
            negativeEmbeddings = EncodeWeightedQwen(negative);
            negativeT5Ids = EncodeT5(_t5Tokenizer, StripGrammar(negative));
        }

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
                positiveT5Ids,
                inner,
                cfgScale: cfg,
                negativeTextEmbeddings: negativeEmbeddings,
                negativeT5TokenIds: negativeT5Ids,
                onProgress: bridge);

            return new ImageResult
            {
                Rgb = rgb,
                Width = width,
                Height = height,
                Seed = usedSeed,
                Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["arch"] = "anima",
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

    /// <summary>Produces Anima's T5 main-stream token ids: raw SentencePiece tokens + a single EOS with NO padding (the pipeline right-pads the adapter output to 512 itself), mirroring the Python reference <c>sp.encode(prompt, add_eos=True)</c>.</summary>
    private static int[] EncodeT5(T5Tokenizer t5, string text)
    {
        IReadOnlyList<int> raw = t5.EncodeRaw(text);
        int tokenCount = Math.Min(raw.Count, AnimaRecipe.T5MaxTokens - 1);
        int[] result = new int[tokenCount + 1];
        for (int i = 0; i < tokenCount; i++)
        {
            result[i] = raw[i];
        }
        result[tokenCount] = T5Tokenizer.EosTokenId;
        return result;
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

    /// <summary>Slices a [batch, fullLen, hidden] F32 tensor down to [batch, realLen, hidden], dropping the Qwen3 right-padding before the adapter sees the hidden states.</summary>
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

    /// <summary>Encodes one prompt through the Qwen-3 arm and applies its per-token weights. The blend lands on
    /// the FULL padded window, before the real-length slice: the weights describe the padded rows and the
    /// baseline is only subtractable row for row while both still have them.</summary>
    /// <remarks>An unweighted prompt keeps <see cref="Qwen3Tokenizer.Encode"/> verbatim, so its ids and its
    /// encoder shape are exactly what they were before weighting existed.</remarks>
    private Tensor EncodeWeightedQwen(string prompt)
    {
        (int[] tokenIds, float[]? weights) = TokenizeWeightedQwen(prompt);
        int realLen = ComputeRealLength(tokenIds);
        Tensor encodedFull = _qwen.Encode(_backend, new[] { tokenIds });
        if (weights is not null)
        {
            using Tensor emptyFull = _qwen.Encode(_backend, new[] { _tokenizer.Encode("", appendEos: true) });
            if (ComfyBlend.Apply(_backend, encodedFull, emptyFull, weights) is Tensor blended)
            {
                encodedFull.Dispose();
                encodedFull = blended;
            }
        }
        Tensor sliced = SliceFirstSeqF32(encodedFull, realLen);
        encodedFull.Dispose();
        return sliced;
    }

    /// <summary>Mirrors <see cref="Qwen3Tokenizer.Encode"/>'s padding — EOS then BOS-as-pad to the fixed window —
    /// while carrying one weight per row. Pad and EOS rows weigh 1: they are not part of the prompt, and blending
    /// them would pull the padding toward the empty encode along with the words.</summary>
    private (int[] Tokens, float[]? Weights) TokenizeWeightedQwen(string prompt)
    {
        IReadOnlyList<WeightedSpan> spans = PromptWeighting.Parse(PromptTagFlattening.Flatten(prompt));
        if (!PromptWeighting.HasWeights(spans))
        {
            return (_tokenizer.Encode(PromptWeighting.Join(spans), appendEos: true), null);
        }
        WeightedTokenSequence built = WeightedTokenBuilder.Build(spans, _tokenizer.EncodeRaw, [], []);
        int window = _tokenizer.MaxLength;
        int[] tokens = new int[window];
        float[] weights = new float[window];
        Array.Fill(tokens, Qwen3Tokenizer.BosTokenId);
        Array.Fill(weights, 1f);
        int real = Math.Min(built.Tokens.Length, window - 1);
        Array.Copy(built.Tokens, tokens, real);
        Array.Copy(built.Weights, weights, real);
        tokens[real] = Qwen3Tokenizer.EosTokenId;
        return (tokens, weights);
    }

    /// <summary>The prompt with its emphasis grammar removed, for the arm that cannot act on it.</summary>
    private static string StripGrammar(string prompt) =>
        PromptWeighting.Join(PromptWeighting.Parse(PromptTagFlattening.Flatten(prompt)));

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _qwen.Dispose();
        _tokenizer.Dispose();
        _t5Tokenizer.Dispose();
        _transformer.Dispose();
        _llmAdapter.Dispose();
        _checkpoint.Dispose();
        foreach (IDisposable loader in _sideModelLoaders)
        {
            loader.Dispose();
        }
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
