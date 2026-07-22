using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

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
    private readonly SafeTensorsLoader _checkpointLoader;
    private readonly SafeTensorsLoader _qwenLoader;
    private readonly SafeTensorsLoader _vaeLoader;

    /// <summary>Wraps the constructed Anima pipeline plus its dual text stack, taking ownership of every disposable.</summary>
    public AnimaRecipePipeline(AnimaPipeline pipeline, LlamaStyleEncoder qwen, Qwen3Tokenizer tokenizer, T5Tokenizer t5Tokenizer,
        AnimaTransformer transformer, AnimaLlmAdapter llmAdapter, IBackend backend,
        SafeTensorsLoader checkpointLoader, SafeTensorsLoader qwenLoader, SafeTensorsLoader vaeLoader)
    {
        _pipeline = pipeline;
        _qwen = qwen;
        _tokenizer = tokenizer;
        _t5Tokenizer = t5Tokenizer;
        _transformer = transformer;
        _llmAdapter = llmAdapter;
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
        int steps = request.Steps ?? AnimaRecipe.FamilyDefaults.Steps;
        float cfg = request.CfgScale ?? AnimaRecipe.FamilyDefaults.CfgScale;

        // TODO(E-IMG-4): img2img/inpaint (request.Img2Img/Inpaint) is not yet mapped — text-to-image only.

        // Plain (non-chat) tokenization: Anima's reference workflow uses CLIPLoader type="stable_diffusion", which is
        // Comfy's path for raw Qwen-3 text encoding (no chat template).
        int[] tokenIds = _tokenizer.Encode(prompt, appendEos: true);
        int realLen = ComputeRealLength(tokenIds);
        Tensor encodedFull = _qwen.Encode(_backend, new[] { tokenIds });
        Tensor positiveEmbeddings = SliceFirstSeqF32(encodedFull, realLen);
        encodedFull.Dispose();
        int[] positiveT5Ids = EncodeT5(_t5Tokenizer, prompt);

        Tensor? negativeEmbeddings = null;
        int[]? negativeT5Ids = null;
        if (cfg > 1.0f)
        {
            int[] negTokens = _tokenizer.Encode(negative, appendEos: true);
            int negRealLen = ComputeRealLength(negTokens);
            Tensor negEncodedFull = _qwen.Encode(_backend, new[] { negTokens });
            negativeEmbeddings = SliceFirstSeqF32(negEncodedFull, negRealLen);
            negEncodedFull.Dispose();
            negativeT5Ids = EncodeT5(_t5Tokenizer, negative);
        }

        TextToImageRequest inner = new TextToImageRequest
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = request.Width,
            Height = request.Height,
            Steps = steps,
            Seed = RecipeRequestMapper.MapSeed(request.Seed),
        };

        try
        {
            Action<GenerationProgress> bridge = p =>
            {
                cancel.ThrowIfCancellationRequested();
                progress?.Report(new StepPreview { Step = p.Step, TotalSteps = steps });
            };

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

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _qwen.Dispose();
        _tokenizer.Dispose();
        _t5Tokenizer.Dispose();
        _transformer.Dispose();
        _llmAdapter.Dispose();
        _checkpointLoader.Dispose();
        _qwenLoader.Dispose();
        _vaeLoader.Dispose();
    }
}
