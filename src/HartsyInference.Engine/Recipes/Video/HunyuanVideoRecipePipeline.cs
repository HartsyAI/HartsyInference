using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed HunyuanVideo pipeline driven against the native <see cref="VideoRequest"/>.
/// <see cref="HunyuanVideoPipeline"/> takes only pre-computed embeddings, so this owns the dual text stack: LLaVA
/// -Llama-3-8B through the diffusers prompt template (layer −3, cropped to drop the template preamble) and CLIP-L's
/// pooled EOS embedding. Mirrors <c>HunyuanVideoGenerationTests</c>' proven construction.</summary>
public sealed class HunyuanVideoRecipePipeline : IVideoRecipePipeline
{
    private readonly IBackend _backend;
    private readonly HunyuanVideoPipeline _pipeline;
    private readonly LlamaStyleEncoder _llava;
    private readonly ClipTextEncoder _clipL;
    private readonly ClipTokenizer _clipTokenizer;
    private readonly HunyuanVideoDit _dit;
    private readonly List<SafeTensorsLoader> _loaders;

    /// <summary>Wraps the constructed HunyuanVideo pipeline plus its dual text stack, taking ownership of every disposable.</summary>
    public HunyuanVideoRecipePipeline(IBackend backend, HunyuanVideoPipeline pipeline, LlamaStyleEncoder llava, ClipTextEncoder clipL, ClipTokenizer clipTokenizer, HunyuanVideoDit dit, List<SafeTensorsLoader> loaders)
    {
        _backend = backend;
        _pipeline = pipeline;
        _llava = llava;
        _clipL = clipL;
        _clipTokenizer = clipTokenizer;
        _dit = dit;
        _loaders = loaders;
    }

    /// <inheritdoc/>
    public IReadOnlyList<VideoFrame> Generate(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        IBackend backend = _backend;
        string prompt = request.Prompt;
        (int width, int height) = VideoRecipeUtils.ResolveResolution(request, multiple: 16);
        int numFrames = VideoRecipeUtils.ResolveFrames(request, modelDefault: 25, step: 4);
        int steps = request.Steps ?? 20;

        Tensor? promptEmbeds = null;
        Tensor? pooled = null;
        try
        {
            int[] llamaTokens = HunyuanVideoRecipe.BuildTemplatedTokens(prompt);
            backend.PreloadWeights(_llava.EnumerateWeights());
            Tensor full = _llava.EncodeMultiLayer(backend, [llamaTokens], [HunyuanVideoRecipe.LlamaLayer]);
            promptEmbeds = CropSequence(full, HunyuanVideoRecipe.CropStart);
            full.Dispose();
            backend.Sync();
            backend.FreeWeights(_llava.EnumerateWeights());

            int[] clipTokens = _clipTokenizer.Encode(prompt);
            int eos = ClipTokenizer.FindEosPosition(clipTokens);
            backend.PreloadWeights(_clipL.EnumerateWeights());
            (Tensor hidden, Tensor? p) = _clipL.EncodePenultimate(backend, [clipTokens], [eos], layersFromEnd: 1);
            hidden.Dispose();
            pooled = p ?? throw new InvalidOperationException("CLIP-L returned no pooled output.");
            backend.Sync();
            backend.FreeWeights(_clipL.EnumerateWeights());

            TextToImageRequest inner = new TextToImageRequest
            {
                Prompt = prompt,
                NegativePrompt = request.NegativePrompt ?? "",
                Width = width,
                Height = height,
                Steps = steps,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
            };

            Action<GenerationProgress> bridge = p2 =>
            {
                cancel.ThrowIfCancellationRequested();
                progress?.Report(new StepPreview { Step = p2.Step, TotalSteps = p2.TotalSteps });
            };

            (byte[][] frames, int outW, int outH, int _) = _pipeline.GenerateFromEmbeddings(promptEmbeds, pooled, inner, numFrames, bridge);
            Logs.Info($"[HunyuanVideoRecipePipeline] Pipeline returned {frames.Length} frames {outW}x{outH}.");
            return VideoRecipeUtils.ToVideoFrames(frames, outW, outH, request);
        }
        catch (Exception ex)
        {
            Logs.Error("[HunyuanVideoRecipePipeline] Generation failed.", ex);
            throw;
        }
        finally
        {
            promptEmbeds?.Dispose();
            pooled?.Dispose();
        }
    }

    /// <summary>Crops the leading <paramref name="crop"/> tokens along the sequence dim of a <c>[1, S, H]</c> tensor.</summary>
    private static unsafe Tensor CropSequence(Tensor t, int crop)
    {
        int s = (int)t.Shape[1], h = (int)t.Shape[2];
        int keep = s - crop;
        if (keep <= 0)
        {
            throw new InvalidOperationException($"Template crop {crop} >= sequence length {s}.");
        }
        Tensor outT = new Tensor(new TensorShape(1, keep, h), DType.F32);
        float* sp = (float*)t.DataPointer, dp = (float*)outT.DataPointer;
        long bytes = (long)keep * h * 4;
        Buffer.MemoryCopy(sp + (long)crop * h, dp, bytes, bytes);
        return outT;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _llava.Dispose();
        _clipTokenizer.Dispose();
        _dit.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
