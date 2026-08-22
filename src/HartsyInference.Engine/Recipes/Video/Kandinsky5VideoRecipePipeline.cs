using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
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

/// <summary>A constructed Kandinsky 5 T2V-Lite pipeline driven against the native <see cref="VideoRequest"/>. <see cref="Kandinsky5VideoPipeline"/> takes only pre-computed embeddings, so — mirroring <see cref="Image.Kandinsky5RecipePipeline"/> — this owns the dual text stack via the shared <see cref="Kandinsky5TextEncoding"/> helper: Qwen2.5-VL-7B for the sequence embeddings, CLIP-L for the pooled vector, both through Kandinsky's fixed "promt engineer" ChatML template.</summary>
public sealed class Kandinsky5VideoRecipePipeline : IVideoRecipePipeline
{
    private readonly Kandinsky5VideoPipeline _pipeline;
    private readonly LlamaStyleEncoder _qwen;
    private readonly ClipTextEncoder _clipL;
    private readonly Qwen2Tokenizer _qwenTokenizer;
    private readonly ClipTokenizer _clipTokenizer;
    private readonly IBackend _backend;
    private readonly Kandinsky5Transformer _transformer;
    private readonly List<SafeTensorsLoader> _loaders;
    private readonly MergedLoraStack? _loraStack;

    /// <summary>Wraps the constructed Kandinsky 5 Video pipeline plus its dual text stack, taking ownership of every disposable.</summary>
    public Kandinsky5VideoRecipePipeline(Kandinsky5VideoPipeline pipeline, LlamaStyleEncoder qwen, ClipTextEncoder clipL,
        Qwen2Tokenizer qwenTokenizer, ClipTokenizer clipTokenizer, IBackend backend, Kandinsky5Transformer transformer, List<SafeTensorsLoader> loaders, MergedLoraStack? loraStack = null)
    {
        _loraStack = loraStack;
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
    public VideoGenerationResult Generate(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        float cfg = request.CfgScale ?? 5.0f;
        bool useCfg = cfg > 1.0f;
        (int width, int height) = VideoRecipeUtils.ResolveResolution(request, multiple: 8);
        int numFrames = VideoRecipeUtils.ResolveFrames(request, modelDefault: 121, step: 4);

        // Image-to-video: the start frame is VAE-encoded once and pinned as latent frame 0, so only frames 1..
        // denoise. Resized to the resolved generation size first, matching the other i2v families.
        Tensor? firstFrameLatent = null;
        if (request.InitImage is not null)
        {
            byte[] rgb = VideoRecipeUtils.ResizeRgb24(request.InitImage, width, height);
            firstFrameLatent = _pipeline.EncodeFirstFrame(rgb, width, height);
        }

        Tensor? qwenEmbeds = null;
        Tensor? negQwenEmbeds = null;
        Tensor? clipPooled = null;
        Tensor? negClipPooled = null;
        try
        {
            // Qwen2.5-VL is ~16 GB: bulk-upload, encode both branches, then free before the transformer preload
            // (the T2I recipe's preload -> encode -> free pattern).
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
                Width = width,
                Height = height,
                Steps = request.Steps ?? 50,
                CfgScale = cfg,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
            };

            Action<GenerationProgress> bridge = p =>
            {
                cancel.ThrowIfCancellationRequested();
                progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
            };

            (byte[][] frames, int outW, int outH, int _) = _pipeline.GenerateFromEmbeddings(
                qwenEmbeds, clipPooled, negQwenEmbeds, negClipPooled, inner, numFrames, bridge, firstFrameLatent);
            Logs.Info($"[Kandinsky5VideoRecipePipeline] Pipeline returned {frames.Length} frames {outW}x{outH}.");
            return VideoRecipeUtils.ToResult(frames, outW, outH, request);
        }
        catch (Exception ex)
        {
            Logs.Error("[Kandinsky5VideoRecipePipeline] Generation failed.", ex);
            throw;
        }
        finally
        {
            qwenEmbeds?.Dispose();
            negQwenEmbeds?.Dispose();
            clipPooled?.Dispose();
            negClipPooled?.Dispose();
            firstFrameLatent?.Dispose();
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
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
