using HartsyInference.Engine.Features;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video.Pipelines;
using HartsyInference.Vision.Clip;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed Wan T2V / I2V / TI2V pipeline driven against the native <see cref="VideoRequest"/>. Encodes
/// the prompt pair with umT5-XXL (zero-padding the rows past the real tokens — Wan cross-attends all 512 context rows
/// with no text mask), then routes to the concat-conditioned I2V path (36-channel <c>[noise, mask, cond-latent]</c>,
/// plus CLIP-ViT-H context on Wan2.1 I2V) or the TI2V <c>expand_timesteps</c> / plain T2V path. Mirrors the SwarmUI
/// backend's <c>WanVideoLoader.RunPipeline</c>.</summary>
public sealed class WanVideoRecipePipeline : IVideoRecipePipeline
{
    /// <summary>ComfyUI's Wan sampling shift (<c>WAN21_T2V</c>, inherited by I2V/VACE) — the reference look, rather
    /// than the official repo's resolution-tied 3.0/5.0 presets which under-form structure at 15-20 steps.</summary>
    private const float DefaultFlowShift = 8f;

    private readonly IBackend _backend;
    /// <summary>Backend the umT5 prompt encoder runs on — separable from <see cref="_backend"/> because the
    /// embeddings are host-materialized (SliceBatchElement/ZeroPaddedRows are host loops) before the denoiser
    /// consumes them. Wan always runs real CFG with a ~T5-XXL-class encoder, so moving it off the denoiser GPU is
    /// the single biggest consumer-tier VRAM win in the video stack.</summary>
    private readonly IBackend _textBackend;
    /// <summary>Backend the VAE encode/decode runs on — separable from <see cref="_backend"/> for the same reason
    /// as <see cref="_textBackend"/>. Used directly for the TI2V first-frame encode below, which calls
    /// <see cref="IWanVaeEncoder.EncodeRgbFrame"/> instead of routing through <see cref="_pipeline"/>'s own
    /// <see cref="WanVideoPipeline.VaeBackend"/> — this field must stay in sync with that pipeline's VaeBackend
    /// (both are set from <c>context.VaeBackendOrDefault</c> in <c>WanVideoRecipe</c>).</summary>
    private readonly IBackend _vaeBackend;
    private readonly WanVideoPipeline _pipeline;
    private readonly WanVideoConfig _config;
    private readonly bool _isClipI2V;
    private readonly T5Tokenizer _tokenizer;
    private readonly T5TextEncoder _umt5;
    private readonly WanVideoTransformer _transformer;
    private readonly WanVideoTransformer? _transformer2;
    private readonly IWanVaeEncoder _vaeEncoder;
    private readonly ClipVisionEncoder? _clipVision;
    private readonly List<SafeTensorsLoader> _loaders;
    private readonly MergedLoraStack? _loraStack;

    /// <summary>Wraps the constructed Wan pipeline plus its encoders, taking ownership of every disposable.
    /// <paramref name="textBackend"/>/<paramref name="vaeBackend"/> may equal <paramref name="backend"/>
    /// (single-device default).</summary>
    public WanVideoRecipePipeline(IBackend backend, IBackend textBackend, IBackend vaeBackend, WanVideoPipeline pipeline, WanVideoConfig config, bool isClipI2V, T5Tokenizer tokenizer,
        T5TextEncoder umt5, WanVideoTransformer transformer, IWanVaeEncoder vaeEncoder, ClipVisionEncoder? clipVision, List<SafeTensorsLoader> loaders,
        WanVideoTransformer? transformer2 = null, MergedLoraStack? loraStack = null)
    {
        _transformer2 = transformer2;
        _backend = backend;
        _textBackend = textBackend;
        _vaeBackend = vaeBackend;
        _pipeline = pipeline;
        _config = config;
        _isClipI2V = isClipI2V;
        _tokenizer = tokenizer;
        _umt5 = umt5;
        _transformer = transformer;
        _vaeEncoder = vaeEncoder;
        _clipVision = clipVision;
        _loaders = loaders;
        _loraStack = loraStack;
    }

    /// <inheritdoc/>
    public VideoGenerationResult Generate(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? _config.NumInferenceSteps;
        int numFrames = VideoRecipeUtils.ResolveFrames(request, modelDefault: 81, step: _config.VaeTemporalCompression);
        float cfgScale = request.CfgScale ?? _config.GuidanceScale;

        // TODO(E-IMG-4/5): the SigmaShift T2IParam override is not on the native contract — the ComfyUI default is used.
        int width, height;
        if (request.InitImage is not null)
        {
            (width, height) = VideoRecipeUtils.ResolveI2VResolution(request, request.InitImage.Width, request.InitImage.Height, _config.VaeSpatialCompression);
        }
        else
        {
            (width, height) = VideoRecipeUtils.ResolveResolution(request, _config.VaeSpatialCompression);
        }

        int[] promptTokens = _tokenizer.Encode(prompt);
        int[] negTokens = _tokenizer.Encode(negative);
        // umT5 runs on the (possibly separate) text backend; the host-side slice/zero passes below ARE the
        // cross-device boundary — they force the embeddings to host, so the denoiser's backend re-uploads from
        // there. Load-bearing for TextEncoderDevice placement: keep them host-side.
        Tensor batch = _umt5.Encode(_textBackend, [promptTokens, negTokens],
            [T5Tokenizer.CreateAttentionMask(promptTokens), T5Tokenizer.CreateAttentionMask(negTokens)]);
        Tensor promptEmbeds = CfgHelper.SliceBatchElement(batch, 0, WanVideoRecipe.TokenLength, _config.TextDim);
        Tensor negEmbeds = CfgHelper.SliceBatchElement(batch, 1, WanVideoRecipe.TokenLength, _config.TextDim);
        batch.Dispose();
        VideoRecipeUtils.ZeroPaddedRows(promptEmbeds, promptTokens, _config.TextDim);
        VideoRecipeUtils.ZeroPaddedRows(negEmbeds, negTokens, _config.TextDim);
        _textBackend.Sync();
        _textBackend.FreeWeights(_umt5.EnumerateWeights());

        VideoGenerationRequest inner = new VideoGenerationRequest
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = width,
            Height = height,
            Steps = steps,
            CfgScale = cfgScale,
            Seed = RecipeRequestMapper.MapSeed(request.Seed),
            // Routed through the resolver rather than read raw, so an unavailable sampler is refused by name here —
            // before the checkpoint loads — instead of deep inside the pipeline, or silently dropped.
            Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
            FlowShift = DefaultFlowShift,
        };

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
        };

        Tensor? imageEmbeds = null;
        Tensor? firstFrameLatent = null;
        Tensor? lastFrameLatent = null;
        try
        {
            bool isConcatI2V = _config.InChannels > _config.VaeLatentChannels;
            if (isConcatI2V)
            {
                if (request.InitImage is null)
                {
                    throw new InvalidOperationException("This Wan I2V model requires an init image (VideoRequest.InitImage).");
                }
                if (_isClipI2V && _clipVision is not null)
                {
                    _backend.PreloadWeights(_clipVision.EnumerateWeights());
                    ClipImagePreprocessor preprocessor = new ClipImagePreprocessor(imageSize: 224);
                    Tensor pixels = preprocessor.Preprocess(request.InitImage.Rgb, request.InitImage.Width, request.InitImage.Height);
                    Tensor batched = _clipVision.EncodeHiddenStates(_backend, pixels);
                    pixels.Dispose();
                    _backend.Sync();
                    _backend.FreeWeights(_clipVision.EnumerateWeights());
                    imageEmbeds = VideoRecipeUtils.DropBatch(batched);
                    batched.Dispose();
                }
                byte[] frameRgb = VideoRecipeUtils.ResizeRgb24(request.InitImage, width, height);
                byte[]? lastRgb = request.VideoEndFrame is null ? null : VideoRecipeUtils.ResizeRgb24(request.VideoEndFrame, width, height);
                (byte[][] concatFrames, int concatW, int concatH, int _) = _pipeline.GenerateImageToVideoConcat(
                    promptEmbeds, negEmbeds, imageEmbeds, frameRgb, inner, numFrames, bridge, lastRgb);
                Logs.Info($"[WanVideoRecipePipeline] Concat-I2V returned {concatFrames.Length} frames {concatW}x{concatH}.");
                return VideoRecipeUtils.ToResult(concatFrames, concatW, concatH, request);
            }

            // TI2V expand_timesteps path: encode whichever of InitImage/VideoEndFrame are present, sharing one
            // weight preload/free pass (both calls hit the same resident VAE encoder weights — freeing after the
            // first would force a needless reload for the second). Symmetric to MiniMaxH3RecipePipeline's
            // EncodeKeyframes; unlike the concat-I2V branch above (Wan2.1/14B family), this path has no
            // model-level requirement that both ends be set together — either alone is a valid TI2V conditioning,
            // though end-frame-alone is unverified (Wan's real-world usage always pairs an init image).
            if (request.InitImage is not null)
            {
                byte[] frameRgb = VideoRecipeUtils.ResizeRgb24(request.InitImage, width, height);
                firstFrameLatent = _vaeEncoder.EncodeRgbFrame(_vaeBackend, frameRgb, width, height);
            }
            if (request.VideoEndFrame is not null)
            {
                byte[] endRgb = VideoRecipeUtils.ResizeRgb24(request.VideoEndFrame, width, height);
                lastFrameLatent = _vaeEncoder.EncodeRgbFrame(_vaeBackend, endRgb, width, height);
            }
            if (firstFrameLatent is not null || lastFrameLatent is not null)
            {
                _vaeBackend.Sync();
                _vaeBackend.FreeWeights(_vaeEncoder.EnumerateWeights());
            }
            (byte[][] frames, int outW, int outH, int _) = _pipeline.GenerateFromEmbeddings(promptEmbeds, negEmbeds, inner, numFrames, bridge, firstFrameLatent, lastFrameLatent);
            string mode = firstFrameLatent is not null && lastFrameLatent is not null ? "FLF2V"
                : firstFrameLatent is not null ? "I2V"
                : lastFrameLatent is not null ? "EndFrame-only"
                : "T2V";
            Logs.Info($"[WanVideoRecipePipeline] Pipeline returned {frames.Length} frames {outW}x{outH} ({mode}).");
            return VideoRecipeUtils.ToResult(frames, outW, outH, request);
        }
        catch (Exception ex)
        {
            Logs.Error("[WanVideoRecipePipeline] Generation failed.", ex);
            throw;
        }
        finally
        {
            firstFrameLatent?.Dispose();
            lastFrameLatent?.Dispose();
            imageEmbeds?.Dispose();
            promptEmbeds.Dispose();
            negEmbeds.Dispose();
        }
    }

    /// <inheritdoc/>
    public bool SupportsStreaming => true;

    /// <summary>Streaming sibling of <see cref="Generate"/> (Tier 3.5). Scoped tight and honestly: plain T2V only
    /// today — no init image / end frame (the concat-I2V and TI2V-conditioned paths both add extra
    /// encode/dispose bookkeeping around the denoise call that hasn't been exercised through this path, so they
    /// throw rather than silently reusing the buffered branch's logic unverified). Text-encode setup is a
    /// deliberate near-duplicate of <see cref="Generate"/>'s (not extracted into a shared helper) — the two
    /// methods' cleanup shapes differ enough (a plain try/finally here vs. <see cref="Generate"/>'s
    /// multi-tensor try/finally covering branches this method never takes) that a shared helper would need its
    /// own parameter surface for "which of these five tensors exist," which isn't simpler than the duplication.</summary>
    public async IAsyncEnumerable<VideoFrame> GenerateFramesAsync(VideoRequest request, IProgress<StepPreview>? progress,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        if (request.InitImage is not null || request.VideoEndFrame is not null)
        {
            throw new NotSupportedException(
                "HartsyInference: streaming generation only supports plain text-to-video for Wan today (no init image / end frame) — use the buffered path.");
        }
        if (_config.InChannels > _config.VaeLatentChannels)
        {
            throw new NotSupportedException(
                "HartsyInference: streaming generation is not available for this Wan I2V (concat-conditioned) variant — use the buffered path.");
        }

        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? _config.NumInferenceSteps;
        int numFrames = VideoRecipeUtils.ResolveFrames(request, modelDefault: 81, step: _config.VaeTemporalCompression);
        float cfgScale = request.CfgScale ?? _config.GuidanceScale;
        (int width, int height) = VideoRecipeUtils.ResolveResolution(request, _config.VaeSpatialCompression);

        int[] promptTokens = _tokenizer.Encode(prompt);
        int[] negTokens = _tokenizer.Encode(negative);
        Tensor batch = _umt5.Encode(_textBackend, [promptTokens, negTokens],
            [T5Tokenizer.CreateAttentionMask(promptTokens), T5Tokenizer.CreateAttentionMask(negTokens)]);
        Tensor promptEmbeds = CfgHelper.SliceBatchElement(batch, 0, WanVideoRecipe.TokenLength, _config.TextDim);
        Tensor negEmbeds = CfgHelper.SliceBatchElement(batch, 1, WanVideoRecipe.TokenLength, _config.TextDim);
        batch.Dispose();
        VideoRecipeUtils.ZeroPaddedRows(promptEmbeds, promptTokens, _config.TextDim);
        VideoRecipeUtils.ZeroPaddedRows(negEmbeds, negTokens, _config.TextDim);
        _textBackend.Sync();
        _textBackend.FreeWeights(_umt5.EnumerateWeights());

        VideoGenerationRequest inner = new VideoGenerationRequest
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = width,
            Height = height,
            Steps = steps,
            CfgScale = cfgScale,
            Seed = RecipeRequestMapper.MapSeed(request.Seed),
            // Routed through the resolver rather than read raw, so an unavailable sampler is refused by name here —
            // before the checkpoint loads — instead of deep inside the pipeline, or silently dropped.
            Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
            FlowShift = DefaultFlowShift,
        };
        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
        };

        try
        {
            int emitted = 0;
            // _pipeline.GenerateFramesAsync yields HartsyInference.Video.VideoFrame (the low-level decode-side
            // record) — mapped here into HartsyInference.Engine.Requests.VideoFrame (the native request/result
            // DTO this interface's callers expect), same field-for-field shape as VideoRecipeUtils.ToVideoFrames
            // does for the buffered path.
            await foreach (HartsyInference.Video.VideoFrame frame in _pipeline.GenerateFramesAsync(promptEmbeds, negEmbeds, inner, numFrames, bridge, cancellationToken: cancel).ConfigureAwait(false))
            {
                emitted++;
                yield return new VideoFrame { Rgb = frame.Rgb, Width = frame.Width, Height = frame.Height, Index = frame.Index };
            }
            Logs.Info($"[WanVideoRecipePipeline] Streamed {emitted} frames {width}x{height} (T2V).");
        }
        finally
        {
            promptEmbeds.Dispose();
            negEmbeds.Dispose();
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _tokenizer.Dispose();
        _umt5.Dispose();
        _transformer.Dispose();
        _transformer2?.Dispose();
        (_vaeEncoder as IDisposable)?.Dispose();
        // The LoRA stack owns the merged tensors both transformers reference, so it outlives them by exactly
        // this much (same pattern as Sd3RecipePipeline/Flux1RecipePipeline/SdxlRecipePipeline).
        _loraStack?.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
