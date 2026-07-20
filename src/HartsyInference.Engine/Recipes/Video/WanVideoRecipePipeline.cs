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
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;
using HartsyInference.Video.Pipelines;
using HartsyInference.Vision.Clip;

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
    private readonly WanVideoPipeline _pipeline;
    private readonly WanVideoConfig _config;
    private readonly bool _isClipI2V;
    private readonly T5Tokenizer _tokenizer;
    private readonly T5TextEncoder _umt5;
    private readonly WanVideoTransformer _transformer;
    private readonly IWanVaeEncoder _vaeEncoder;
    private readonly ClipVisionEncoder? _clipVision;
    private readonly List<SafeTensorsLoader> _loaders;

    /// <summary>Wraps the constructed Wan pipeline plus its encoders, taking ownership of every disposable.</summary>
    public WanVideoRecipePipeline(IBackend backend, WanVideoPipeline pipeline, WanVideoConfig config, bool isClipI2V, T5Tokenizer tokenizer,
        T5TextEncoder umt5, WanVideoTransformer transformer, IWanVaeEncoder vaeEncoder, ClipVisionEncoder? clipVision, List<SafeTensorsLoader> loaders)
    {
        _backend = backend;
        _pipeline = pipeline;
        _config = config;
        _isClipI2V = isClipI2V;
        _tokenizer = tokenizer;
        _umt5 = umt5;
        _transformer = transformer;
        _vaeEncoder = vaeEncoder;
        _clipVision = clipVision;
        _loaders = loaders;
    }

    /// <inheritdoc/>
    public IReadOnlyList<VideoFrame> Generate(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps > 0 ? request.Steps : _config.NumInferenceSteps;
        int numFrames = VideoRecipeUtils.ResolveFrames(request, modelDefault: 81, step: _config.VaeTemporalCompression);
        float cfgScale = request.CfgScale <= 0 ? _config.GuidanceScale : request.CfgScale;

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
        Tensor batch = _umt5.Encode(_backend, [promptTokens, negTokens],
            [T5Tokenizer.CreateAttentionMask(promptTokens), T5Tokenizer.CreateAttentionMask(negTokens)]);
        Tensor promptEmbeds = CfgHelper.SliceBatchElement(batch, 0, WanVideoRecipe.TokenLength, _config.TextDim);
        Tensor negEmbeds = CfgHelper.SliceBatchElement(batch, 1, WanVideoRecipe.TokenLength, _config.TextDim);
        batch.Dispose();
        VideoRecipeUtils.ZeroPaddedRows(promptEmbeds, promptTokens, _config.TextDim);
        VideoRecipeUtils.ZeroPaddedRows(negEmbeds, negTokens, _config.TextDim);
        _backend.Sync();
        _backend.FreeWeights(_umt5.EnumerateWeights());

        VideoGenerationRequest inner = new VideoGenerationRequest
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = width,
            Height = height,
            Steps = steps,
            CfgScale = cfgScale,
            Seed = request.Seed < 0 ? null : (int?)(int)(request.Seed & 0x7FFFFFFF),
            FlowShift = DefaultFlowShift,
        };

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
        };

        Tensor? imageEmbeds = null;
        Tensor? firstFrameLatent = null;
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
                return VideoRecipeUtils.ToVideoFrames(concatFrames, concatW, concatH, request);
            }

            if (request.InitImage is not null)
            {
                byte[] frameRgb = VideoRecipeUtils.ResizeRgb24(request.InitImage, width, height);
                firstFrameLatent = _vaeEncoder.EncodeRgbFrame(_backend, frameRgb, width, height);
                _backend.Sync();
                _backend.FreeWeights(_vaeEncoder.EnumerateWeights());
            }
            (byte[][] frames, int outW, int outH, int _) = _pipeline.GenerateFromEmbeddings(promptEmbeds, negEmbeds, inner, numFrames, bridge, firstFrameLatent);
            Logs.Info($"[WanVideoRecipePipeline] Pipeline returned {frames.Length} frames {outW}x{outH} ({(firstFrameLatent is null ? "T2V" : "I2V")}).");
            return VideoRecipeUtils.ToVideoFrames(frames, outW, outH, request);
        }
        catch (Exception ex)
        {
            Logs.Error("[WanVideoRecipePipeline] Generation failed.", ex);
            throw;
        }
        finally
        {
            firstFrameLatent?.Dispose();
            imageEmbeds?.Dispose();
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
        (_vaeEncoder as IDisposable)?.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
