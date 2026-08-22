using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed Wan2.2-S2V pipeline driven against the native <see cref="VideoRequest"/>: the driving speech comes from <see cref="VideoRequest.VideoAudioReference"/> (falling back to <see cref="VideoRequest.VideoAudioInput"/>), is Wav2Vec2-encoded to stacked layer features, resampled from 50 Hz to the clip's 16 fps buckets, and injected per-frame; an optional identity portrait in <see cref="VideoRequest.InitImage"/> becomes appended reference tokens. Mirrors the SwarmUI backend's <c>WanS2VLoader.Generate</c>.</summary>
public sealed class WanS2VRecipePipeline : IVideoRecipePipeline
{
    /// <summary>ComfyUI's Wan sampling shift — <c>WAN22_S2V</c> inherits <c>WAN21_T2V</c>'s sampling settings.</summary>
    private const float DefaultFlowShift = 8f;

    private readonly IBackend _backend;
    private readonly WanS2VPipeline _pipeline;
    private readonly WanVideoConfig _config;
    private readonly T5Tokenizer _tokenizer;
    private readonly T5TextEncoder _umt5;
    private readonly WanS2VTransformer _transformer;
    private readonly WanS2VAudioEncoder _audioEncoder;
    private readonly Wav2Vec2Encoder _wav2vec2;
    private readonly IWanVaeEncoder _vaeEncoder;
    private readonly List<SafeTensorsLoader> _loaders;
    private readonly MergedLoraStack? _loraStack;

    /// <summary>Wraps the constructed S2V pipeline plus its encoders, taking ownership of every disposable.</summary>
    public WanS2VRecipePipeline(IBackend backend, WanS2VPipeline pipeline, WanVideoConfig config, T5Tokenizer tokenizer, T5TextEncoder umt5,
        WanS2VTransformer transformer, WanS2VAudioEncoder audioEncoder, Wav2Vec2Encoder wav2vec2, IWanVaeEncoder vaeEncoder, List<SafeTensorsLoader> loaders, MergedLoraStack? loraStack = null)
    {
        _loraStack = loraStack;
        _backend = backend;
        _pipeline = pipeline;
        _config = config;
        _tokenizer = tokenizer;
        _umt5 = umt5;
        _transformer = transformer;
        _audioEncoder = audioEncoder;
        _wav2vec2 = wav2vec2;
        _vaeEncoder = vaeEncoder;
        _loaders = loaders;
    }

    /// <inheritdoc/>
    public VideoGenerationResult Generate(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        AudioClip speech = request.VideoAudioReference ?? request.VideoAudioInput
            ?? throw new InvalidOperationException(
                "Wan S2V needs driving speech in VideoRequest.VideoAudioReference (or VideoAudioInput) — that's what drives the video.");
        float[] waveform = VideoRecipeUtils.DecodeMono16k(speech);
        // The speech that drove the video is the soundtrack; conditioning downmixes to 16k mono, muxing keeps the source.
        AudioBuffer drivingSpeech = AudioClipCodec.DecodeNative(speech);

        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? _config.NumInferenceSteps;
        int numFrames = VideoRecipeUtils.ResolveFrames(request, modelDefault: 81, step: _config.VaeTemporalCompression);
        float cfgScale = request.CfgScale ?? _config.GuidanceScale;
        (int width, int height) = VideoRecipeUtils.ResolveResolution(request, _config.VaeSpatialCompression);

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

        Tensor? audioLayers = null;
        Tensor? resampled = null;
        Tensor? refLatent = null;
        try
        {
            _backend.PreloadWeights(_wav2vec2.EnumerateWeights());
            Tensor allLayers = _wav2vec2.EncodeAllLayers(_backend, waveform);
            _backend.Sync();
            _backend.FreeWeights(_wav2vec2.EnumerateWeights());
            audioLayers = VideoRecipeUtils.HostCopy(allLayers);
            allLayers.Dispose();

            if (request.InitImage is not null)
            {
                byte[] referenceRgb = VideoRecipeUtils.ResizeRgb24(request.InitImage, width, height);
                Tensor refDev = _pipeline.EncodeReferenceImage(referenceRgb, width, height);
                refLatent = VideoRecipeUtils.HostCopy(refDev);
                refDev.Dispose();
            }

            VideoGenerationRequest inner = new VideoGenerationRequest
            {
                Prompt = prompt,
                NegativePrompt = negative,
                Width = width,
                Height = height,
                Steps = steps,
                CfgScale = cfgScale,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
                FlowShift = DefaultFlowShift,
            };

            Action<GenerationProgress> bridge = p =>
            {
                cancel.ThrowIfCancellationRequested();
                progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
            };

            int tLat = (numFrames - 1) / _config.VaeTemporalCompression + 1;
            resampled = WanS2VPipeline.ResampleAudioFeatures(audioLayers, tLat * 4);
            // The pipeline runs the causal audio encoder before the DiT; its weights must be resident first.
            _backend.PreloadWeights(_audioEncoder.EnumerateWeights());
            try
            {
                (byte[][] frames, int outW, int outH, int _) = _pipeline.GenerateFromAudioFeatures(
                    promptEmbeds, negEmbeds, resampled, inner, numFrames, onProgress: bridge, referenceLatent: refLatent);
                Logs.Info($"[WanS2VRecipePipeline] Pipeline returned {frames.Length} frames {outW}x{outH}.");
                return VideoRecipeUtils.ToResult(frames, outW, outH, request, drivingSpeech.IsEmpty ? null : drivingSpeech);
            }
            finally
            {
                _backend.Sync();
                _backend.FreeWeights(_audioEncoder.EnumerateWeights());
            }
        }
        catch (Exception ex)
        {
            Logs.Error("[WanS2VRecipePipeline] Generation failed.", ex);
            throw;
        }
        finally
        {
            resampled?.Dispose();
            audioLayers?.Dispose();
            refLatent?.Dispose();
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
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
