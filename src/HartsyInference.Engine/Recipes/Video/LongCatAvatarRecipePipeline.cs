using HartsyInference.Audio.Models.LongCat;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed LongCat-Video-Avatar pipeline for one segment: the driving speech goes through Whisper into per-frame windows, the DiT's
/// audio projection, and the avatar denoising loop with the portrait as the first latent frame (Ai2v) when one is given.</summary>
public sealed class LongCatAvatarRecipePipeline : IVideoRecipePipeline
{
    private readonly IBackend _backend;
    private readonly LongCatAvatarPipeline _pipeline;
    private readonly LongCatParts _parts;
    private readonly WhisperEncoder _whisper;
    private readonly MelSpectrogramExtractor _mel;
    private readonly List<IDisposable> _loaders;

    /// <summary>Wraps the constructed pipeline, taking ownership of every disposable in <paramref name="loaders"/>.</summary>
    internal LongCatAvatarRecipePipeline(IBackend backend, LongCatAvatarPipeline pipeline, LongCatParts parts, WhisperEncoder whisper,
        MelSpectrogramExtractor mel, List<IDisposable> loaders)
    {
        _backend = backend;
        _pipeline = pipeline;
        _parts = parts;
        _whisper = whisper;
        _mel = mel;
        _loaders = loaders;
    }

    /// <inheritdoc/>
    public VideoGenerationResult Generate(VideoRequest request, VideoEffectiveSettings settings, IProgress<StepPreview>? progress,
        CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        AudioClip speech = request.VideoAudioReference
            ?? throw new InvalidOperationException("LongCat-Video-Avatar needs driving speech in VideoRequest.VideoAudioReference.");
        float[] waveform = VideoRecipeUtils.DecodeMono16k(speech);
        AudioBuffer drivingSpeech = AudioClipCodec.DecodeNative(speech);

        int width = settings.Width, height = settings.Height, frames = settings.Frames;
        if (frames > LongCatAvatarRecipe.SegmentFrames)
            throw new ArgumentOutOfRangeException(nameof(settings), frames, $"One LongCat-Video-Avatar segment is at most {LongCatAvatarRecipe.SegmentFrames} frames.");
        int latentFrames = (frames - 1) / LongCatRecipeSupport.TemporalCompressionFactor + 1;
        int latentHeight = height / LongCatRecipeSupport.SpatialCompressionFactor;
        int latentWidth = width / LongCatRecipeSupport.SpatialCompressionFactor;
        int seed = RecipeRequestMapper.MapSeed(request.Seed) ?? SeedGenerator.RandomSeed();

        float[] rawAudio = DrivingWindows(waveform, settings.Fps, frames);
        (float[] audioTokens, float[] unconditionalTokens) = ProjectAudio(rawAudio, frames, latentFrames);

        (LongCatPrompt positive, LongCatPrompt negative) = LongCatRecipeSupport.EncodePrompts(_backend, _parts, request.Prompt,
            request.NegativePrompt ?? "");
        float[] noise = LongCatRecipeSupport.Noise(seed, _parts.Channels, latentFrames, latentHeight, latentWidth);

        LongCatAvatarMode mode = LongCatAvatarMode.At2v;
        int condLatents = 0;
        float[]? firstFrame = null;
        if (request.InitImage is not null)
        {
            byte[] rgb = VideoRecipeUtils.ResizeRgb24(request.InitImage, width, height);
            firstFrame = LongCatRecipeSupport.EncodeFirstFrame(_backend, _parts.Encoder, rgb, width, height, _parts.Channels);
            mode = LongCatAvatarMode.Ai2v;
            condLatents = 1;
        }

        LongCatAvatarRun run = new(_parts.Channels, latentFrames, latentHeight, latentWidth, noise, positive, negative,
            audioTokens, unconditionalTokens, settings.Steps, LongCatAvatarSchedule.Kind.Linear, LongCatAvatarPipelineSettings.Shift,
            settings.CfgScale, LongCatAvatarPipelineSettings.AudioGuidance, mode, condLatents, firstFrame);

        float[] latents;
        _backend.PreloadWeights(_parts.Dit.EnumerateWeights());
        try
        {
            latents = _pipeline.Denoise(run).Latents;
        }
        finally
        {
            _backend.Sync();
            _backend.FreeWeights(_parts.Dit.EnumerateWeights());
        }
        cancel.ThrowIfCancellationRequested();

        byte[][] decoded = LongCatRecipeSupport.DecodeFrames(_backend, _parts.Decoder, latents, _parts.Channels, latentFrames, latentHeight, latentWidth);
        Logs.Info($"[LongCatAvatarRecipePipeline] Decoded {decoded.Length} frames {width}x{height}.");
        return VideoRecipeUtils.ToResult(decoded, width, height, request, drivingSpeech.IsEmpty ? null : drivingSpeech, settings.Fps);
    }

    /// <summary>Whisper features of the driving speech, cut into one 5-frame window per video frame: <c>[frames, 5 · 5 · 1280]</c>.</summary>
    private float[] DrivingWindows(float[] waveform, int fps, int frames)
    {
        float[] features = LongCatWhisperFeatures.Extract(_backend, _whisper, _mel, waveform, fps);
        int featureFrames = features.Length / (LongCatAudioLayout.Whisper.Blocks * LongCatAudioLayout.Whisper.Channels);
        if (featureFrames < 1)
            throw new InvalidOperationException("The driving speech is shorter than one video frame.");
        return LongCatAudioFeatures.BuildWindows(features, featureFrames, LongCatAudioLayout.Whisper, 0, frames);
    }

    /// <summary>The audio tokens of the clip and the same clip with zeroed raw audio (the unconditional pass of the three-pass CFG).</summary>
    private (float[] Conditional, float[] Unconditional) ProjectAudio(float[] rawWindows, int frames, int latentFrames)
    {
        int rowWidth = LongCatAudioLayout.Whisper.Window * LongCatAudioLayout.Whisper.Blocks * LongCatAudioLayout.Whisper.Channels;
        using Tensor raw = new(new TensorShape([frames, rowWidth]), DType.F32);
        rawWindows.AsSpan().CopyTo(raw.AsSpan<float>());
        using Tensor zeroRaw = new(new TensorShape([frames, rowWidth]), DType.F32);
        zeroRaw.AsSpan<float>().Clear();

        using Tensor cond = _parts.Dit.ProjectAudio(_backend, raw, latentFrames);
        using Tensor uncond = _parts.Dit.ProjectAudio(_backend, zeroRaw, latentFrames);
        using Tensor condHost = VideoRecipeUtils.HostCopy(cond);
        using Tensor uncondHost = VideoRecipeUtils.HostCopy(uncond);
        return (condHost.AsReadOnlySpan<float>().ToArray(), uncondHost.AsReadOnlySpan<float>().ToArray());
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (IDisposable loader in _loaders)
            loader.Dispose();
        _loaders.Clear();
    }
}
