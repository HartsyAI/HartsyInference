using HartsyInference.Audio.Models.LongCat;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Diffusion.Requests;
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

/// <summary>A constructed LongCat-Video-Avatar pipeline: the driving speech goes through Whisper into per-frame windows, the DiT's audio projection,
/// and the avatar denoising loop with the portrait as the first latent frame (Ai2v) when one is given. A request longer than one segment runs as a
/// chain: each later segment is conditioned on a fixed reference frame and the last latents of the chain so far.</summary>
public sealed class LongCatAvatarRecipePipeline : IVideoRecipePipeline
{
    private readonly IBackend _backend;
    private readonly LongCatAvatarPipeline _pipeline;
    private readonly LongCatParts _parts;
    private readonly WhisperEncoder _whisper;
    private readonly MelSpectrogramExtractor _mel;
    private readonly List<IDisposable> _loaders;
    private readonly bool _distilled;
    private readonly AdjunctGate? _loraGate;

    /// <summary>Wraps the constructed pipeline, taking ownership of every disposable in <paramref name="loaders"/>.</summary>
    internal LongCatAvatarRecipePipeline(IBackend backend, LongCatAvatarPipeline pipeline, LongCatParts parts, WhisperEncoder whisper,
        MelSpectrogramExtractor mel, List<IDisposable> loaders, bool distilled, AdjunctGate? loraGate)
    {
        // A distilled pipeline is only valid with its LoRA gate: the gate is what makes it the distilled model.
        if (distilled && loraGate is null)
            throw new InvalidOperationException("The distilled avatar pipeline needs its DMD LoRA gate; refusing to run the bare base.");
        _distilled = distilled;
        _loraGate = loraGate;
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
        LongCatAvatarRecipe.ValidateChainRequest(request.ChainTotalFrames, frames, width, height, settings.Fps);
        if (width % LongCatRecipeSupport.PatchGrid != 0 || height % LongCatRecipeSupport.PatchGrid != 0)
            throw new ArgumentException($"LongCat-Video-Avatar needs width and height in multiples of {LongCatRecipeSupport.PatchGrid}, got {width}×{height}.");
        int latentFrames = (frames - 1) / LongCatRecipeSupport.TemporalCompressionFactor + 1;
        int latentHeight = height / LongCatRecipeSupport.SpatialCompressionFactor;
        int latentWidth = width / LongCatRecipeSupport.SpatialCompressionFactor;
        int seed = RecipeRequestMapper.MapSeed(settings.Seed) ?? SeedGenerator.RandomSeed();

        // A chain request longer than the segment runs the chain plan; its audio covers the whole clip, so the windows are cut once for all of it.
        int total = request.ChainTotalFrames is int chainTotal && chainTotal > frames ? chainTotal : frames;
        LongCatAvatarChainPlan? plan = total > frames ? LongCatAvatarChainPlan.For(latentFrames, total) : null;
        // The clip delivered is the requested length; the windows past the driving audio repeat its last frame.
        if (!drivingSpeech.IsEmpty && LongCatAvatarRecipe.DrivingAudioIsShort(drivingSpeech.Seconds, total, settings.Fps))
            Logs.Warning($"[LongCatAvatarRecipePipeline] The driving audio lasts {drivingSpeech.Seconds:F2} s and the clip {total / (double)settings.Fps:F2} s; "
                + "the windows past the audio's end repeat its last frame.");
        int clipFrames = plan?.TotalPixelFrames ?? frames;
        float[] rawAudio = DrivingWindows(waveform, settings.Fps, clipFrames);

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

        // The DiT is placed by the streaming scope (see the base recipe); each finished step returns the stream's reservations.
        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel);
        float[] firstWindows = rawAudio.Length == frames * RowWidth ? rawAudio : rawAudio[..(frames * RowWidth)];
        (float[] audioTokens, float[] unconditionalTokens) = ProjectAudio(firstWindows, frames, latentFrames);
        byte[][] decoded;
        if (plan is null)
        {
            float[] latents = LongCatRecipeSupport.RunDenoiseStreamed(_backend, _parts.Dit,
                () => LongCatRecipeSupport.OpenDenoiseStream(_backend, _parts, "longcat-avatar",
                    LongCatRecipeSupport.TokenCount(latentFrames, latentHeight, latentWidth)),
                onStep =>
                {
                    LongCatAvatarRun run = SegmentRun(latentFrames, latentHeight, latentWidth, noise, positive, negative, audioTokens,
                        unconditionalTokens, settings, mode, condLatents, firstFrame, onStep, cancel);
                    return _pipeline.Denoise(run).Latents;
                },
                bridge);
            cancel.ThrowIfCancellationRequested();
            decoded = LongCatRecipeSupport.DecodeFrames(_backend, _parts.Decoder, latents, _parts.Channels, latentFrames, latentHeight, latentWidth);
        }
        else
        {
            // Each segment is decoded once inside the chain and its output frames are kept; no joined latent is decoded.
            decoded = RunChain(plan, rawAudio, latentFrames, latentHeight, latentWidth, seed, noise, positive, negative, settings, mode,
                condLatents, firstFrame, audioTokens, unconditionalTokens, bridge, cancel);
            cancel.ThrowIfCancellationRequested();
        }
        Logs.Info($"[LongCatAvatarRecipePipeline] Decoded {decoded.Length} frames {width}x{height}.");
        // The clip is cut to the requested length, and so is the soundtrack.
        byte[][] clip = decoded.Length > total ? decoded[..total] : decoded;
        AudioBuffer? soundtrack = drivingSpeech.IsEmpty ? null : drivingSpeech.TrimTo((double)clip.Length / settings.Fps);
        return VideoRecipeUtils.ToResult(clip, width, height, request, soundtrack, settings.Fps);
    }

    /// <summary>The chain: segment 0 is an ordinary generation, and the later segments run through the pipeline's chain with their own audio and
    /// noise. Every segment is denoised in its own streaming scope and decoded once, outside any scope; each decoded segment keeps its output frames
    /// (segment 0 all of them, a later segment those after its context), and its last frames are re-encoded as the next context on the 1.5 path.
    /// Returns the chain's output frames, <c>plan.TotalPixelFrames</c> of them.</summary>
    private byte[][] RunChain(LongCatAvatarChainPlan plan, float[] rawAudio, int latentFrames, int latentHeight, int latentWidth, int seed,
        float[] noise, LongCatPrompt positive, LongCatPrompt negative, VideoEffectiveSettings settings, LongCatAvatarMode mode, int condLatents,
        float[]? firstFrame, float[] firstAudio, float[] firstUnconditionalAudio, Action<GenerationProgress> bridge, CancellationToken cancel)
    {
        int channels = _parts.Channels, hw = latentHeight * latentWidth, runLatents = plan.RunLatents;
        // The unconditional tokens are the projection of zeroed audio, the same for every later segment. The run's audio is its windows
        // (RunWindows) projected for the run's latents without the reference slot, which then takes a copy of the first frame's tokens.
        (_, float[] laterUnconditional) = ProjectAudio(new float[plan.RunWindows * RowWidth], plan.RunWindows, runLatents - 1);
        laterUnconditional = LongCatAvatarChainPlan.ReferenceSlotTokens(laterUnconditional, laterUnconditional.Length / (runLatents - 1));
        List<float[]> laterAudio = [];
        List<float[]> laterNoise = [];
        for (int s = 1; s < plan.Segments; s++)
        {
            int[] pixels = plan.RunPixelIds(s);
            float[] raw = GatherWindows(rawAudio, pixels);
            float[] projected = ProjectConditional(raw, pixels.Length, runLatents - 1);
            float[] tokens = LongCatAvatarChainPlan.ReferenceSlotTokens(projected, projected.Length / (runLatents - 1));
            laterAudio.Add(tokens);
            laterNoise.Add(LongCatRecipeSupport.Noise(seed + s, channels, runLatents, latentHeight, latentWidth));
        }

        // Before segment 0: an upper bound on the later runs' budget. It refuses only when even that bound is negative. The later run's own check,
        // before its denoise, is the one that sees segment 0's residue.
        // The later run's own figures: its cached step runs the new latents through the blocks and holds the cache's two pairs.
        // A measured size budgets its later runs on the reserved peak it was measured at; any other size keeps the per-token estimate and its refusal.
        int pixelWidth = latentWidth * LongCatRecipeSupport.SpatialCompressionFactor, pixelHeight = latentHeight * LongCatRecipeSupport.SpatialCompressionFactor;
        long? measuredPeak = LongCatRecipeSupport.MeasuredReservedPeak(pixelWidth, pixelHeight, plan.NewLatentsPerSegment, plan.ContextLatents, _distilled);
        Logs.Info(measuredPeak is long m
            ? $"[LongCatAvatarRecipePipeline] {pixelWidth}x{pixelHeight} has a measured later-run peak of {m >> 20} MiB reserved, plus a {LongCatRecipeSupport.MeasuredMarginBytes >> 20} MiB margin."
            : $"[LongCatAvatarRecipePipeline] {pixelWidth}x{pixelHeight} has no measured later-run peak: the per-token estimate and its refusal apply.");
        long laterTokens = LongCatRecipeSupport.TokenCount(plan.NewLatentsPerSegment, latentHeight, latentWidth);
        long laterCache = LongCatRecipeSupport.CachedStepBytes(_parts.Dit.Config, plan.ContextLatents + 1, latentHeight, latentWidth);
        LongCatRecipeSupport.RequireChainRoom(_backend, _parts, laterTokens, laterCache, measuredPeak);

        // Segment 0: an ordinary generation in its own streaming scope.
        float[] first = LongCatRecipeSupport.RunDenoiseStreamed(_backend, _parts.Dit,
            () => LongCatRecipeSupport.OpenDenoiseStream(_backend, _parts, "longcat-avatar",
                LongCatRecipeSupport.TokenCount(latentFrames, latentHeight, latentWidth)),
            onStep => _pipeline.Denoise(SegmentRun(latentFrames, latentHeight, latentWidth, noise, positive, negative, firstAudio,
                firstUnconditionalAudio, settings, mode, condLatents, firstFrame, onStep, cancel)).Latents,
            bridge);

        // The later segments: each run is denoised in its own scope, so the decode and the encode between runs are outside any scope.
        LongCatAvatarRun template = new(channels, runLatents, latentHeight, latentWidth, laterNoise[0],
            positive, negative, laterAudio[0], laterUnconditional,
            settings.Steps, _distilled ? LongCatAvatarSchedule.Kind.Distill8 : LongCatAvatarSchedule.Kind.Linear,
            settings.FlowShift ?? LongCatAvatarPipelineSettings.Shift, settings.CfgScale,
            _distilled ? 1f : LongCatAvatarPipelineSettings.AudioGuidance, LongCatAvatarMode.Avc, plan.ContextLatents + 1, null,
            Cancel: cancel, LoraGate: _loraGate);

        // Each segment is decoded once. Segment 0 keeps all its frames; a later segment keeps the frames after its context. The last frames of a
        // segment are re-encoded as the next context (the 1.5 build: use_vcond False in the reference, pipeline 1467-1490). The 1.6 build carries
        // the latents instead (see below). The last segment's context is not encoded.
        LongCatChainFrames output = new(1 + LongCatVaeSeam.TemporalScale * (plan.ContextLatents - 1));
        // The 1.6 build carries the segment's last latent frames straight into the next segment (no decode-encode round trip), as the reference's
        // use_vcond True path does. Its frames are still decoded once for the output, so only the context changes.
        Func<float[], bool, float[]> contextFromSegment = (segment, last) =>
        {
            byte[][] decoded = LongCatRecipeSupport.DecodeFrames(_backend, _parts.Decoder, segment, channels, plan.SegmentLatents, latentHeight, latentWidth);
            return output.Step(decoded, last, frames => _distilled
                ? LongCatRecipeSupport.EncodeContext(_backend, _parts.Encoder, frames, plan.ContextLatents, channels, latentHeight, latentWidth)
                : LongCatRecipeSupport.CarriedContext(segment, channels, plan.SegmentLatents, plan.ContextLatents, latentHeight * latentWidth));
        };

        // The later runs: each in its own streaming scope, so the reservations are returned before the decode and the encode.
        int laterRun = 0;
        Func<LongCatAvatarRun, float[]> denoiseSegment = run =>
        {
            laterRun++;
            // The progress steps restart for each segment; this line names the segment they belong to.
            Logs.Info($"[LongCatAvatarRecipePipeline] Later segment {laterRun} of {plan.Segments - 1} (overall segment {laterRun + 1} of {plan.Segments}).");
            return LongCatRecipeSupport.RunDenoiseStreamed(_backend, _parts.Dit,
                // A cached later step runs only its noisy frames through the blocks. The conditioning frames' keys and values come from the offloaded
                // cache: a staged pair and the joined copy of it stay on the device for the block, and are counted with the activations, while the
                // noisy rows' own keys and values are activations. So the headroom counts the new latents, and the cache's two pairs.
                () => LongCatRecipeSupport.OpenDenoiseStream(_backend, _parts, "longcat-avatar",
                    LongCatRecipeSupport.TokenCount(plan.NewLatentsPerSegment, latentHeight, latentWidth),
                    LongCatRecipeSupport.CachedStepBytes(_parts.Dit.Config, plan.ContextLatents + 1, latentHeight, latentWidth),
                    measuredPeakBytes: measuredPeak),
                // The later runs offload their cache to host memory: the device holds one block's keys and values at a time.
                onStep => _pipeline.Denoise(run with { OnProgress = onStep, OffloadKv = true }).Latents,
                bridge);
        };

        _pipeline.Chain(template, plan.ContextLatents, first, laterAudio, laterNoise, contextFromSegment, denoiseSegment);
        if (output.Segments != plan.Segments || output.Frames.Length != plan.TotalPixelFrames)
            throw new InvalidOperationException($"The chain kept {output.Frames.Length} frames over {output.Segments} segments; the plan has {plan.TotalPixelFrames} over {plan.Segments}.");
        return output.Frames;
    }

    /// <summary>One generation's run: the segment's latents, its audio and its conditioning.</summary>
    private LongCatAvatarRun SegmentRun(int latentFrames, int latentHeight, int latentWidth, float[] noise, LongCatPrompt positive,
        LongCatPrompt negative, float[] audioTokens, float[] unconditionalTokens, VideoEffectiveSettings settings, LongCatAvatarMode mode,
        int condLatents, float[]? firstFrame, Action<GenerationProgress> onStep, CancellationToken cancel) =>
        new(_parts.Channels, latentFrames, latentHeight, latentWidth, noise, positive, negative, audioTokens, unconditionalTokens,
            settings.Steps, _distilled ? LongCatAvatarSchedule.Kind.Distill8 : LongCatAvatarSchedule.Kind.Linear,
            settings.FlowShift ?? LongCatAvatarPipelineSettings.Shift, settings.CfgScale,
            _distilled ? 1f : LongCatAvatarPipelineSettings.AudioGuidance, mode, condLatents, firstFrame,
            OnProgress: onStep, Cancel: cancel, LoraGate: _loraGate);

    /// <summary>Whisper features of the driving speech, cut into one 5-frame window per video frame: <c>[frames, 5 · 5 · 1280]</c>.</summary>
    private float[] DrivingWindows(float[] waveform, int fps, int frames)
    {
        float[] features = LongCatWhisperFeatures.Extract(_backend, _whisper, _mel, waveform, fps);
        int featureFrames = features.Length / (LongCatAudioLayout.Whisper.Blocks * LongCatAudioLayout.Whisper.Channels);
        if (featureFrames < 1)
            throw new InvalidOperationException("The driving speech is shorter than one video frame.");
        return LongCatAudioFeatures.BuildWindows(features, featureFrames, LongCatAudioLayout.Whisper, 0, frames);
    }

    /// <summary>The width of one video frame's raw audio window: the Whisper window of blocks and channels.</summary>
    private static int RowWidth => LongCatAudioLayout.Whisper.Window * LongCatAudioLayout.Whisper.Blocks * LongCatAudioLayout.Whisper.Channels;

    /// <summary>The window rows of the given pixel frames, in order.</summary>
    private static float[] GatherWindows(float[] windows, int[] pixels)
    {
        int rowWidth = RowWidth;
        float[] result = new float[pixels.Length * rowWidth];
        for (int i = 0; i < pixels.Length; i++)
            Array.Copy(windows, pixels[i] * rowWidth, result, i * rowWidth, rowWidth);
        return result;
    }

    /// <summary>The conditional audio tokens of a run, without the unconditional projection a later segment does not use.</summary>
    private float[] ProjectConditional(float[] rawWindows, int frames, int latentFrames)
    {
        using Tensor raw = new(new TensorShape([frames, RowWidth]), DType.F32);
        rawWindows.AsSpan().CopyTo(raw.AsSpan<float>());
        using Tensor cond = _parts.Dit.ProjectAudio(_backend, raw, latentFrames);
        using Tensor condHost = VideoRecipeUtils.HostCopy(cond);
        return condHost.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>The audio tokens of the clip and the same clip with zeroed raw audio (the unconditional pass of the three-pass CFG).</summary>
    private (float[] Conditional, float[] Unconditional) ProjectAudio(float[] rawWindows, int frames, int latentFrames)
    {
        using Tensor raw = new(new TensorShape([frames, RowWidth]), DType.F32);
        rawWindows.AsSpan().CopyTo(raw.AsSpan<float>());
        using Tensor zeroRaw = new(new TensorShape([frames, RowWidth]), DType.F32);
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
