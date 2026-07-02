using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Video.Pipelines;

/// <summary>Wan2.2-S2V (speech-to-video) pipeline — single-chunk. Drives the <see cref="WanS2VTransformer"/>:
/// per-frame Wav2Vec2 audio features are encoded (<see cref="WanS2VAudioEncoder"/>) into audio tokens that the DiT's
/// audio injector cross-attends to, alongside umT5 text. Flow-match Euler + 2-way text CFG; reuses the Wan2.1 z=16 VAE.
///
/// <para><b>Status:</b> structural, reconstructed (no diffusers reference). The <b>audio features are pre-computed</b>
/// (the Wav2Vec2 front-end is the deferred Phase S0), the <b>reference-image identity conditioning and the
/// autoregressive multi-chunk long-video loop are NOT modeled here</b> (single clip, audio+text only). Numerics +
/// structure validation-pending.</para></summary>
public sealed unsafe class WanS2VPipeline : DiffusionPipelineBase
{
    private readonly WanS2VTransformer _transformer;
    private readonly WanS2VAudioEncoder _audioEncoder;
    private readonly IWanVaeDecoder _vae;
    private readonly IWanVaeEncoder? _encoder;
    private readonly WanVideoConfig _config;

    /// <summary><paramref name="encoder"/> is required only for the reference-conditioned / multi-chunk path
    /// (<see cref="GenerateChunked"/>); the audio+text single-clip path doesn't need it.</summary>
    public WanS2VPipeline(IBackend backend, WanS2VTransformer transformer, WanS2VAudioEncoder audioEncoder,
        IWanVaeDecoder vae, WanVideoConfig config, IWanVaeEncoder? encoder = null)
        : base(backend)
    {
        _transformer = transformer;
        _audioEncoder = audioEncoder;
        _vae = vae;
        _encoder = encoder;
        _config = config;
    }

    /// <summary>End-to-end from a raw mono 16 kHz waveform: runs <paramref name="wav2vec2"/> over the audio, resamples
    /// its per-timestep hidden states to the <c>gt</c> latent frames, then generates. (Phase S0 front-end → S3.)</summary>
    public (byte[][] frames, int width, int height, int seed) GenerateFromWaveform(
        Tensor promptEmbeds, Tensor negativeEmbeds, ReadOnlySpan<float> waveform, Wav2Vec2Encoder wav2vec2,
        TextToImageRequest request, int numFrames, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        int tp = _config.VaeTemporalCompression;
        int tLat = (numFrames - 1) / tp + 1;
        Tensor allLayers = wav2vec2.EncodeAllLayers(Backend, waveform);   // [T_feat, numStates, hidden]
        Tensor resampled = ResampleToFrames(allLayers, tLat);            // [tLat, numStates, hidden]
        allLayers.Dispose();
        try
        {
            return GenerateFromAudioFeatures(promptEmbeds, negativeEmbeds, resampled, request, numFrames, onProgress);
        }
        finally { resampled.Dispose(); }
    }

    /// <summary>Reference-conditioned, multi-chunk autoregressive long-video generation. The reference image is
    /// VAE-encoded and concatenated (broadcast over time) to the model input each step (so the DiT must be configured
    /// with <c>InChannels == 2·z</c>); audio is split into clips of <paramref name="framesPerClip"/>, and each clip
    /// after the first carries the previous clip's last <paramref name="motionFrames"/> (latent) frames as clean
    /// motion conditioning (imposed every step), giving temporal continuity.
    ///
    /// <para><b>Structural / validation-gated:</b> the exact motioner channel layout, clip overlap, and audio-window
    /// mapping are reconstructed and provisional.</para></summary>
    public (byte[][] frames, int width, int height, int seed) GenerateChunked(
        Tensor promptEmbeds, Tensor negativeEmbeds, Tensor audioFeaturesFull, ReadOnlySpan<byte> referenceRgb24,
        TextToImageRequest request, int framesPerClip, int motionFrames = 1, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        if (_encoder is null)
            throw new InvalidOperationException("S2V reference/multi-chunk needs a Wan VAE encoder.");
        int z = _config.VaeLatentChannels;
        if (_config.InChannels != 2 * z)
            throw new InvalidOperationException($"Reference-conditioned S2V needs InChannels == 2·z ({2 * z}); got {_config.InChannels}.");
        int width = request.Width ?? 832, height = request.Height ?? 480;
        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        if (width % sp != 0 || height % sp != 0)
            throw new ArgumentException($"Width/height must be divisible by {sp}.");
        if ((framesPerClip - 1) % tp != 0)
            throw new ArgumentException($"framesPerClip must satisfy (n-1) % {tp} == 0; got {framesPerClip}.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int hLat = height / sp, wLat = width / sp;
        int gtPerClip = (framesPerClip - 1) / tp + 1;
        int totalLat = (int)audioFeaturesFull.Shape[0];
        int steps = request.Steps ?? _config.NumInferenceSteps;
        float guidance = request.CfgScale ?? _config.GuidanceScale;
        float shift = _config.FlowShift;
        int motionLat = Math.Clamp(motionFrames, 0, gtPerClip - 1);

        Logs.Info($"Wan-S2V (chunked): {width}x{height}, clip {gtPerClip} latent frames, {totalLat} total, " +
            $"motion {motionLat}, {steps} steps, seed={seed}");
        Logs.Warning("Wan-S2V chunked pipeline is reconstructed + validation-pending (motioner layout provisional).");

        // Reference latent (broadcast across each clip's frames as the conditioning half of the input).
        Backend.PreloadWeights(_encoder.EnumerateWeights());
        Tensor refLatent = _encoder.EncodeRgbFrame(Backend, referenceRgb24, width, height);   // [1,z,1,hLat,wLat]
        Backend.Sync();
        Backend.FreeWeights(_encoder.EnumerateWeights());

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        // VALIDATION-PENDING: Wan 2.2 UniPC scheduler (solver_order=2, bh2, predict_x0, flow sigmas, exponential
        // shift) — verify the S2V per-clip UniPC trajectory (with motion-frame re-imposition each step) vs the
        // diffusers Wan S2V pipeline. A fresh trajectory is started per clip via SetTimesteps.
        FlowUniPCMultistepScheduler scheduler = new(solverOrder: 2);
        List<byte[]> allFrames = new();
        Tensor? prevTail = null;   // [1,z,motionLat,hLat,wLat] clean latent carried between clips
        int clipStart = 0, clipIdx = 0;

        while (clipStart < totalLat)
        {
            int gt = Math.Min(gtPerClip, totalLat - clipStart);
            if (gt <= motionLat && clipIdx > 0) break;   // nothing new to generate
            Tensor audioClip = SliceFrames(audioFeaturesFull, clipStart, gt);
            Tensor audioTokens = _audioEncoder.Forward(Backend, audioClip);   // [gt, tokens, dim]
            audioClip.Dispose();

            Tensor latents = SeedGenerator.CreateNoise(new TensorShape([1L, z, gt, hLat, wLat]), seed + clipIdx);
            scheduler.SetTimesteps(steps, shift);
            for (int k = 0; k < steps; k++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                float tEmb = scheduler.Timesteps[k];
                if (prevTail is not null && motionLat > 0) ImposeMotionFrames(latents, prevTail, motionLat);
                Tensor modelInput = ConcatReference(latents, refLatent, z);   // [1, 2z, gt, hLat, wLat]
                Tensor vCond = _transformer.Forward(Backend, modelInput, audioTokens, promptEmbeds, tEmb);
                Tensor vUncond = _transformer.Forward(Backend, modelInput, audioTokens, negativeEmbeds, tEmb);
                modelInput.Dispose();
                LancePipelineCommon.CfgCombineRenormInPlace(vCond, vUncond, guidance, _config.CfgRescale);
                scheduler.Step(latents, vCond);
                vCond.Dispose(); vUncond.Dispose();
                sw.Stop();
                onProgress?.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds) { Latent = latents, LatentArch = LatentArchitecture.Wan });
            }
            if (prevTail is not null && motionLat > 0) ImposeMotionFrames(latents, prevTail, motionLat);
            audioTokens.Dispose();

            // Carry the last motionLat frames as the next clip's motion conditioning.
            prevTail?.Dispose();
            prevTail = motionLat > 0 ? SliceLatentFrames(latents, gt - motionLat, motionLat, z, hLat, wLat) : null;

            Tensor rgb = _vae.Decode(Backend, latents);
            latents.Dispose();
            int f = (int)rgb.Shape[2];
            // Drop the overlap (motion) output frames on clips after the first.
            int startF = clipIdx == 0 ? 0 : motionLat * tp;
            for (int i = startF; i < f; i++) allFrames.Add(VideoRgbFrames.ExtractFrame(rgb, i));
            rgb.Dispose();

            clipStart += gt - motionLat;
            clipIdx++;
            if (gt < gtPerClip) break;
        }

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());
        prevTail?.Dispose();
        refLatent.Dispose();
        Logs.Info($"Wan-S2V chunked complete ({allFrames.Count} frames, {clipIdx} clips, seed={seed})");
        return (allFrames.ToArray(), width, height, seed);
    }

    /// <summary>Concatenates the reference latent (frame 0, broadcast across all <c>gt</c> frames) as channels [z,2z).</summary>
    private static Tensor ConcatReference(Tensor noisy, Tensor refLatent, int z)
    {
        int gt = (int)noisy.Shape[2], h = (int)noisy.Shape[3], w = (int)noisy.Shape[4];
        long frame = (long)h * w, perCh = (long)gt * frame;
        Tensor o = new Tensor(new TensorShape([1L, 2 * z, gt, h, w]), DType.F32);
        float* np = (float*)noisy.DataPointer, rp = (float*)refLatent.DataPointer, op = (float*)o.DataPointer;
        for (int c = 0; c < z; c++) Buffer.MemoryCopy(np + (long)c * perCh, op + (long)c * perCh, perCh * 4, perCh * 4);
        for (int c = 0; c < z; c++)
            for (int ti = 0; ti < gt; ti++)
                Buffer.MemoryCopy(rp + (long)c * frame, op + ((long)(z + c) * perCh + (long)ti * frame), frame * 4, frame * 4);
        return o;
    }

    private static void ImposeMotionFrames(Tensor latents, Tensor motion, int motionLat)
    {
        int z = (int)latents.Shape[1], gt = (int)latents.Shape[2];
        long frame = latents.Shape[3] * latents.Shape[4];
        float* lp = (float*)latents.DataPointer, mp = (float*)motion.DataPointer;
        for (int c = 0; c < z; c++)
            Buffer.MemoryCopy(mp + (long)c * motionLat * frame, lp + (long)c * gt * frame, (long)motionLat * frame * 4, (long)motionLat * frame * 4);
    }

    private static Tensor SliceLatentFrames(Tensor latents, int start, int count, int z, int hLat, int wLat)
    {
        int gt = (int)latents.Shape[2];
        long frame = (long)hLat * wLat;
        Tensor o = new Tensor(new TensorShape([1L, z, count, hLat, wLat]), DType.F32);
        float* lp = (float*)latents.DataPointer, op = (float*)o.DataPointer;
        for (int c = 0; c < z; c++)
            Buffer.MemoryCopy(lp + ((long)c * gt + start) * frame, op + (long)c * count * frame, (long)count * frame * 4, (long)count * frame * 4);
        return o;
    }

    private static Tensor SliceFrames(Tensor audio, int start, int count)
    {
        int layers = (int)audio.Shape[1], dim = (int)audio.Shape[2];
        long per = (long)layers * dim;
        Tensor o = new Tensor(new TensorShape(count, layers, dim), DType.F32);
        Buffer.MemoryCopy((float*)audio.DataPointer + (long)start * per, (float*)o.DataPointer, (long)count * per * 4, (long)count * per * 4);
        return o;
    }

    /// <summary>Nearest-neighbor temporal resample of <c>[T_in, layers, dim]</c> → <c>[T_out, layers, dim]</c>
    /// (maps the ≈50 Hz Wav2Vec2 features to the latent frame rate; structural — linear/area resample is a refinement).</summary>
    private static Tensor ResampleToFrames(Tensor x, int tOut)
    {
        int tIn = (int)x.Shape[0], layers = (int)x.Shape[1], dim = (int)x.Shape[2];
        Tensor o = new Tensor(new TensorShape(tOut, layers, dim), DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        long perFrame = (long)layers * dim;
        for (int to = 0; to < tOut; to++)
        {
            int ti = tOut == 1 ? 0 : (int)((long)to * (tIn - 1) / (tOut - 1));
            ti = Math.Clamp(ti, 0, tIn - 1);
            Buffer.MemoryCopy(xp + (long)ti * perFrame, op + (long)to * perFrame, perFrame * 4, perFrame * 4);
        }
        return o;
    }

    /// <summary>Generates a clip from per-frame audio features. <paramref name="audioFeatures"/> is the Wav2Vec2
    /// stacked-layer features resampled to the latent frame rate: <c>[gt, numLayers, audioDim]</c> where
    /// <c>gt = (numFrames-1)/tp + 1</c> (one audio group per latent frame).</summary>
    public (byte[][] frames, int width, int height, int seed) GenerateFromAudioFeatures(
        Tensor promptEmbeds, Tensor negativeEmbeds, Tensor audioFeatures, TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        int width = request.Width ?? 832, height = request.Height ?? 480;
        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        if (width % sp != 0 || height % sp != 0)
            throw new ArgumentException($"Width/height must be divisible by {sp} for Wan-S2V.");
        if (numFrames < 1 || (numFrames - 1) % tp != 0)
            throw new ArgumentException($"num_frames must satisfy (num_frames-1) % {tp} == 0; got {numFrames}.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int tLat = (numFrames - 1) / tp + 1, hLat = height / sp, wLat = width / sp, latentCh = _config.VaeLatentChannels;
        if ((int)audioFeatures.Shape[0] != tLat)
            throw new ArgumentException($"audioFeatures must have {tLat} frame groups (latent frames); got {audioFeatures.Shape[0]}.", nameof(audioFeatures));
        int steps = request.Steps ?? _config.NumInferenceSteps;
        float guidance = request.CfgScale ?? _config.GuidanceScale;
        float shift = _config.FlowShift;

        Logs.Info($"Wan-S2V: {numFrames}f {width}x{height}, {steps} steps, cfg={guidance}, seed={seed} " +
            $"(latent {latentCh}x{tLat}x{hLat}x{wLat})");
        Logs.Warning("Wan-S2V pipeline is reconstructed + first-run-validation pending — reference/multi-chunk not modeled.");

        // Encode the audio once (it's fixed across denoise steps).
        Tensor audioTokens = _audioEncoder.Forward(Backend, audioFeatures);   // [gt, tokens, dim]

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        Tensor latents = SeedGenerator.CreateNoise(new TensorShape([1L, latentCh, tLat, hLat, wLat]), seed);
        // VALIDATION-PENDING: Wan 2.2 UniPC scheduler (solver_order=2, bh2, predict_x0, flow sigmas, exponential
        // shift) — verify the single-clip S2V UniPC trajectory vs the diffusers Wan S2V pipeline.
        FlowUniPCMultistepScheduler scheduler = new(solverOrder: 2);
        scheduler.SetTimesteps(steps, shift);

        for (int k = 0; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float tEmb = scheduler.Timesteps[k];
            Tensor vCond = _transformer.Forward(Backend, latents, audioTokens, promptEmbeds, tEmb);
            Tensor vUncond = _transformer.Forward(Backend, latents, audioTokens, negativeEmbeds, tEmb);
            LancePipelineCommon.CfgCombineRenormInPlace(vCond, vUncond, guidance, _config.CfgRescale);
            scheduler.Step(latents, vCond);
            vCond.Dispose(); vUncond.Dispose();
            sw.Stop();
            onProgress?.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds)
            {
                Latent = latents,
                LatentArch = LatentArchitecture.Wan,
            });
        }

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());
        audioTokens.Dispose();

        Tensor rgb;
        try { rgb = _vae.Decode(Backend, latents); }
        finally { latents.Dispose(); }
        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = VideoRgbFrames.ExtractFrame(rgb, i);
        rgb.Dispose();
        Logs.Info($"Wan-S2V complete ({frames.Length} frames, seed={seed})");
        return (frames, width, height, seed);
    }
}
