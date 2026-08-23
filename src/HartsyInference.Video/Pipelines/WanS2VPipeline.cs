using HartsyInference.Diffusion.Sampling;
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

/// <summary>Wan2.2-S2V (speech-to-video) pipeline, matching ComfyUI's <c>WanSoundImageToVideo</c> conditioning flow:
/// Wav2Vec2 stacked-layer features are resampled 50 Hz → 30 fps (linear, align-corners) then bucket-sampled at 16 fps
/// to one feature stack per VIDEO frame (<c>4·T_lat</c> frames); <see cref="WanS2VAudioEncoder"/> turns them into
/// per-latent-frame audio tokens the DiT's audio injector cross-attends to. The reference image is VAE-encoded and
/// passed to the transformer as APPENDED TOKENS (not channel-concat). CFG runs the negative pass with the audio
/// features zeroed (then re-encoded), per the reference node. Flow-match UniPC + CFG-renorm; Wan2.1 z=16 VAE.
///
/// <para><b>TODO:</b> the autoregressive extend path (<c>WanSoundImageToVideoExtend</c> — FramePackMotioner motion
/// tokens) is not implemented; single-clip only.</para></summary>
public sealed unsafe class WanS2VPipeline : DiffusionPipelineBase
{
    private const int AudioSourceFps = 50;    // Wav2Vec2 hop = 320 samples at 16 kHz
    private const int AudioVideoRate = 30;    // intermediate rate the reference node interpolates to
    private const int VideoFps = 16;          // S2V output frame rate the buckets are sampled at

    private readonly WanS2VTransformer _transformer;
    private readonly WanS2VAudioEncoder _audioEncoder;
    private readonly IWanVaeDecoder _vae;
    private readonly IWanVaeEncoder? _encoder;
    private readonly WanVideoConfig _config;

    /// <summary><paramref name="encoder"/> (a Wan VAE encoder) is required only for reference-image conditioning.</summary>
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

    /// <summary>End-to-end from a raw mono 16 kHz waveform: runs <paramref name="wav2vec2"/>, resamples the stacked
    /// hidden states to one feature per video frame (50 Hz → 30 fps linear → 16 fps bucket, per the reference node),
    /// then generates. <paramref name="referenceRgb24"/> is an optional identity reference image (RGB24, request-sized).</summary>
    public (byte[][] frames, int width, int height, int seed) GenerateFromWaveform(
        Tensor promptEmbeds, Tensor negativeEmbeds, ReadOnlySpan<float> waveform, Wav2Vec2Encoder wav2vec2,
        TextToImageRequest request, int numFrames, ReadOnlySpan<byte> referenceRgb24 = default,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        int tLat = (numFrames - 1) / _config.VaeTemporalCompression + 1;
        Tensor allLayers = wav2vec2.EncodeAllLayers(Backend, waveform);          // [T50, numStates, hidden]
        Tensor resampled = ResampleAudioFeatures(allLayers, tLat * 4);           // [4·tLat, numStates, hidden]
        allLayers.Dispose();
        try
        {
            return GenerateFromAudioFeatures(promptEmbeds, negativeEmbeds, resampled, request, numFrames, referenceRgb24, onProgress);
        }
        finally { resampled.Dispose(); }
    }

    /// <summary>Generates a clip from per-video-frame audio features <c>[≥ 4·T_lat, numLayers, audioDim]</c> (one
    /// stacked Wav2Vec2 feature per output video frame; extra frames are sliced off, missing ones zero-padded, matching
    /// the reference's <c>audio_embed[:, :, :, :T·4]</c>). <paramref name="referenceRgb24"/> optionally conditions
    /// identity via appended reference tokens (needs the VAE encoder); <paramref name="referenceLatent"/> is an
    /// optional caller-owned precomputed reference latent (from <see cref="EncodeReferenceImage"/> — the cross-gen
    /// same-image cache) that takes priority over <paramref name="referenceRgb24"/> and is NOT disposed here.</summary>
    public (byte[][] frames, int width, int height, int seed) GenerateFromAudioFeatures(
        Tensor promptEmbeds, Tensor negativeEmbeds, Tensor audioFeatures, TextToImageRequest request, int numFrames,
        ReadOnlySpan<byte> referenceRgb24 = default, Action<GenerationProgress>? onProgress = null,
        Tensor? referenceLatent = null)
    {
        // Sampler selection is NOT wired on this family (2026-08-20 audit): the family samples with a UniPC multistep predictor/corrector, not an Euler step, so it has no sampler seam to drive. This family samples with UniPC, so even an explicit 'euler' cannot be honoured here.
        // Refuse rather than accepting the request and sampling with something else.
        FlowMatchSampling.ThrowIfSamplerSelected(request.Scheduler, "Wan S2V");

        ThrowIfDisposed();
        int width = request.Width ?? 832, height = request.Height ?? 480;
        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        if (width % sp != 0 || height % sp != 0)
            throw new ArgumentException($"Width/height must be divisible by {sp} for Wan-S2V.");
        if (numFrames < 1 || (numFrames - 1) % tp != 0)
            throw new ArgumentException($"num_frames must satisfy (num_frames-1) % {tp} == 0; got {numFrames}.");
        if (referenceLatent is null && !referenceRgb24.IsEmpty && _encoder is null)
            throw new InvalidOperationException("S2V reference-image conditioning needs a Wan VAE encoder.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int tLat = (numFrames - 1) / tp + 1, hLat = height / sp, wLat = width / sp, latentCh = _config.VaeLatentChannels;
        int tVideo = tLat * 4;
        int steps = request.Steps ?? _config.NumInferenceSteps;
        float guidance = request.CfgScale ?? _config.GuidanceScale;
        float shift = (request as VideoGenerationRequest)?.FlowShift ?? _config.FlowShift;
        bool hasRef = referenceLatent is not null || !referenceRgb24.IsEmpty;

        Logs.Info($"Wan-S2V: {numFrames}f {width}x{height}, {steps} steps, cfg={guidance}, seed={seed} " +
            $"(latent {latentCh}x{tLat}x{hLat}x{wLat}, ref={(hasRef ? (referenceLatent is not null ? "cached" : "yes") : "no")})");

        // Audio is fixed across steps: encode the real features for the conditional pass and ZEROED features for the
        // unconditional pass (the reference node sets negative audio_embed = audio_embed · 0, which still runs the
        // causal audio encoder — its biases/padding token make that a non-zero "silence" embedding).
        Tensor features = FitToFrames(audioFeatures, tVideo);
        (Tensor audioGlobalC, Tensor audioLocalC) = _audioEncoder.Forward(Backend, features);
        if (WanVideoDebugDump.Enabled)
        {
            WanVideoDebugDump.SetTag("cond");
            WanVideoDebugDump.Dump("audio_features", features);
            WanVideoDebugDump.Dump("audio_global", audioGlobalC);
            WanVideoDebugDump.Dump("audio_local", audioLocalC);
        }
        ZeroTensor(features);
        (Tensor audioGlobalU, Tensor audioLocalU) = _audioEncoder.Forward(Backend, features);
        if (WanVideoDebugDump.Enabled)
        {
            WanVideoDebugDump.SetTag("uncond");
            WanVideoDebugDump.Dump("audio_features", features);
            WanVideoDebugDump.Dump("audio_global", audioGlobalU);
            WanVideoDebugDump.Dump("audio_local", audioLocalU);
            WanVideoDebugDump.SetTag(null);
        }
        features.Dispose();

        Tensor? refLatent = referenceLatent;
        bool ownsRefLatent = false;
        if (refLatent is null && !referenceRgb24.IsEmpty)
        {
            refLatent = EncodeReferenceImage(referenceRgb24, width, height);   // [1, z, 1, hLat, wLat]
            ownsRefLatent = true;
            WanVideoDebugDump.Dump("ref_latent", refLatent);
        }

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        Tensor latents = SeedGenerator.CreateNoise(new TensorShape([1L, latentCh, tLat, hLat, wLat]), seed);
        // VALIDATION-PENDING: Wan 2.2 UniPC scheduler trajectory vs the ComfyUI S2V sampler.
        FlowUniPCMultistepScheduler scheduler = new(solverOrder: 2);
        scheduler.SetTimesteps(steps, shift);

        bool wanDebug = Environment.GetEnvironmentVariable("HARTSY_WAN_DEBUG") == "1";
        // Text-only CFG: run the SAME audio on both branches so guidance steers text adherence without pushing the
        // sample away from the audio contribution. The reference zeroes the negative branch's audio, but our
        // silence-audio uncond darkens the output roughly linearly in (cfg−1) until the numeric parity of that path
        // is settled — this mode is the usable interim for cfg>2 (HARTSY_S2V_TEXT_CFG=1).
        bool textOnlyCfg = Environment.GetEnvironmentVariable("HARTSY_S2V_TEXT_CFG") == "1";
        Tensor uncondLocal = textOnlyCfg ? audioLocalC : audioLocalU;
        Tensor uncondGlobal = textOnlyCfg ? audioGlobalC : audioGlobalU;
        for (int k = 0; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float tEmb = scheduler.Timesteps[k];
            if (wanDebug) Logs.Info($"[S2VDBG] step {k} pre-cond    free={Backend.FreeMemoryBytes() >> 20}MB");
            WanVideoDebugDump.SetTag("cond");
            Tensor vCond = _transformer.Forward(Backend, latents, promptEmbeds, tEmb, audioLocalC, audioGlobalC, refLatent);
            if (wanDebug) Logs.Info($"[S2VDBG] step {k} post-cond   free={Backend.FreeMemoryBytes() >> 20}MB");
            WanVideoDebugDump.SetTag("uncond");
            Tensor vUncond = _transformer.Forward(Backend, latents, negativeEmbeds, tEmb, uncondLocal, uncondGlobal, refLatent);
            if (wanDebug) Logs.Info($"[S2VDBG] step {k} post-uncond free={Backend.FreeMemoryBytes() >> 20}MB");
            WanVideoDebugDump.SetTag(null);
            LancePipelineCommon.CfgCombineRenormInPlace(vCond, vUncond, guidance, _config.CfgRescale);
            WanVideoDebugDump.Dump("cfg_combined", vCond);
            scheduler.Step(latents, vCond);
            vCond.Dispose(); vUncond.Dispose();
            sw.Stop();
            onProgress?.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds)
            {
                Latent = latents,
                LatentArch = LatentArchitecture.Wan,
            });
            // Reclaim GPU-resident activation buffers between steps AND trim the stream-ordered pool (the latent
            // is host-side, so nothing cross-step is lost). Measured 2026-07-11 (post multi-group WanDitOps port,
            // 8-step A/B, 15.7 GB resident DiT + shared-card ambient): trim ON 3.44 s/step / 68 async-OOM retries
            // vs trim OFF 4.23 s/step / 169 retries — beside a near-capacity pool the per-step trim PREVENTS the
            // allocation-retry stalls, it is not merely an OOM band-aid. WAN_S2V_TRIM=0 skips it for A/Bs.
            Backend.FreeActivations();
            if (Environment.GetEnvironmentVariable("WAN_S2V_TRIM") != "0")
                Backend.TrimMemoryPool();
            if (wanDebug) Logs.Info($"[S2VDBG] step {k} post-free   free={Backend.FreeMemoryBytes() >> 20}MB");
        }

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());
        audioGlobalC.Dispose(); audioLocalC.Dispose(); audioGlobalU.Dispose(); audioLocalU.Dispose();
        if (ownsRefLatent) refLatent?.Dispose();

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

    /// <summary>VAE-encodes the identity reference portrait to the appended-token latent <c>[1, z, 1, hLat, wLat]</c>
    /// (preloads, syncs, and frees the encoder weights around the encode). Exposed so callers can compute the latent
    /// once and cache it across generations keyed on the image (pass it back via <c>referenceLatent</c>).</summary>
    public Tensor EncodeReferenceImage(ReadOnlySpan<byte> rgb24, int width, int height)
    {
        ThrowIfDisposed();
        if (_encoder is null)
            throw new InvalidOperationException("S2V reference-image conditioning needs a Wan VAE encoder.");
        Backend.PreloadWeights(_encoder.EnumerateWeights());
        Tensor refLatent = _encoder.EncodeRgbFrame(Backend, rgb24, width, height);
        Backend.Sync();
        Backend.FreeWeights(_encoder.EnumerateWeights());
        return refLatent;
    }

    /// <summary>Reference audio resampling (nodes_wan.py <c>wan_sound_to_video</c>): 50 Hz Wav2Vec2 features →
    /// 30 fps via linear interpolation (align_corners=True), then per-video-frame bucket sampling at 16 fps
    /// (<c>index = round(i·30/16)</c>, numpy banker's rounding; frames past the audio are zeros).
    /// <paramref name="allLayers"/> is <c>[T50, layers, dim]</c>; returns <c>[tVideo, layers, dim]</c>.</summary>
    public static Tensor ResampleAudioFeatures(Tensor allLayers, int tVideo)
    {
        int t50 = (int)allLayers.Shape[0], layers = (int)allLayers.Shape[1], dim = (int)allLayers.Shape[2];
        long perFrame = (long)layers * dim;
        int t30 = (int)((long)t50 * AudioVideoRate / AudioSourceFps);
        Tensor o = new Tensor(new TensorShape(tVideo, layers, dim), DType.F32);
        float* xp = (float*)allLayers.DataPointer, op = (float*)o.DataPointer;
        new Span<float>(op, (int)Math.Min(int.MaxValue, o.Shape.ElementCount)).Clear();
        for (int i = 0; i < tVideo; i++)
        {
            // numpy's np.round is round-half-to-even, which C# Math.Round matches by default.
            int bi = (int)Math.Round(i * (double)AudioVideoRate / VideoFps);
            if (bi >= t30) continue;   // past the audio: stays zero (the reference pads the bucket with zeros)
            // align_corners=True linear interpolation from the 50 Hz grid to 30 fps frame bi.
            double src = t30 == 1 ? 0 : bi * (double)(t50 - 1) / (t30 - 1);
            int lo = (int)src;
            int hi = Math.Min(lo + 1, t50 - 1);
            float frac = (float)(src - lo);
            float* dst = op + (long)i * perFrame;
            float* a = xp + (long)lo * perFrame;
            float* b = xp + (long)hi * perFrame;
            for (long j = 0; j < perFrame; j++) dst[j] = a[j] + (b[j] - a[j]) * frac;
        }
        return o;
    }

    /// <summary>Slices (or zero-pads) the per-video-frame features to exactly <paramref name="tVideo"/> frames —
    /// the reference's <c>audio_embed[:, :, :, :T·4]</c> slice + zero bucket padding. Always returns a fresh copy.</summary>
    private static Tensor FitToFrames(Tensor features, int tVideo)
    {
        int tIn = (int)features.Shape[0], layers = (int)features.Shape[1], dim = (int)features.Shape[2];
        long perFrame = (long)layers * dim;
        Tensor o = new Tensor(new TensorShape(tVideo, layers, dim), DType.F32);
        float* xp = (float*)features.DataPointer, op = (float*)o.DataPointer;
        int copy = Math.Min(tIn, tVideo);
        Buffer.MemoryCopy(xp, op, (long)copy * perFrame * 4, (long)copy * perFrame * 4);
        for (long i = (long)copy * perFrame; i < (long)tVideo * perFrame; i++) op[i] = 0f;
        return o;
    }

    private static void ZeroTensor(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.Shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = 0f;
    }
}
