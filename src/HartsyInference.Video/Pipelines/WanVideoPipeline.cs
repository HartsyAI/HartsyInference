using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Video.Pipelines;

/// <summary>Wan-Video (Wan-AI, Apache-2.0) text-to-video pipeline — Wan2.2 TI2V-5B. Maximum reuse: the DiT denoises directly in VAE-latent space <c>[1,48,T,H,W]</c> (the transformer patchifies/unpatchifies internally), and the VAE is the **already-built <see cref="Wan22VaeDecoder"/>** (z=48, 16×/4×, streaming). Flow-match Euler + 2-way text CFG; reuses <see cref="LancePipelineCommon"/> + frame streaming.
///
/// <para>Takes pre-computed umT5 features (encode upstream with the shared T5 encoder). <c>T_lat = (num_frames−1)/4 + 1</c>; latent <c>H/16 × W/16</c>. <b>Status: built, first-run validation pending</b> — the flow-match shift (5.0/3.0), scheduler (UniPC vs Euler), and DiT timestep scaling are validation-gated.</para></summary>
public sealed unsafe class WanVideoPipeline : DiffusionPipelineBase
{
    private readonly WanVideoTransformer _transformer;
    private readonly WanVideoTransformer? _transformer2;   // low-noise expert (Wan2.2 A14B MoE)
    private readonly IWanVaeDecoder _vae;
    private readonly IWanVaeEncoder? _encoder;
    private readonly WanVideoConfig _config;

    /// <summary><paramref name="encoder"/> is optional and only needed for RGB-input I2V (<see cref="EncodeFirstFrame"/>);
    /// it loads from the same VAE weight dict as the decoder. <paramref name="transformer2"/> is the low-noise expert
    /// for the Wan2.2 A14B MoE (the constructor's <paramref name="transformer"/> is then the high-noise expert); leave
    /// it null for single-expert variants.</summary>
    public WanVideoPipeline(IBackend backend, WanVideoTransformer transformer, IWanVaeDecoder vae, WanVideoConfig config,
        IWanVaeEncoder? encoder = null, WanVideoTransformer? transformer2 = null)
        : base(backend)
    {
        _transformer = transformer;
        _vae = vae;
        _config = config;
        _encoder = encoder;
        _transformer2 = transformer2;
    }

    /// <summary>Selects the MoE expert for a given (×1000) timestep: high-noise expert while <c>tEmb ≥ boundary·1000</c>,
    /// else the low-noise expert. Single-expert variants always return the primary transformer.</summary>
    private WanVideoTransformer Expert(float tEmb) =>
        (_config.IsMixtureOfExperts && _transformer2 is not null && tEmb < _config.BoundaryRatio * 1000f)
            ? _transformer2 : _transformer;

    /// <summary>Encodes an interleaved-RGB24 conditioning frame to the normalized first-frame latent for the TI2V
    /// I2V path — pass the result as <c>firstFrameLatent</c> to <see cref="GenerateFromEmbeddings"/> /
    /// <see cref="GenerateFramesAsync"/>. The caller owns (disposes) the returned tensor. Requires the pipeline to be
    /// constructed with a <see cref="Wan22VaeEncoder"/>.</summary>
    public Tensor EncodeFirstFrame(ReadOnlySpan<byte> rgb24, int width, int height)
    {
        ThrowIfDisposed();
        if (_encoder is null)
            throw new InvalidOperationException("RGB-input I2V needs a Wan22VaeEncoder — construct the pipeline with one (it loads from the same VAE weights).");
        return _encoder.EncodeRgbFrame(Backend, rgb24, width, height);
    }

    /// <summary>Generates frames from pre-computed umT5 features <c>[L, textDim]</c>. Returns one interleaved-RGB <c>byte[]</c> per frame.
    /// <para><paramref name="firstFrameLatent"/> switches to the TI2V image-to-video path (diffusers
    /// <c>expand_timesteps</c>): a <c>[1, 48, 1, H/16, W/16]</c> VAE-encoded <b>and latent-normalized</b>
    /// (<see cref="Wan22VaeLatentNorm.Normalize"/>) first frame that is re-imposed into the model input each step at
    /// per-frame timestep 0 while the remaining frames denoise. The Wan2.2 VAE <i>encoder</i> is not built yet —
    /// produce the conditioning latent offline (validation-gated).</para></summary>
    public (byte[][] frames, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor promptEmbeds, Tensor negativeEmbeds, TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress = null, Tensor? firstFrameLatent = null)
    {
        Tensor latent = RunDenoise(promptEmbeds, negativeEmbeds, request, numFrames, onProgress, firstFrameLatent, out int seed);
        Tensor rgb;
        try { rgb = _vae.Decode(Backend, latent); }
        finally { latent.Dispose(); }

        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = FrameToBytes(rgb, i);
        rgb.Dispose();
        Logs.Info($"Wan-Video T2V complete ({frames.Length} frames, seed={seed})");
        return (frames, request.Width, request.Height, seed);
    }

    /// <summary>Streams decoded frames (pull-based → memory bounded; pair with an <c>IVideoEncoder</c>).
    /// <paramref name="firstFrameLatent"/> enables the TI2V image-to-video path — see <see cref="GenerateFromEmbeddings"/>.</summary>
    public async IAsyncEnumerable<VideoFrame> GenerateFramesAsync(
        Tensor promptEmbeds, Tensor negativeEmbeds, TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress = null, Tensor? firstFrameLatent = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Tensor latent = RunDenoise(promptEmbeds, negativeEmbeds, request, numFrames, onProgress, firstFrameLatent, out _);
        if (_vae is Wan22VaeDecoder w22)
        {
            try
            {
                int idx = 0;
                foreach (Tensor group in w22.DecodeStreaming(Backend, latent))   // [1,3,groupT,H,W] per latent frame
                {
                    int gT = (int)group.Shape[2], h = (int)group.Shape[3], w = (int)group.Shape[4];
                    for (int gi = 0; gi < gT; gi++)
                        yield return new VideoFrame(idx++, w, h, FrameToBytesGroup(group, gi));
                    group.Dispose();
                    cancellationToken.ThrowIfCancellationRequested();
                    await Task.Yield();
                }
            }
            finally { latent.Dispose(); }
            yield break;
        }
        // Non-streaming VAE (Wan2.1): full decode, then emit each frame.
        Tensor rgb;
        try { rgb = _vae.Decode(Backend, latent); }
        finally { latent.Dispose(); }
        try
        {
            int f = (int)rgb.Shape[2], w = (int)rgb.Shape[4], h = (int)rgb.Shape[3];
            for (int i = 0; i < f; i++)
            {
                yield return new VideoFrame(i, w, h, FrameToBytes(rgb, i));
                cancellationToken.ThrowIfCancellationRequested();
                await Task.Yield();
            }
        }
        finally { rgb.Dispose(); }
    }

    /// <summary>Runs the flow-match denoise loop in (normalized) latent space and returns <c>[1,48,T_lat,H_lat,W_lat]</c>.
    /// With <paramref name="firstFrameLatent"/> set, follows the diffusers <c>expand_timesteps</c> I2V path: the model
    /// input gets the condition imposed on frame 0 with per-frame timestep 0 each step, the evolving latents are
    /// stepped freely, and the condition is re-imposed once after the loop.</summary>
    private Tensor RunDenoise(Tensor promptEmbeds, Tensor negativeEmbeds, TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress, Tensor? firstFrameLatent, out int seed)
    {
        ThrowIfDisposed();
        seed = request.Seed ?? SeedGenerator.RandomSeed();
        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        if (request.Width % sp != 0 || request.Height % sp != 0)
            throw new ArgumentException($"Width/height must be divisible by {sp} for Wan-Video.");
        if (numFrames < 1 || (numFrames - 1) % tp != 0)
            throw new ArgumentException($"num_frames must satisfy (num_frames-1) % {tp} == 0; got {numFrames}.");

        int tLat = (numFrames - 1) / tp + 1;
        int hLat = request.Height / sp, wLat = request.Width / sp;
        if (firstFrameLatent is not null &&
            (firstFrameLatent.Shape.Rank != 5 || firstFrameLatent.Shape[0] != 1 || firstFrameLatent.Shape[1] != _config.InChannels
             || firstFrameLatent.Shape[2] != 1 || firstFrameLatent.Shape[3] != hLat || firstFrameLatent.Shape[4] != wLat))
            throw new ArgumentException($"firstFrameLatent must be [1,{_config.InChannels},1,{hLat},{wLat}]; got {firstFrameLatent.Shape}.", nameof(firstFrameLatent));

        int steps = request.Steps > 0 ? request.Steps : _config.NumInferenceSteps;
        float guidance = request.CfgScale > 0 ? request.CfgScale : _config.GuidanceScale;
        float shift = _config.FlowShift;

        string mode = firstFrameLatent is null ? "T2V" : "I2V";
        Logs.Info($"Wan-Video {mode}: {numFrames}f {request.Width}x{request.Height}, {steps} steps, cfg={guidance}, seed={seed} (latent {_config.InChannels}x{tLat}x{hLat}x{wLat}, shift={shift})");
        Logs.Warning("Wan-Video pipeline is first-run-validation pending — numerics unverified vs the reference checkpoint.");

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        if (_transformer2 is not null) Backend.PreloadWeights(_transformer2.EnumerateWeights());
        // T2V/TI2V denoise in VAE-latent space; the latent channel count is the VAE z, not the (possibly larger,
        // I2V-concat) transformer in_channels.
        int latentCh = _config.VaeLatentChannels;
        Tensor latents = SeedGenerator.CreateNoise(new TensorShape([1L, latentCh, tLat, hLat, wLat]), seed);
        float[] tsteps = LancePipelineCommon.BuildShiftedTimesteps(steps, shift);
        float[]? frameTs = firstFrameLatent is null ? null : new float[tLat];

        for (int k = 0; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float t = tsteps[k], dt = t - tsteps[k + 1];
            float tEmb = t * 1000f;   // DiT timestep scaling (validation-gated)
            WanVideoTransformer expert = Expert(tEmb);
            Tensor vCond, vUncond;
            if (firstFrameLatent is null)
            {
                vCond = expert.Forward(Backend, latents, promptEmbeds, tEmb);
                vUncond = expert.Forward(Backend, latents, negativeEmbeds, tEmb);
            }
            else
            {
                // Model input: condition on frame 0 (timestep 0), evolving noise elsewhere (timestep t).
                Tensor modelInput = CloneLatents(latents);
                WriteFirstFrame(modelInput, firstFrameLatent);
                frameTs![0] = 0f;
                for (int f = 1; f < tLat; f++) frameTs[f] = tEmb;
                vCond = expert.Forward(Backend, modelInput, promptEmbeds, frameTs);
                vUncond = expert.Forward(Backend, modelInput, negativeEmbeds, frameTs);
                modelInput.Dispose();
            }
            LancePipelineCommon.EulerCfgStep(latents, vCond, vUncond, guidance, dt);
            vCond.Dispose();
            vUncond.Dispose();
            sw.Stop();
            // Latent is a borrowed reference for preview encoders (latent2rgb decodes the middle frame).
            onProgress?.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds)
            {
                Latent = latents,
                LatentArch = LatentArchitecture.Wan,
            });
        }

        if (firstFrameLatent is not null) WriteFirstFrame(latents, firstFrameLatent);

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());
        if (_transformer2 is not null) Backend.FreeWeights(_transformer2.EnumerateWeights());
        return latents;
    }

    /// <summary>Wan concat-conditioned I2V: denoises a z-channel latent while the model input each step is the
    /// concat of <c>[noisy(z), mask(tp), cond-latent(z)]</c> (e.g. 16+4+16 = 36). Serves both Wan2.1 I2V-14B (pass
    /// <paramref name="imageEmbeds"/> = the CLIP-ViT-H penultimate hidden state <c>[seqImg, imageDim]</c> for the
    /// per-block image cross-attention) and Wan2.2 I2V-A14B (no CLIP — pass <paramref name="imageEmbeds"/> = null;
    /// conditioning is purely the concatenated cond-latent + mask). <paramref name="condRgb24"/> is the interleaved-RGB
    /// first frame. Honors the MoE expert switch when a second expert was supplied.</summary>
    public (byte[][] frames, int width, int height, int seed) GenerateImageToVideoConcat(
        Tensor promptEmbeds, Tensor negativeEmbeds, Tensor? imageEmbeds, ReadOnlySpan<byte> condRgb24,
        TextToImageRequest request, int numFrames, Action<GenerationProgress>? onProgress = null,
        byte[]? lastRgb24 = null)
    {
        ThrowIfDisposed();
        if (_encoder is null)
            throw new InvalidOperationException("Wan I2V needs a Wan VAE encoder for the conditioning latent.");
        int latentCh = _config.VaeLatentChannels;
        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        if (_config.InChannels != 2 * latentCh + tp)
            throw new InvalidOperationException(
                $"Concat I2V expects InChannels == 2·z + tp ({2 * latentCh + tp}); got {_config.InChannels}. "
                + "Use a Wan I2V config preset (e.g. I2V_14B_480p / I2V_A14B), or the TI2V firstFrameLatent path.");
        if (_config.HasImageConditioning && imageEmbeds is null)
            throw new ArgumentException("This variant has CLIP image conditioning — imageEmbeds is required.", nameof(imageEmbeds));
        if (request.Width % sp != 0 || request.Height % sp != 0)
            throw new ArgumentException($"Width/height must be divisible by {sp} for Wan-Video.");
        if (numFrames < 1 || (numFrames - 1) % tp != 0)
            throw new ArgumentException($"num_frames must satisfy (num_frames-1) % {tp} == 0; got {numFrames}.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int tLat = (numFrames - 1) / tp + 1;
        int hLat = request.Height / sp, wLat = request.Width / sp;
        int steps = request.Steps > 0 ? request.Steps : _config.NumInferenceSteps;
        float guidance = request.CfgScale > 0 ? request.CfgScale : _config.GuidanceScale;
        float shift = _config.FlowShift;

        Logs.Info($"Wan-Video I2V ({(imageEmbeds is null ? "concat" : "CLIP")}): {numFrames}f {request.Width}x{request.Height}, " +
            $"{steps} steps, cfg={guidance}, seed={seed} (latent {latentCh}x{tLat}x{hLat}x{wLat}, shift={shift})");
        Logs.Warning("Wan-Video I2V pipeline is first-run-validation pending — numerics unverified vs the reference checkpoint.");

        // Build the fixed [1, 20, T, H, W] conditioning: [mask(4), cond-latent(16)]. The first frame is VAE-encoded
        // into latent frame 0; the remaining frames are zero, and the mask marks frame 0 as known.
        Backend.PreloadWeights(_encoder.EnumerateWeights());
        Tensor frame0 = _encoder.EncodeRgbFrame(Backend, condRgb24, request.Width, request.Height);   // [1,z,1,hLat,wLat]
        Tensor? frameLast = lastRgb24 is not null
            ? _encoder.EncodeRgbFrame(Backend, lastRgb24, request.Width, request.Height) : null;
        Backend.Sync();
        Backend.FreeWeights(_encoder.EnumerateWeights());
        Tensor condition = BuildI2VCondition(frame0, frameLast, latentCh, tp, tLat, hLat, wLat);      // [1,tp+z,tLat,hLat,wLat]
        frame0.Dispose();
        frameLast?.Dispose();

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        if (_transformer2 is not null) Backend.PreloadWeights(_transformer2.EnumerateWeights());
        Tensor latents = SeedGenerator.CreateNoise(new TensorShape([1L, latentCh, tLat, hLat, wLat]), seed);
        float[] tsteps = LancePipelineCommon.BuildShiftedTimesteps(steps, shift);

        for (int k = 0; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float t = tsteps[k], dt = t - tsteps[k + 1];
            float tEmb = t * 1000f;
            WanVideoTransformer expert = Expert(tEmb);
            Tensor modelInput = ConcatChannels(latents, condition);   // [1, 2z+tp, tLat, hLat, wLat]
            Tensor vCond, vUncond;
            if (imageEmbeds is not null)
            {
                vCond = expert.Forward(Backend, modelInput, promptEmbeds, [tEmb], imageEmbeds);
                vUncond = expert.Forward(Backend, modelInput, negativeEmbeds, [tEmb], imageEmbeds);
            }
            else
            {
                vCond = expert.Forward(Backend, modelInput, promptEmbeds, [tEmb]);
                vUncond = expert.Forward(Backend, modelInput, negativeEmbeds, [tEmb]);
            }
            modelInput.Dispose();
            LancePipelineCommon.EulerCfgStep(latents, vCond, vUncond, guidance, dt);
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
        if (_transformer2 is not null) Backend.FreeWeights(_transformer2.EnumerateWeights());
        condition.Dispose();

        Tensor rgb;
        try { rgb = _vae.Decode(Backend, latents); }
        finally { latents.Dispose(); }
        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = FrameToBytes(rgb, i);
        rgb.Dispose();
        Logs.Info($"Wan-Video I2V complete ({frames.Length} frames, seed={seed})");
        return (frames, request.Width, request.Height, seed);
    }

    /// <summary>Video-to-video: VAE-encodes <paramref name="rgbClip"/> <c>[1, 3, T, H, W]</c>, noises it to the
    /// flow-match start determined by <paramref name="strength"/> (1 = full re-generation, &lt;1 = stay closer to the
    /// input), and runs the standard T2V denoise from that step (img2img-for-video). Honors the MoE expert switch.</summary>
    public (byte[][] frames, int width, int height, int seed) GenerateVideoToVideo(
        Tensor promptEmbeds, Tensor negativeEmbeds, Tensor rgbClip, float strength, TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        if (_encoder is null)
            throw new InvalidOperationException("Video2Video needs a Wan VAE encoder.");
        if (rgbClip.Shape.Rank != 5 || rgbClip.Shape[1] != 3)
            throw new ArgumentException($"rgbClip must be [1,3,T,H,W]; got {rgbClip.Shape}.", nameof(rgbClip));

        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        int pixT = (int)rgbClip.Shape[2], pixH = (int)rgbClip.Shape[3], pixW = (int)rgbClip.Shape[4];
        if (pixH % sp != 0 || pixW % sp != 0)
            throw new ArgumentException($"clip H/W must be divisible by {sp}.");
        if ((pixT - 1) % tp != 0)
            throw new ArgumentException($"clip frame count must satisfy (T-1) % {tp} == 0; got {pixT}.");
        strength = Math.Clamp(strength, 0f, 1f);

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int tLat = (pixT - 1) / tp + 1, hLat = pixH / sp, wLat = pixW / sp, latentCh = _config.VaeLatentChannels;
        int steps = request.Steps > 0 ? request.Steps : _config.NumInferenceSteps;
        float guidance = request.CfgScale > 0 ? request.CfgScale : _config.GuidanceScale;
        float shift = _config.FlowShift;
        int startStep = Math.Clamp((int)Math.Round((1f - strength) * steps), 0, steps - 1);

        Logs.Info($"Wan-Video V2V: {pixT}f {pixW}x{pixH}, strength={strength}, start step {startStep}/{steps}, " +
            $"cfg={guidance}, seed={seed} (latent {latentCh}x{tLat}x{hLat}x{wLat}, shift={shift})");
        Logs.Warning("Wan-Video V2V pipeline is first-run-validation pending — numerics unverified vs the reference checkpoint.");

        Backend.PreloadWeights(_encoder.EnumerateWeights());
        Tensor real = _encoder.Encode(Backend, rgbClip);                 // [1, z, tLat, hLat, wLat]
        Backend.Sync();
        Backend.FreeWeights(_encoder.EnumerateWeights());

        float[] tsteps = LancePipelineCommon.BuildShiftedTimesteps(steps, shift);
        float sigma0 = tsteps[startStep];
        Tensor noise = SeedGenerator.CreateNoise(new TensorShape([1L, latentCh, tLat, hLat, wLat]), seed);
        Tensor latents = new Tensor(real.Shape, DType.F32);
        long n = real.Shape.ElementCount;
        float* rp = (float*)real.DataPointer, np = (float*)noise.DataPointer, lp = (float*)latents.DataPointer;
        for (long i = 0; i < n; i++) lp[i] = (1f - sigma0) * rp[i] + sigma0 * np[i];   // flow-match noising
        real.Dispose(); noise.Dispose();

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        if (_transformer2 is not null) Backend.PreloadWeights(_transformer2.EnumerateWeights());
        for (int k = startStep; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float t = tsteps[k], dt = t - tsteps[k + 1], tEmb = t * 1000f;
            WanVideoTransformer expert = Expert(tEmb);
            Tensor vCond = expert.Forward(Backend, latents, promptEmbeds, tEmb);
            Tensor vUncond = expert.Forward(Backend, latents, negativeEmbeds, tEmb);
            LancePipelineCommon.EulerCfgStep(latents, vCond, vUncond, guidance, dt);
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
        if (_transformer2 is not null) Backend.FreeWeights(_transformer2.EnumerateWeights());

        Tensor rgb;
        try { rgb = _vae.Decode(Backend, latents); }
        finally { latents.Dispose(); }
        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = FrameToBytes(rgb, i);
        rgb.Dispose();
        Logs.Info($"Wan-Video V2V complete ({frames.Length} frames, seed={seed})");
        return (frames, pixW, pixH, seed);
    }

    /// <summary>Builds the <c>[1, tp+z, T, H, W]</c> I2V conditioning <c>[mask(tp), cond-latent(z)]</c>: the VAE-encoded
    /// first frame occupies latent frame 0 (and <paramref name="frameLast"/>, when given, the last latent frame for
    /// first-last-frame I2V), the rest are zero; the mask channels are 1 on the known frame(s) and 0 elsewhere — a
    /// structural stand-in for diffusers' temporal mask interleave (validation-gated).</summary>
    private static Tensor BuildI2VCondition(Tensor frame0, Tensor? frameLast, int latentCh, int temporalFactor,
        int tLat, int hLat, int wLat)
    {
        int maskCh = temporalFactor;
        int condCh = maskCh + latentCh;
        Tensor o = new Tensor(new TensorShape([1L, condCh, tLat, hLat, wLat]), DType.F32);
        float* op = (float*)o.DataPointer;
        long frame = (long)hLat * wLat;
        long perChannel = (long)tLat * frame;
        new Span<float>(op, (int)((long)condCh * perChannel)).Clear();
        int lastFrame = tLat - 1;
        // Mask channels: 1 at latent frame 0 (and the last latent frame for FLF2V).
        for (int m = 0; m < maskCh; m++)
        {
            for (long p = 0; p < frame; p++) op[(long)m * perChannel + p] = 1f;
            if (frameLast is not null)
                for (long p = 0; p < frame; p++) op[(long)m * perChannel + (long)lastFrame * frame + p] = 1f;
        }
        // Cond-latent channels: first-frame latent at frame 0, optional last-frame latent at the last frame.
        float* fp = (float*)frame0.DataPointer;   // [1, latentCh, 1, hLat, wLat]
        for (int c = 0; c < latentCh; c++)
            Buffer.MemoryCopy(fp + (long)c * frame, op + ((long)(maskCh + c) * perChannel), frame * 4, frame * 4);
        if (frameLast is not null)
        {
            float* lp = (float*)frameLast.DataPointer;
            for (int c = 0; c < latentCh; c++)
                Buffer.MemoryCopy(lp + (long)c * frame,
                    op + ((long)(maskCh + c) * perChannel + (long)lastFrame * frame), frame * 4, frame * 4);
        }
        return o;
    }

    /// <summary>Channel-concatenates two <c>[1, C, T, H, W]</c> tensors → <c>[1, Ca+Cb, T, H, W]</c>.</summary>
    private static Tensor ConcatChannels(Tensor a, Tensor b)
    {
        int ca = (int)a.Shape[1], cb = (int)b.Shape[1];
        int t = (int)a.Shape[2], h = (int)a.Shape[3], w = (int)a.Shape[4];
        long perChannel = (long)t * h * w;
        Tensor o = new Tensor(new TensorShape([1L, ca + cb, t, h, w]), DType.F32);
        float* op = (float*)o.DataPointer;
        Buffer.MemoryCopy((float*)a.DataPointer, op, (long)ca * perChannel * 4, (long)ca * perChannel * 4);
        Buffer.MemoryCopy((float*)b.DataPointer, op + (long)ca * perChannel, (long)cb * perChannel * 4, (long)cb * perChannel * 4);
        return o;
    }

    private static Tensor CloneLatents(Tensor latents)
    {
        Tensor o = new Tensor(latents.Shape, DType.F32);
        long bytes = latents.Shape.ElementCount * 4;
        Buffer.MemoryCopy((float*)latents.DataPointer, (float*)o.DataPointer, bytes, bytes);
        return o;
    }

    /// <summary>Overwrites latent frame 0 of <paramref name="latents"/> <c>[1,C,T,H,W]</c> with <paramref name="condition"/> <c>[1,C,1,H,W]</c>.</summary>
    private static void WriteFirstFrame(Tensor latents, Tensor condition)
    {
        int c = (int)latents.Shape[1], t = (int)latents.Shape[2];
        long frame = latents.Shape[3] * latents.Shape[4];
        float* lp = (float*)latents.DataPointer;
        float* cp = (float*)condition.DataPointer;
        for (int ci = 0; ci < c; ci++)
            Buffer.MemoryCopy(cp + ci * frame, lp + (long)ci * t * frame, frame * 4, frame * 4);
    }

    private static byte[] FrameToBytes(Tensor rgb, int frameIndex) => VideoRgbFrames.ExtractFrame(rgb, frameIndex);

    private static byte[] FrameToBytesGroup(Tensor group, int frameIndex) => VideoRgbFrames.ExtractFrame(group, frameIndex);
}
