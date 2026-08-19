using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Video.Pipelines;

/// <summary>Wan2.2-Animate (character animation) pipeline, following ComfyUI's <c>WanAnimateToVideo</c> node +
/// <c>WAN22_Animate.extra_conds</c>. The reference image is VAE-encoded into an extra leading latent frame
/// (<c>trim_latent</c>, removed before decode); the generated frames are represented by a VAE-encoded mid-gray clip in
/// the concat conditioning. Each step's model input is <c>[noisy(z), mask(4), cond-latent(z)]</c> where the mask (in
/// model space, after the WAN21 <c>1 − mask</c> inversion) is 1 on the reference frame and 0 on generated frames.
/// The pose video is VAE-encoded and passed to the DiT (added
/// in-DiT to latent frames 1..); the face video ([-1,1], 512×512 pixels) drives the motion-encoder → face-encoder →
/// face-adapter pathway — the negative CFG branch gets black face frames (<c>face·0 − 1</c>), per the node.
///
/// <para><b>Not modeled (structural gaps, flagged):</b> <c>continue_motion</c> chunked extension,
/// <c>background_video</c>/<c>character_mask</c> replace-mode conditioning. Numerics validation-pending vs the real
/// Wan22Animate checkpoint (fp8-scaled, Kijai/WanVideo_comfy_fp8_scaled).</para></summary>
public sealed unsafe class WanAnimatePipeline : DiffusionPipelineBase
{
    private readonly WanAnimateTransformer _transformer;
    private readonly IWanVaeDecoder _vae;
    private readonly IWanVaeEncoder _encoder;
    private readonly WanVideoConfig _config;

    public WanAnimatePipeline(IBackend backend, WanAnimateTransformer transformer, IWanVaeDecoder vae,
        IWanVaeEncoder encoder, WanVideoConfig config)
        : base(backend)
    {
        _transformer = transformer;
        _vae = vae;
        _encoder = encoder;
        _config = config;
    }

    /// <summary>Generates an animated clip. <paramref name="referenceRgb"/> is the character reference image
    /// <c>[1, 3, 1, H, W]</c> in [-1, 1]; <paramref name="poseRgbClip"/> is the driving pose/skeleton video
    /// <c>[1, 3, T, H, W]</c>; <paramref name="faceRgbClip"/> is the driving face video <c>[1, 3, Tface, 512, 512]</c>
    /// in [-1, 1] (already cropped/resized to the motion-encoder resolution); <paramref name="clipImageEmbeds"/> is
    /// the optional CLIP-ViT-H penultimate hidden state of the reference image. Output length follows the pose clip.</summary>
    public (byte[][] frames, int width, int height, int seed, WanAnimateConditioning conditioning) GenerateAnimation(
        Tensor promptEmbeds, Tensor negativeEmbeds, Tensor? referenceRgb, Tensor? poseRgbClip, Tensor? faceRgbClip,
        TextToImageRequest request, Tensor? clipImageEmbeds = null, WanAnimateConditioning? cachedConditioning = null,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        int latentCh = _config.VaeLatentChannels;
        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        if (_config.InChannels != 2 * latentCh + tp)
            throw new InvalidOperationException(
                $"Wan-Animate expects InChannels == 2·z + tp ({2 * latentCh + tp}); got {_config.InChannels}.");

        // Geometry: from the cached pose latent on a conditioning-cache hit, else from the pose pixel clip.
        bool haveCached = cachedConditioning is not null;
        int pixT, pixH, pixW, tLat, hLat, wLat;
        const int trimLatent = 1;   // the reference latent frame, removed before decode
        if (haveCached)
        {
            Tensor pl = cachedConditioning!.PoseLatent;   // [1, poseC, tLat, hLat, wLat]
            tLat = (int)pl.Shape[2]; hLat = (int)pl.Shape[3]; wLat = (int)pl.Shape[4];
            pixH = hLat * sp; pixW = wLat * sp; pixT = (tLat - 1) * tp + 1;
        }
        else
        {
            if (referenceRgb is null || poseRgbClip is null || faceRgbClip is null)
                throw new ArgumentException("Uncached Wan-Animate generation requires reference, pose, and face clips.");
            if (referenceRgb.Shape.Rank != 5 || referenceRgb.Shape[1] != 3 || referenceRgb.Shape[2] != 1)
                throw new ArgumentException($"referenceRgb must be [1,3,1,H,W]; got {referenceRgb.Shape}.", nameof(referenceRgb));
            pixT = (int)poseRgbClip.Shape[2]; pixH = (int)poseRgbClip.Shape[3]; pixW = (int)poseRgbClip.Shape[4];
            if (referenceRgb.Shape[3] != pixH || referenceRgb.Shape[4] != pixW)
                throw new ArgumentException($"reference image must match the pose video size {pixW}x{pixH}.", nameof(referenceRgb));
            if (pixH % sp != 0 || pixW % sp != 0)
                throw new ArgumentException($"pose H/W must be divisible by {sp}.");
            if ((pixT - 1) % tp != 0)
                throw new ArgumentException($"pose frame count must satisfy (T-1) % {tp} == 0; got {pixT}.");
            tLat = (pixT - 1) / tp + 1; hLat = pixH / sp; wLat = pixW / sp;
        }

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int tTotal = tLat + trimLatent;
        int steps = request.Steps ?? _config.NumInferenceSteps;
        float guidance = request.CfgScale ?? _config.GuidanceScale;
        float shift = (request as VideoGenerationRequest)?.FlowShift ?? _config.FlowShift;

        Logs.Info($"Wan-Animate: {pixT}f {pixW}x{pixH}, {steps} steps, cfg={guidance}, seed={seed} " +
            $"(latent {latentCh}x{tTotal}x{hLat}x{wLat}, ref-trim {trimLatent}{(haveCached ? ", cond CACHED" : "")})");
        // Animation mode validated end-to-end 2026-08-19 (real driving video, YOLO-pose auto-preprocess,
        // identity + motion both confirmed visually against the reference inputs).
        Logs.Warning("Wan-Animate: continue-motion and background/mask replace conditioning are not modeled yet.");

        WanAnimateConditioning conditioning;
        if (haveCached)
        {
            conditioning = cachedConditioning!;
            Backend.PreloadWeights(_transformer.EnumerateWeights());
        }
        else
        {
            // VAE-encode the conditioning: reference frame, pose video, and the mid-gray placeholder for generated frames.
            Backend.PreloadWeights(_encoder.EnumerateWeights());
            Tensor refLatent = _encoder.Encode(Backend, referenceRgb!);      // [1, z, 1, hLat, wLat]
            Tensor poseLatentDev = _encoder.Encode(Backend, poseRgbClip!);   // [1, z, tLat, hLat, wLat]
            Tensor gray = new Tensor(new TensorShape([1L, 3, pixT, pixH, pixW]), DType.F32);   // 0 in [-1,1] = mid gray
            new Span<float>((float*)gray.DataPointer, (int)gray.Shape.ElementCount).Clear();
            Tensor grayLatent = _encoder.Encode(Backend, gray);             // [1, z, tLat, hLat, wLat]
            gray.Dispose();
            Backend.Sync();
            Backend.FreeWeights(_encoder.EnumerateWeights());

            // Concat conditioning [1, 4+z, tTotal, hLat, wLat]: the node emits concat_mask 0 on the ref frame / 1 on
            // generated frames, and WAN21.concat_cond INVERTS it (mask = 1.0 - mask) before cat((mask, image)) — so the
            // model sees 1 on the known ref frame, 0 elsewhere. Cond-latent = [ref frame, gray frames].
            Tensor conditionDev = BuildAnimateCondition(refLatent, grayLatent, latentCh, tp, trimLatent, tTotal, hLat, wLat);
            refLatent.Dispose();
            grayLatent.Dispose();

            // Negative CFG branch drives the face pathway with black frames (face·0 − 1), per the node.
            Tensor faceNeg = new Tensor(faceRgbClip!.Shape, DType.F32);
            float* fnp = (float*)faceNeg.DataPointer;
            long fn = faceNeg.Shape.ElementCount;
            for (long i = 0; i < fn; i++) fnp[i] = -1f;

            Backend.PreloadWeights(_transformer.EnumerateWeights());
            // The face clip is constant across the denoise: encode motion ONCE per CFG branch (the StyleGAN motion
            // encoder is host-side; running it inside every forward made a 20-step 14B run take >30 min).
            Tensor motionCondDev = _transformer.EncodeMotion(Backend, faceRgbClip!, tTotal);
            Tensor motionUncondDev = _transformer.EncodeMotion(Backend, faceNeg, tTotal);
            faceNeg.Dispose();
            Backend.Sync();

            // Host-materialize the whole conditioning for cross-generation caching (small tensors — a few MB; they
            // re-fault to device inside the loop, exactly like the S2V reference-latent cache). Bit-identical to the
            // device originals, so the numerics are unchanged; the device copies are freed here.
            conditioning = new WanAnimateConditioning
            {
                Condition = HostCopy(conditionDev),
                PoseLatent = HostCopy(poseLatentDev),
                MotionCond = HostCopy(motionCondDev),
                MotionUncond = HostCopy(motionUncondDev),
            };
            conditionDev.Dispose();
            poseLatentDev.Dispose();
            motionCondDev.Dispose();
            motionUncondDev.Dispose();
        }

        Tensor condition = conditioning.Condition;
        Tensor poseLatent = conditioning.PoseLatent;
        Tensor motionCond = conditioning.MotionCond;
        Tensor motionUncond = conditioning.MotionUncond;

        Tensor latents = SeedGenerator.CreateNoise(new TensorShape([1L, latentCh, tTotal, hLat, wLat]), seed);
        // VALIDATION-PENDING: Wan 2.2 UniPC scheduler (solver_order=2, bh2, predict_x0, flow sigmas, exponential
        // shift) — verify the Animate trajectory vs ComfyUI at the configured step count.
        FlowUniPCMultistepScheduler scheduler = new(solverOrder: 2);
        scheduler.SetTimesteps(steps, shift);

        for (int k = 0; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float tEmb = scheduler.Timesteps[k];
            Tensor modelInput = ConcatChannels(latents, condition);      // [1, 2z+4, tTotal, hLat, wLat]
            Tensor vCond = _transformer.Forward(Backend, modelInput, poseLatent, null, promptEmbeds, tEmb, clipImageEmbeds, motion: motionCond);
            Tensor vUncond = _transformer.Forward(Backend, modelInput, poseLatent, null, negativeEmbeds, tEmb, clipImageEmbeds, motion: motionUncond);
            modelInput.Dispose();
            LancePipelineCommon.CfgCombineRenormInPlace(vCond, vUncond, guidance, _config.CfgRescale);
            scheduler.Step(latents, vCond);
            vCond.Dispose(); vUncond.Dispose();
            sw.Stop();
            onProgress?.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds)
            {
                Latent = latents,
                LatentArch = LatentArchitecture.Wan,
            });
            Backend.FreeActivations();
        }

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());
        // conditioning (condition/poseLatent/motionCond/motionUncond) is owned by the caller's cross-generation
        // cache — never disposed here.

        // Trim the reference latent frame (trim_latent), then decode the generated frames.
        Tensor video = DropLeadingFrames(latents, latentCh, tTotal, trimLatent, hLat, wLat);
        latents.Dispose();
        Tensor rgb;
        try { rgb = _vae.Decode(Backend, video); }
        finally { video.Dispose(); }
        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = VideoRgbFrames.ExtractFrame(rgb, i);
        rgb.Dispose();
        Logs.Info($"Wan-Animate complete ({frames.Length} frames, seed={seed})");
        return (frames, pixW, pixH, seed, conditioning);
    }

    /// <summary>Fresh host-materialized copy of a (possibly device-resident) tensor, preserving shape and dtype —
    /// the cross-generation cache form that survives per-step activation sweeps and re-faults to device on use.</summary>
    private static Tensor HostCopy(Tensor x)
    {
        Tensor o = new Tensor(x.Shape, x.DType);
        long bytes = x.DType.ComputeByteCount(x.ElementCount);
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)o.DataPointer, bytes, bytes);
        return o;
    }

    /// <summary>Builds the <c>[1, tp+z, tTotal, H, W]</c> concat conditioning in model space (post the WAN21
    /// <c>1 − mask</c> inversion): mask channels 1 on the leading reference frame(s) / 0 on generated frames,
    /// cond-latent = reference latent then gray-clip latent.</summary>
    private static Tensor BuildAnimateCondition(Tensor refLatent, Tensor grayLatent, int latentCh, int maskCh,
        int trimLatent, int tTotal, int hLat, int wLat)
    {
        Tensor o = new Tensor(new TensorShape([1L, maskCh + latentCh, tTotal, hLat, wLat]), DType.F32);
        float* op = (float*)o.DataPointer;
        long frame = (long)hLat * wLat;
        long perChannel = (long)tTotal * frame;
        for (int m = 0; m < maskCh; m++)
            for (int t = 0; t < tTotal; t++)
            {
                float v = t < trimLatent ? 1f : 0f;
                for (long p = 0; p < frame; p++) op[(long)m * perChannel + t * frame + p] = v;
            }
        float* rp = (float*)refLatent.DataPointer;    // [1, z, trimLatent, h, w]
        float* gp = (float*)grayLatent.DataPointer;   // [1, z, tTotal-trimLatent, h, w]
        int tGen = tTotal - trimLatent;
        for (int c = 0; c < latentCh; c++)
        {
            long dstBase = (long)(maskCh + c) * perChannel;
            Buffer.MemoryCopy(rp + (long)c * trimLatent * frame, op + dstBase,
                (long)trimLatent * frame * 4, (long)trimLatent * frame * 4);
            Buffer.MemoryCopy(gp + (long)c * tGen * frame, op + dstBase + (long)trimLatent * frame,
                (long)tGen * frame * 4, (long)tGen * frame * 4);
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

    /// <summary>Drops the leading <paramref name="skip"/> latent frames → <c>[1, C, T−skip, H, W]</c>.</summary>
    private static Tensor DropLeadingFrames(Tensor x, int c, int t, int skip, int h, int w)
    {
        Tensor o = new Tensor(new TensorShape([1L, c, t - skip, h, w]), DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        long frame = (long)h * w;
        for (int ci = 0; ci < c; ci++)
            Buffer.MemoryCopy(xp + ((long)ci * t + skip) * frame, op + (long)ci * (t - skip) * frame,
                (long)(t - skip) * frame * 4, (long)(t - skip) * frame * 4);
        return o;
    }
}
