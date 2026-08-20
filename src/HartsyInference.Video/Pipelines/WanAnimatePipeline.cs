using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Runtime;
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
/// face-adapter pathway — the negative CFG branch gets black face frames (<c>face·0 − 1</c>), per the node, and runs
/// only above guidance 1.0 (upstream <c>wan/animate.py</c> takes a single forward at or below it).
///
/// <c>continue_motion</c> (the chunked extension) writes the previous chunk's last frames over the head of that gray
/// clip before the single VAE encode and marks the matching latent frames known in the mask; the chunk re-renders them
/// and the caller drops <c>trimImage</c> leading frames.
///
/// <para><b>Numerics validation-pending</b> vs the real Wan22Animate checkpoint (fp8-scaled,
/// Kijai/WanVideo_comfy_fp8_scaled).</para></summary>
public sealed unsafe class WanAnimatePipeline : DiffusionPipelineBase
{
    private readonly WanAnimateTransformer _transformer;
    private readonly IWanVaeDecoder _vae;
    private readonly IWanVaeEncoder _encoder;
    private readonly WanVideoConfig _config;

    /// <summary>Measured per-token activation slope of the Animate denoise loop (~0.64 MB/token). The binding
    /// constraint here is activations, not weights, so the streaming headroom has to scale with the geometry —
    /// a flat margin lets the resident prefix eat the VRAM the forward needs at exactly the sizes streaming exists
    /// for.</summary>
    private const long ActivationBytesPerToken = 671_089;

    /// <summary>Allowance for cuBLAS workspace, the prefetch window and pool slack, on top of the token-scaled term.</summary>
    private const long FixedHeadroomBytes = 1L << 30;

    public WanAnimatePipeline(IBackend backend, WanAnimateTransformer transformer, IWanVaeDecoder vae,
        IWanVaeEncoder encoder, WanVideoConfig config)
        : base(backend)
    {
        _transformer = transformer;
        _vae = vae;
        _encoder = encoder;
        _config = config;
    }

    /// <summary>True when the denoise must run the negative branch at all: upstream (<c>wan/animate.py</c>) folds CFG
    /// only above guidance 1.0, and at exactly 1.0 the fold is the identity.</summary>
    public static bool UsesCfgBranch(float guidance) => guidance > 1f;

    /// <summary>Generates an animated clip. <paramref name="referenceRgb"/> is the character reference image
    /// <c>[1, 3, 1, H, W]</c> in [-1, 1]; <paramref name="poseRgbClip"/> is the driving pose/skeleton video
    /// <c>[1, 3, T, H, W]</c>; <paramref name="faceRgbClip"/> is the driving face video <c>[1, 3, Tface, 512, 512]</c>
    /// in [-1, 1] (already cropped/resized to the motion-encoder resolution); <paramref name="clipImageEmbeds"/> is
    /// the optional CLIP-ViT-H penultimate hidden state of the reference image. Output length follows the pose clip.
    /// <paramref name="continueMotionRgbClip"/> is the previous chunk's last <c>N</c> decoded frames <c>[1,3,N,H,W]</c>
    /// in [-1, 1] (<c>N</c> on the <c>4n+1</c> grid and shorter than the clip); the returned <c>trimImage</c> is how
    /// many leading decoded frames the caller must discard because the chunk re-rendered them.</summary>
    public (byte[][] frames, int width, int height, int seed, WanAnimateConditioning conditioning, int trimImage) GenerateAnimation(
        Tensor promptEmbeds, Tensor negativeEmbeds, Tensor? referenceRgb, Tensor? poseRgbClip, Tensor? faceRgbClip,
        TextToImageRequest request, Tensor? clipImageEmbeds = null, WanAnimateConditioning? cachedConditioning = null,
        Tensor? backgroundRgbClip = null, Tensor? characterMaskClip = null, Tensor? continueMotionRgbClip = null,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        int latentCh = _config.VaeLatentChannels;
        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        if (_config.InChannels != 2 * latentCh + tp)
            throw new InvalidOperationException(
                $"Wan-Animate expects InChannels == 2·z + tp ({2 * latentCh + tp}); got {_config.InChannels}.");

        float guidance = request.CfgScale ?? _config.GuidanceScale;
        bool useCfg = UsesCfgBranch(guidance);

        // Geometry: from the cached pose latent on a conditioning-cache hit, else from the pose pixel clip.
        // A continuation chunk's conditioning is prefix-specific: never reuse a cache entry for one, and never serve
        // an entry that was built with a prefix, nor a single-pass entry when CFG needs the black-face features.
        bool haveCached = cachedConditioning is not null && continueMotionRgbClip is null
            && cachedConditioning.RefMotionLatentLength == 0
            && (!useCfg || cachedConditioning.MotionUncond is not null);
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

        int motionFrames = 0;
        if (continueMotionRgbClip is not null)
        {
            if (continueMotionRgbClip.Shape.Rank != 5 || continueMotionRgbClip.Shape[1] != 3)
                throw new ArgumentException($"continueMotionRgbClip must be [1,3,N,H,W]; got {continueMotionRgbClip.Shape}.",
                    nameof(continueMotionRgbClip));
            if (continueMotionRgbClip.Shape[3] != pixH || continueMotionRgbClip.Shape[4] != pixW)
                throw new ArgumentException($"the continue-motion prefix must be {pixW}x{pixH}; got "
                    + $"{continueMotionRgbClip.Shape[4]}x{continueMotionRgbClip.Shape[3]}.", nameof(continueMotionRgbClip));
            motionFrames = (int)continueMotionRgbClip.Shape[2];
            if (motionFrames >= pixT)
                throw new ArgumentException($"the continue-motion prefix ({motionFrames}f) must be shorter than the chunk "
                    + $"({pixT}f), else the chunk produces no new frames.", nameof(continueMotionRgbClip));
            if (motionFrames % WanAnimateChunkMath.TemporalStep != 1)
                throw new ArgumentException("the continue-motion prefix must be on the 4n+1 grid, else the trim under-drops "
                    + $"and the background overwrites part of it; got {motionFrames}f.", nameof(continueMotionRgbClip));
        }
        int refMotionLatentLength = WanAnimateChunkMath.RefMotionLatentLength(motionFrames);
        int trimImage = WanAnimateChunkMath.TrimImageFrames(refMotionLatentLength);

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int tTotal = tLat + trimLatent;
        int steps = request.Steps ?? _config.NumInferenceSteps;
        float shift = (request as VideoGenerationRequest)?.FlowShift ?? _config.FlowShift;

        Logs.Info($"Wan-Animate: {pixT}f {pixW}x{pixH}, {steps} steps, cfg={guidance}{(useCfg ? "" : " single forward")}, seed={seed} " +
            $"(latent {latentCh}x{tTotal}x{hLat}x{wLat}, ref-trim {trimLatent}{(haveCached ? ", cond CACHED" : "")}"
            + $"{(motionFrames > 0 ? $", continue-motion {motionFrames}f → {refMotionLatentLength} latent frame(s), trim {trimImage}f" : "")})");
        // Animation mode validated end-to-end 2026-08-19 (real driving video, YOLO-pose auto-preprocess,
        // identity + motion both confirmed visually against the reference inputs).

        WanAnimateConditioning conditioning;
        if (haveCached)
        {
            conditioning = cachedConditioning!;
        }
        else
        {
            // VAE-encode the conditioning: reference frame, pose video, and the mid-gray placeholder for generated frames.
            Backend.PreloadWeights(_encoder.EnumerateWeights());
            Tensor refLatent = _encoder.Encode(Backend, referenceRgb!);      // [1, z, 1, hLat, wLat]
            Tensor poseLatentDev = _encoder.Encode(Backend, poseRgbClip!);   // [1, z, tLat, hLat, wLat]
            Tensor gray = new Tensor(new TensorShape([1L, 3, pixT, pixH, pixW]), DType.F32);   // 0 in [-1,1] = mid gray
            new Span<float>((float*)gray.DataPointer, (int)gray.Shape.ElementCount).Clear();
            // Chunked extension: the previous chunk's tail goes over the HEAD of the gray clip, before the background,
            // and both are encoded in the SAME vae call — the Wan VAE is temporally causal, so encoding the prefix
            // separately and splicing latents is not equivalent.
            if (continueMotionRgbClip is not null)
            {
                OverlayClipFrames(gray, continueMotionRgbClip, pixT, pixH, pixW);
            }
            // Replacement mode: the generated-frames conditioning carries the background video instead of
            // mid-gray (ComfyUI background_video — frames past the clip's end stay gray). It starts at the first
            // non-prefix frame so the motion prefix survives.
            if (backgroundRgbClip is not null)
            {
                OverlayClipFrames(gray, backgroundRgbClip, pixT, pixH, pixW, startFrame: trimImage);
            }
            Tensor grayLatent = _encoder.Encode(Backend, gray);             // [1, z, tLat, hLat, wLat]
            gray.Dispose();
            Backend.Sync();
            Backend.FreeWeights(_encoder.EnumerateWeights());

            // Concat conditioning [1, 4+z, tTotal, hLat, wLat]: the node emits concat_mask 0 on the ref frame / 1 on
            // generated frames, and WAN21.concat_cond INVERTS it (mask = 1.0 - mask) before cat((mask, image)) — so the
            // model sees 1 on the known ref frame, 0 elsewhere. Cond-latent = [ref frame, gray frames].
            Tensor conditionDev = BuildAnimateCondition(refLatent, grayLatent, latentCh, tp, trimLatent, tTotal, hLat, wLat,
                characterMaskClip, refMotionLatentLength);
            refLatent.Dispose();
            grayLatent.Dispose();

            // Only the motion/face encoders — the DiT itself is placed after this, and preloading it here would
            // hold ~15 GB of block weights across an encode that never touches them.
            Backend.PreloadWeights(_transformer.EnumerateMotionEncoderWeights());
            // The face clip is constant across the denoise: encode motion ONCE per CFG branch (the StyleGAN motion
            // encoder is host-side; running it inside every forward made a 20-step 14B run take >30 min).
            Tensor motionCondDev = _transformer.EncodeMotion(Backend, faceRgbClip!, tTotal);
            Tensor? motionUncondDev = null;
            if (useCfg)
            {
                // Negative CFG branch drives the face pathway with black frames (face·0 − 1), per the node.
                Tensor faceNeg = new Tensor(faceRgbClip!.Shape, DType.F32);
                float* fnp = (float*)faceNeg.DataPointer;
                long fn = faceNeg.Shape.ElementCount;
                for (long i = 0; i < fn; i++) fnp[i] = -1f;
                motionUncondDev = _transformer.EncodeMotion(Backend, faceNeg, tTotal);
                faceNeg.Dispose();
            }
            Backend.Sync();
            Backend.FreeWeights(_transformer.EnumerateMotionEncoderWeights());
            Backend.TrimMemoryPool();

            // Host-materialize the whole conditioning for cross-generation caching (small tensors — a few MB; they
            // re-fault to device inside the loop, exactly like the S2V reference-latent cache). Bit-identical to the
            // device originals, so the numerics are unchanged; the device copies are freed here.
            conditioning = new WanAnimateConditioning
            {
                Condition = HostCopy(conditionDev),
                PoseLatent = HostCopy(poseLatentDev),
                MotionCond = HostCopy(motionCondDev),
                MotionUncond = motionUncondDev is null ? null : HostCopy(motionUncondDev),
                RefMotionLatentLength = refMotionLatentLength,
            };
            conditionDev.Dispose();
            poseLatentDev.Dispose();
            motionCondDev.Dispose();
            motionUncondDev?.Dispose();
        }

        Tensor condition = conditioning.Condition;
        Tensor poseLatent = conditioning.PoseLatent;
        Tensor motionCond = conditioning.MotionCond;
        Tensor? motionUncond = conditioning.MotionUncond;
        if (useCfg && motionUncond is null)
        {
            throw new InvalidOperationException(
                "Wan-Animate CFG needs the black-face motion features, but this conditioning was built single-pass.");
        }

        Tensor latents = SeedGenerator.CreateNoise(new TensorShape([1L, latentCh, tTotal, hLat, wLat]), seed);
        // VALIDATION-PENDING: Wan 2.2 UniPC scheduler (solver_order=2, bh2, predict_x0, flow sigmas, exponential
        // shift) — verify the Animate trajectory vs ComfyUI at the configured step count.
        FlowUniPCMultistepScheduler scheduler = new(solverOrder: 2);
        scheduler.SetTimesteps(steps, shift);

        (int pt2, int ph2, int pw2) = _config.PatchSize;
        long tokenLoad = (long)(tTotal / pt2) * (hLat / ph2) * (wLat / pw2);
        long headroomBytes = Math.Max(EnvSwitch.GetLong("HARTSY_ANIMATE_HEADROOM_MB", 3072) * 1024 * 1024,
            tokenLoad * ActivationBytesPerToken + FixedHeadroomBytes);
        BlockStreamingScope stream = BlockStreamingScope.Open(new BlockStreamingOptions
        {
            Backend = Backend,
            Denoiser = _transformer,
            ModelName = "Wan-Animate",
            HeadroomBytes = headroomBytes,
            TokenLoad = tokenLoad,
        });
        try
        {
            for (int k = 0; k < steps; k++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                float tEmb = scheduler.Timesteps[k];
                Tensor modelInput = ConcatChannels(latents, condition);      // [1, 2z+4, tTotal, hLat, wLat]
                Tensor vCond = _transformer.Forward(Backend, modelInput, poseLatent, null, promptEmbeds, tEmb, clipImageEmbeds, motion: motionCond);
                if (useCfg)
                {
                    Tensor vUncond = _transformer.Forward(Backend, modelInput, poseLatent, null, negativeEmbeds, tEmb, clipImageEmbeds, motion: motionUncond);
                    LancePipelineCommon.CfgCombineRenormInPlace(vCond, vUncond, guidance, _config.CfgRescale);
                    vUncond.Dispose();
                }
                modelInput.Dispose();
                scheduler.Step(latents, vCond);
                vCond.Dispose();
                sw.Stop();
                onProgress?.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds)
                {
                    Latent = latents,
                    LatentArch = LatentArchitecture.Wan,
                });
                Backend.FreeActivations();
                stream.EndStep();
            }
        }
        finally
        {
            Backend.Sync();
            stream.Dispose();
        }
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
        Logs.Info($"Wan-Animate complete ({frames.Length} frames, seed={seed}"
            + $"{(trimImage > 0 ? $", {trimImage} re-rendered prefix frame(s) for the caller to trim" : "")})");
        return (frames, pixW, pixH, seed, conditioning, trimImage);
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
    /// cond-latent = reference latent then gray-clip latent. A character mask (replacement mode) writes
    /// <c>1 − mask</c> per pixel over the generated frames — background regions become "known" (kept from the
    /// background conditioning) and only the character region generates. Mask channel <c>m</c> of latent frame
    /// <c>t</c> carries pixel frame <c>(t − trim)·4 + m</c>, upstream's straight <c>view(L,4).transpose</c>
    /// packing; a shorter mask clip clamps to its last frame (single-frame masks repeat). A non-zero
    /// <paramref name="refMotionLatentLength"/> additionally marks the chunked extension's motion prefix known over
    /// <see cref="WanAnimateChunkMath.MaskPrefixEnd"/> pixel frames.</summary>
    private static Tensor BuildAnimateCondition(Tensor refLatent, Tensor grayLatent, int latentCh, int maskCh,
        int trimLatent, int tTotal, int hLat, int wLat, Tensor? characterMaskClip = null, int refMotionLatentLength = 0)
    {
        Tensor o = new Tensor(new TensorShape([1L, maskCh + latentCh, tTotal, hLat, wLat]), DType.F32);
        float* op = (float*)o.DataPointer;
        long frame = (long)hLat * wLat;
        long perChannel = (long)tTotal * frame;
        int maskFrames = characterMaskClip is null ? 0 : (int)characterMaskClip.Shape[2];
        int maskH = characterMaskClip is null ? 0 : (int)characterMaskClip.Shape[3];
        int maskW = characterMaskClip is null ? 0 : (int)characterMaskClip.Shape[4];
        float* mp = characterMaskClip is null ? null : (float*)characterMaskClip.DataPointer;
        if (characterMaskClip is not null)
        {
            // Coverage log = the cheap sanity check that the mask decoded as intended (a mask that decodes
            // black silently degenerates to "keep everything" and looks like a model bug).
            long total = (long)maskFrames * maskH * maskW;
            double sum = 0; float mn = float.MaxValue, mx = float.MinValue;
            for (long i = 0; i < total; i++) { float v = mp[i]; sum += v; if (v < mn) mn = v; if (v > mx) mx = v; }
            Logs.Info($"[WanAnimate] character mask stats (clip [-1,1] space): min {mn:0.###} max {mx:0.###} " +
                $"mean {sum / total:0.###} over {maskFrames} frame(s) — white(1)=generate character, black(-1)=keep background.");
        }
        // Chunked extension: flat pixel-frame indices j < prefixEnd are "known" (the motion prefix). Upstream zeroes
        // [0, ref·4) and THEN overwrites [ref_images_num, …) with the character mask, so a mask shortens the known run
        // to ref_images_num — replicated rather than tidied, because this tensor is what the layer-diff compares.
        int prefixEnd = WanAnimateChunkMath.MaskPrefixEnd(refMotionLatentLength, characterMaskClip is not null);
        for (int m = 0; m < maskCh; m++)
            for (int t = 0; t < tTotal; t++)
            {
                long baseOff = (long)m * perChannel + t * frame;
                if (WanAnimateChunkMath.IsKnownMaskCell(t, m, trimLatent, prefixEnd))
                {
                    for (long p = 0; p < frame; p++) op[baseOff + p] = 1f;
                }
                else if (characterMaskClip is null)
                {
                    for (long p = 0; p < frame; p++) op[baseOff + p] = 0f;
                }
                else
                {
                    // Channel 0 of the RGB-decoded mask clip; nearest-neighbor down to the latent grid.
                    int pixFrame = Math.Min((t - trimLatent) * 4 + m, maskFrames - 1);
                    float* mf = mp + (long)pixFrame * maskH * maskW;   // clip layout [1,3,T,H,W], channel 0
                    for (int y = 0; y < hLat; y++)
                    {
                        int sy = Math.Min(maskH - 1, y * maskH / hLat);
                        for (int x = 0; x < wLat; x++)
                        {
                            int sx = Math.Min(maskW - 1, x * maskW / wLat);
                            // Clip pixels are [-1,1]; mask semantics want 0..1 (white=1=character).
                            float maskVal = Math.Clamp((mf[(long)sy * maskW + sx] + 1f) * 0.5f, 0f, 1f);
                            op[baseOff + (long)y * wLat + x] = 1f - maskVal;
                        }
                    }
                }
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

    /// <summary>Writes <paramref name="source"/>'s frames from <paramref name="startFrame"/> onward over
    /// <paramref name="target"/> at the same indices (both <c>[1, 3, T, H, W]</c>, same H/W); frames outside
    /// <c>[startFrame, min(srcT, pixT))</c> keep the target's values.</summary>
    private static void OverlayClipFrames(Tensor target, Tensor source, int pixT, int pixH, int pixW, int startFrame = 0)
    {
        int srcT = (int)source.Shape[2];
        if ((int)source.Shape[3] != pixH || (int)source.Shape[4] != pixW)
            throw new ArgumentException($"overlay clip must be {pixW}x{pixH}; got {source.Shape[4]}x{source.Shape[3]}.");
        int copyT = Math.Min(srcT, pixT) - startFrame;
        if (copyT <= 0)
        {
            return;
        }
        float* tp = (float*)target.DataPointer;
        float* sp = (float*)source.DataPointer;
        long frame = (long)pixH * pixW;
        for (int c = 0; c < 3; c++)
        {
            Buffer.MemoryCopy(sp + ((long)c * srcT + startFrame) * frame, tp + ((long)c * pixT + startFrame) * frame,
                copyT * frame * 4, copyT * frame * 4);
        }
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
