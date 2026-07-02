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

/// <summary>Wan VACE (control/editing) pipeline. Drives the <see cref="WanVaceTransformer"/>: the control video is
/// split by the pixel mask into inactive (<c>control·(1−m)</c>) and reactive (<c>control·m</c>) halves, each
/// VAE-encoded, and concatenated with the 8×8 space-to-depth pixel mask into the <c>VaceInChannels</c> control
/// context (16+16+64 = 96), which conditions every denoise step alongside the umT5 text — matching ComfyUI's
/// <c>WanVaceToVideo</c> + <c>WAN21_Vace.extra_conds</c>. UniPC + 2-way text CFG; reuses the Wan VAE (z=16).
///
/// <para><b>Status:</b> reference-image (identity) conditioning is not modeled — control-video/mask only.</para></summary>
public sealed unsafe class WanVacePipeline : DiffusionPipelineBase
{
    private readonly WanVaceTransformer _transformer;
    private readonly IWanVaeDecoder _vae;
    private readonly IWanVaeEncoder _encoder;
    private readonly WanVideoConfig _config;

    public WanVacePipeline(IBackend backend, WanVaceTransformer transformer, IWanVaeDecoder vae, IWanVaeEncoder encoder,
        WanVideoConfig config)
        : base(backend)
    {
        _transformer = transformer;
        _vae = vae;
        _encoder = encoder;
        _config = config;
    }

    /// <summary>Generates frames conditioned on a control video. <paramref name="controlRgbClip"/> is
    /// <c>[1, 3, T, H, W]</c> in [-1, 1]; <paramref name="controlScale"/> scales every VACE hint (the per-layer scales
    /// are uniform here). <paramref name="controlMask"/> is an optional <c>[T, H, W]</c> pixel mask in [0, 1] splitting
    /// the control into inactive (keep, <c>1−m</c>) and reactive (control, <c>m</c>) regions; null = all-ones (pure
    /// control mode: the whole frame is driven by the control video).</summary>
    public (byte[][] frames, int width, int height, int seed) GenerateFromControl(
        Tensor promptEmbeds, Tensor negativeEmbeds, Tensor controlRgbClip, TextToImageRequest request,
        float controlScale = 1.0f, Action<GenerationProgress>? onProgress = null, Tensor? controlMask = null)
    {
        ThrowIfDisposed();
        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        int pixT = (int)controlRgbClip.Shape[2], pixH = (int)controlRgbClip.Shape[3], pixW = (int)controlRgbClip.Shape[4];
        if (pixH % sp != 0 || pixW % sp != 0)
            throw new ArgumentException($"control H/W must be divisible by {sp}.");
        if ((pixT - 1) % tp != 0)
            throw new ArgumentException($"control frame count must satisfy (T-1) % {tp} == 0; got {pixT}.");
        if (controlMask is not null &&
            (controlMask.Shape.Rank != 3 || controlMask.Shape[0] != pixT || controlMask.Shape[1] != pixH || controlMask.Shape[2] != pixW))
            throw new ArgumentException($"controlMask must be [{pixT},{pixH},{pixW}]; got {controlMask.Shape}.", nameof(controlMask));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int tLat = (pixT - 1) / tp + 1, hLat = pixH / sp, wLat = pixW / sp, latentCh = _config.VaeLatentChannels;
        int steps = request.Steps ?? _config.NumInferenceSteps;
        float guidance = request.CfgScale ?? _config.GuidanceScale;
        float shift = _config.FlowShift;
        float[] scales = new float[_config.VaceLayers.Length];
        for (int i = 0; i < scales.Length; i++) scales[i] = controlScale;

        int maskCh = _config.VaceInChannels - 2 * latentCh;
        if (maskCh < 0)
            throw new InvalidOperationException($"VaceInChannels ({_config.VaceInChannels}) must be ≥ 2·z ({2 * latentCh}).");
        // The space-to-depth mask layout requires maskCh == sp² (real VACE: 96 = 2·16 + 8²); a null mask just fills 1s.
        if (controlMask is not null && maskCh != sp * sp)
            throw new InvalidOperationException(
                $"masked VACE expects VaceInChannels == 2·z + sp² ({2 * latentCh + sp * sp}); got {_config.VaceInChannels}.");

        Logs.Info($"Wan-Video VACE: {pixT}f {pixW}x{pixH}, {steps} steps, cfg={guidance}, control_scale={controlScale}, " +
            $"seed={seed} (latent {latentCh}x{tLat}x{hLat}x{wLat})");

        // Split the control clip by the pixel mask (inactive = control·(1−m), reactive = control·m — the [-1,1]-space
        // equivalent of ComfyUI's (c01−0.5)·(1−m)+0.5 / ·m+0.5), VAE-encode each half, and append the 8×8
        // space-to-depth mask channels → [1, 2z+sp², tLat, hLat, wLat].
        Tensor inactivePix = MaskedControl(controlRgbClip, controlMask, inactive: true);
        Tensor reactivePix = MaskedControl(controlRgbClip, controlMask, inactive: false);
        Backend.PreloadWeights(_encoder.EnumerateWeights());
        Tensor inactiveLat = _encoder.Encode(Backend, inactivePix);   // [1, z, tLat, hLat, wLat]
        Tensor reactiveLat = _encoder.Encode(Backend, reactivePix);
        Backend.Sync();
        Backend.FreeWeights(_encoder.EnumerateWeights());
        inactivePix.Dispose();
        reactivePix.Dispose();
        Tensor control = BuildControlContext(inactiveLat, reactiveLat, controlMask, latentCh, maskCh, sp, pixT, tLat, hLat, wLat);
        inactiveLat.Dispose();
        reactiveLat.Dispose();

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        Tensor latents = SeedGenerator.CreateNoise(new TensorShape([1L, latentCh, tLat, hLat, wLat]), seed);
        // VALIDATION-PENDING: Wan 2.2 UniPC scheduler (solver_order=2, bh2, predict_x0, flow sigmas, exponential
        // shift) — verify the VACE control path's UniPC trajectory vs the diffusers Wan VACE pipeline.
        FlowUniPCMultistepScheduler scheduler = new(solverOrder: 2);
        scheduler.SetTimesteps(steps, shift);

        for (int k = 0; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float tEmb = scheduler.Timesteps[k];
            Tensor vCond = _transformer.Forward(Backend, latents, control, promptEmbeds, tEmb, scales);
            Tensor vUncond = _transformer.Forward(Backend, latents, control, negativeEmbeds, tEmb, scales);
            LancePipelineCommon.CfgCombineRenormInPlace(vCond, vUncond, guidance, _config.CfgRescale);
            scheduler.Step(latents, vCond);
            vCond.Dispose(); vUncond.Dispose();
            sw.Stop();
            onProgress?.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds)
            {
                Latent = latents,
                LatentArch = LatentArchitecture.Wan,
            });
            // Reclaim GPU-resident activation buffers between steps — the 14B DiT's intermediates accumulate to OOM
            // over a multi-step run otherwise (the latent is host-side, so nothing cross-step is lost).
            Backend.FreeActivations();
        }

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());
        control.Dispose();

        Tensor rgb;
        try { rgb = _vae.Decode(Backend, latents); }
        finally { latents.Dispose(); }
        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = VideoRgbFrames.ExtractFrame(rgb, i);
        rgb.Dispose();
        Logs.Info($"Wan-Video VACE complete ({frames.Length} frames, seed={seed})");
        return (frames, pixW, pixH, seed);
    }

    /// <summary>Applies the pixel mask to the [-1,1] control clip: inactive = <c>control·(1−m)</c>, reactive =
    /// <c>control·m</c>. A null mask means m=1 everywhere (inactive = gray/zeros, reactive = the control clip).</summary>
    private static Tensor MaskedControl(Tensor controlRgbClip, Tensor? mask, bool inactive)
    {
        Tensor o = new Tensor(controlRgbClip.Shape, DType.F32);
        float* op = (float*)o.DataPointer, cp = (float*)controlRgbClip.DataPointer;
        long t = controlRgbClip.Shape[2];
        long frame = controlRgbClip.Shape[3] * controlRgbClip.Shape[4];
        if (mask is null)
        {
            long n = controlRgbClip.Shape.ElementCount;
            if (inactive) new Span<float>(op, (int)n).Clear();
            else Buffer.MemoryCopy(cp, op, n * 4, n * 4);
            return o;
        }
        float* mp = (float*)mask.DataPointer;   // [T, H, W]
        for (int c = 0; c < 3; c++)
            for (long ti = 0; ti < t; ti++)
                for (long p = 0; p < frame; p++)
                {
                    float m = mp[ti * frame + p];
                    long idx = ((long)c * t + ti) * frame + p;
                    op[idx] = cp[idx] * (inactive ? 1f - m : m);
                }
        return o;
    }

    /// <summary>Assembles the <c>[1, 2z+sp², T_lat, H_lat, W_lat]</c> control context
    /// <c>[inactive-latent(z), reactive-latent(z), space-to-depth mask(sp²)]</c>. Mask channel <c>dy·sp+dx</c> holds
    /// pixel <c>m[t, y·sp+dy, x·sp+dx]</c> with the frame axis nearest-exact-resampled T→T_lat (ComfyUI
    /// WanVaceToVideo); a null mask fills the mask channels with 1.</summary>
    private static Tensor BuildControlContext(Tensor inactiveLat, Tensor reactiveLat, Tensor? mask, int latentCh,
        int maskCh, int sp, int pixT, int tLat, int hLat, int wLat)
    {
        int condCh = 2 * latentCh + maskCh;
        Tensor o = new Tensor(new TensorShape([1L, condCh, tLat, hLat, wLat]), DType.F32);
        float* op = (float*)o.DataPointer;
        float* ip = (float*)inactiveLat.DataPointer, rp = (float*)reactiveLat.DataPointer;
        long frame = (long)hLat * wLat;
        long perChannel = (long)tLat * frame;
        for (int c = 0; c < latentCh; c++)
        {
            Buffer.MemoryCopy(ip + (long)c * perChannel, op + (long)c * perChannel, perChannel * 4, perChannel * 4);
            Buffer.MemoryCopy(rp + (long)c * perChannel, op + ((long)(latentCh + c) * perChannel), perChannel * 4, perChannel * 4);
        }
        float* maskBase = op + (long)(2 * latentCh) * perChannel;
        if (mask is null)
        {
            for (long i = 0; i < (long)maskCh * perChannel; i++) maskBase[i] = 1f;
            return o;
        }
        float* mp = (float*)mask.DataPointer;   // [pixT, hLat·sp, wLat·sp]
        long pixFrame = (long)hLat * sp * wLat * sp;
        for (int dy = 0; dy < sp; dy++)
            for (int dx = 0; dx < sp; dx++)
            {
                float* ch = maskBase + (long)(dy * sp + dx) * perChannel;
                for (int ti = 0; ti < tLat; ti++)
                {
                    int src = Math.Min(pixT - 1, (int)(((long)ti * 2 + 1) * pixT / (2L * tLat)));   // nearest-exact
                    float* mf = mp + (long)src * pixFrame;
                    for (int y = 0; y < hLat; y++)
                        for (int x = 0; x < wLat; x++)
                            ch[((long)ti * hLat + y) * wLat + x] = mf[((long)y * sp + dy) * wLat * sp + (long)x * sp + dx];
                }
            }
        return o;
    }
}
