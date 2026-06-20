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

/// <summary>Wan VACE (control/editing) pipeline. Drives the <see cref="WanVaceTransformer"/>: the control video is
/// VAE-encoded and assembled into the <c>VaceInChannels</c> control context (inactive + reactive latents + a
/// space-to-depth mask), which conditions every denoise step alongside the umT5 text. Flow-match Euler + 2-way text
/// CFG; reuses the Wan VAE (z=16) decoder/encoder.
///
/// <para><b>Status:</b> structural — the exact inactive/reactive split and mask layout are validation-gated (this build
/// uses the control latent for both halves and a constant-fill mask); numerics unverified vs the real checkpoint.</para></summary>
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
    /// are uniform here).</summary>
    public (byte[][] frames, int width, int height, int seed) GenerateFromControl(
        Tensor promptEmbeds, Tensor negativeEmbeds, Tensor controlRgbClip, TextToImageRequest request,
        float controlScale = 1.0f, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        int pixT = (int)controlRgbClip.Shape[2], pixH = (int)controlRgbClip.Shape[3], pixW = (int)controlRgbClip.Shape[4];
        if (pixH % sp != 0 || pixW % sp != 0)
            throw new ArgumentException($"control H/W must be divisible by {sp}.");
        if ((pixT - 1) % tp != 0)
            throw new ArgumentException($"control frame count must satisfy (T-1) % {tp} == 0; got {pixT}.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int tLat = (pixT - 1) / tp + 1, hLat = pixH / sp, wLat = pixW / sp, latentCh = _config.VaeLatentChannels;
        int steps = request.Steps > 0 ? request.Steps : _config.NumInferenceSteps;
        float guidance = request.CfgScale > 0 ? request.CfgScale : _config.GuidanceScale;
        float shift = _config.FlowShift;
        float[] scales = new float[_config.VaceLayers.Length];
        for (int i = 0; i < scales.Length; i++) scales[i] = controlScale;

        Logs.Info($"Wan-Video VACE: {pixT}f {pixW}x{pixH}, {steps} steps, cfg={guidance}, control_scale={controlScale}, " +
            $"seed={seed} (latent {latentCh}x{tLat}x{hLat}x{wLat})");
        Logs.Warning("Wan-Video VACE pipeline is first-run-validation pending — control composition + numerics unverified.");

        // Build the [1, VaceInChannels, tLat, hLat, wLat] control context from the VAE-encoded control video.
        Backend.PreloadWeights(_encoder.EnumerateWeights());
        Tensor controlLatent = _encoder.Encode(Backend, controlRgbClip);   // [1, z, tLat, hLat, wLat]
        Backend.Sync();
        Backend.FreeWeights(_encoder.EnumerateWeights());
        Tensor control = BuildControlContext(controlLatent, latentCh, _config.VaceInChannels, tLat, hLat, wLat);
        controlLatent.Dispose();

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        Tensor latents = SeedGenerator.CreateNoise(new TensorShape([1L, latentCh, tLat, hLat, wLat]), seed);
        float[] tsteps = LancePipelineCommon.BuildShiftedTimesteps(steps, shift);

        for (int k = 0; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float t = tsteps[k], dt = t - tsteps[k + 1], tEmb = t * 1000f;
            Tensor vCond = _transformer.Forward(Backend, latents, control, promptEmbeds, tEmb, scales);
            Tensor vUncond = _transformer.Forward(Backend, latents, control, negativeEmbeds, tEmb, scales);
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

    /// <summary>Assembles the <c>[1, VaceInChannels, T, H, W]</c> control context. Wan VACE uses
    /// <c>[inactive-latent(z), reactive-latent(z), space-to-depth mask(VaceInChannels−2z)]</c>; this structural build
    /// uses the control latent for both the inactive and reactive halves and fills the mask channels with 1 (validation-gated).</summary>
    private static Tensor BuildControlContext(Tensor controlLatent, int latentCh, int vaceInChannels, int tLat, int hLat, int wLat)
    {
        Tensor o = new Tensor(new TensorShape([1L, vaceInChannels, tLat, hLat, wLat]), DType.F32);
        float* op = (float*)o.DataPointer;
        float* cp = (float*)controlLatent.DataPointer;
        long perChannel = (long)tLat * hLat * wLat;
        // [0:z] inactive, [z:2z] reactive — both the control latent.
        for (int c = 0; c < latentCh; c++)
        {
            Buffer.MemoryCopy(cp + (long)c * perChannel, op + (long)c * perChannel, perChannel * 4, perChannel * 4);
            Buffer.MemoryCopy(cp + (long)c * perChannel, op + ((long)(latentCh + c) * perChannel), perChannel * 4, perChannel * 4);
        }
        // [2z:] mask channels filled with 1.
        for (long i = (long)(2 * latentCh) * perChannel; i < (long)vaceInChannels * perChannel; i++) op[i] = 1f;
        return o;
    }
}
