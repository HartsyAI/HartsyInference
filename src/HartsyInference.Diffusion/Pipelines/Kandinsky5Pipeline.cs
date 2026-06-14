using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Kandinsky 5.0 text-to-image pipeline (kandinskylab/Kandinsky-5.0-T2I-Lite).
///
/// This pipeline only handles the diffusion side — the dual text encoder stack (Qwen2.5-VL +
/// CLIP-L) is not yet implemented as a HartsyInference text encoder. Callers must supply the
/// pre-computed Qwen sequence embeddings <c>[B, S, 3584]</c> and the CLIP pooled embeddings
/// <c>[B, 768]</c>, plus their negative-prompt counterparts. This matches the diffusers
/// <c>Kandinsky5T2IPipeline.__call__(prompt_embeds_qwen=, prompt_embeds_clip=, ...)</c> escape
/// hatch — production users can pre-compute embeddings once with a sidecar Python helper, and
/// once a Qwen2.5-VL text encoder lands in HartsyInference this pipeline can grow a tokenizer-fed
/// overload without touching the denoising path.</summary>
public sealed unsafe class Kandinsky5Pipeline : DiffusionPipelineBase
{
    private readonly Kandinsky5Transformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly Kandinsky5Config _config;
    private readonly float _schedulerShift;
    private readonly float _vaeScalingFactor;
    private readonly float _vaeShiftFactor;

    /// <summary>Creates a new Kandinsky 5 pipeline. The VAE used for the Lite model is the Flux VAE
    /// (16-channel latent, 8× downsample), with shift/scale identical to <c>VaeConfig.Flux</c>.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="transformer">Kandinsky 5 transformer (use <see cref="Kandinsky5Config.Lite"/>).</param>
    /// <param name="vaeDecoder">Flux VAE decoder (16 channels).</param>
    /// <param name="config">Transformer configuration.</param>
    /// <param name="schedulerShift">Flow-match scheduler shift. Diffusers config: <c>shift=5.0</c>.</param>
    /// <param name="vaeScalingFactor">VAE scaling factor. Flux: <c>0.3611</c>.</param>
    /// <param name="vaeShiftFactor">VAE shift factor. Flux: <c>0.1159</c>.</param>
    public Kandinsky5Pipeline(IBackend backend, Kandinsky5Transformer transformer, VaeDecoder vaeDecoder,
        Kandinsky5Config config, float schedulerShift = 5.0f,
        float vaeScalingFactor = 0.3611f, float vaeShiftFactor = 0.1159f)
        : base(backend)
    {
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
        _schedulerShift = schedulerShift;
        _vaeScalingFactor = vaeScalingFactor;
        _vaeShiftFactor = vaeShiftFactor;
    }

    /// <summary>Generates an image from pre-computed text encoder outputs. Uses
    /// <c>FlowMatchEulerDiscreteScheduler</c> (shift defaults to 5.0 per the Lite scheduler config).</summary>
    /// <param name="qwenEmbeds">Qwen2.5-VL sequence embeddings <c>[B, S_t, in_text_dim]</c>.</param>
    /// <param name="clipPooled">CLIP-L pooled embeddings <c>[B, in_text_dim2]</c>.</param>
    /// <param name="negQwenEmbeds">Negative-prompt Qwen embeddings (only required if cfg &gt; 1).</param>
    /// <param name="negClipPooled">Negative-prompt CLIP pooled (only required if cfg &gt; 1).</param>
    /// <param name="request">Generation parameters.</param>
    /// <param name="onProgress">Optional progress callback.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor qwenEmbeds, Tensor clipPooled,
        Tensor? negQwenEmbeds, Tensor? negClipPooled,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int latentH = request.Height / 8;
        int latentW = request.Width / 8;
        int steps = request.Steps;
        float cfgScale = request.CfgScale;
        bool useCfg = cfgScale > 1.0f;
        if (useCfg && (negQwenEmbeds is null || negClipPooled is null))
            throw new ArgumentException(
                "cfgScale > 1.0 requires both negQwenEmbeds and negClipPooled.", nameof(negQwenEmbeds));

        Logs.Info($"Kandinsky5: Generating {request.Width}x{request.Height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Initial noise latent [B, 16, latentH, latentW] ──
        TensorShape latentShape = new TensorShape(1, _config.InVisualDim, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);

        // ── 2. Set up flow-match scheduler with shift=5.0 ──
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(_schedulerShift);
        scheduler.SetTimesteps(steps);
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(latentShape, DType.F32);
            Backend.Scale(scaled, latent, initSigma);
            latent.Dispose();
            latent = scaled;
        }

        // ── 3. Denoising loop ──
        // Bulk-upload transformer weights before the denoise loop. Paired with FreeWeights
        // below the VAE handoff. No-op on backends without a weight cache.
        Backend.PreloadWeights(_transformer.EnumerateWeights());

        Logs.Info("Kandinsky5: starting denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            Tensor noisePred;
            if (useCfg)
            {
                Tensor uncond = _transformer.Forward(Backend, latent, t, negQwenEmbeds!, negClipPooled!);
                Tensor cond = _transformer.Forward(Backend, latent, t, qwenEmbeds, clipPooled);
                noisePred = CfgHelper.ApplyCfg(uncond, cond, cfgScale);
                uncond.Dispose();
                cond.Dispose();
            }
            else
            {
                noisePred = _transformer.Forward(Backend, latent, t, qwenEmbeds, clipPooled);
            }

            Tensor newLatent = new Tensor(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePred, latent, i);
            noisePred.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        Kandinsky5Transformer.DumpFinalLatent(latent);

        // ── 4. Free transformer weights before VAE decode (mirrors AuraFlow / Flux pattern). ──
        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        // ── 5. Apply Flux VAE shift+scale before decoding: latent = latent / scale + shift ──
        Tensor decodeIn = new Tensor(latent.Shape, DType.F32);
        ApplyVaeShiftScale(decodeIn, latent, _vaeScalingFactor, _vaeShiftFactor);
        latent.Dispose();

        // ── 6. VAE decode (tiled to keep im2col workspace bounded). ──
        Logs.Verbose("Kandinsky5: decoding latents (tiled)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(Backend, decodeIn);
        decodeIn.Dispose();
        vaeSw.Stop();
        Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Kandinsky5: complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
        return (rgbData, request.Width, request.Height, seed);
    }

    /// <summary>Inverse of the VAE encoder's normalization: <c>x = x / scale + shift</c>. Flux/Kandinsky 5
    /// store latents in the shifted-scaled space, so we reverse it before decoding.</summary>
    private static void ApplyVaeShiftScale(Tensor output, Tensor input, float scale, float shift)
    {
        float* i = (float*)input.DataPointer;
        float* o = (float*)output.DataPointer;
        long n = input.ElementCount;
        float invScale = 1.0f / scale;
        for (long k = 0; k < n; k++)
            o[k] = i[k] * invScale + shift;
    }
}
