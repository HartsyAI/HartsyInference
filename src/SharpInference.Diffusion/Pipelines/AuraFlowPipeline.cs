using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Diffusion.Pipelines;

/// <summary>AuraFlow text-to-image pipeline (<c>AuraFlowPipeline</c>). Encodes prompts with Pile-T5-XL (joint_attention_dim=2048), runs the AuraFlow MMDiT transformer with classifier-free guidance, then decodes with the SDXL-compatible 4-channel VAE. Uses a static-shift FlowMatch scheduler. Mirrors the diffusers pipeline at <c>diffusers/pipelines/aura_flow/pipeline_aura_flow.py</c>.</summary>
public sealed class AuraFlowPipeline : DiffusionPipelineBase
{
    private readonly T5TextEncoder _t5;
    private readonly AuraFlowTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly AuraFlowConfig _config;
    private readonly float _schedulerShift;

    /// <summary>Creates a new AuraFlow pipeline with all components pre-loaded.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="t5">Pile-T5-XL text encoder (output dim 2048, max length 256).</param>
    /// <param name="transformer">AuraFlow MMDiT transformer.</param>
    /// <param name="vaeDecoder">SDXL VAE decoder (<see cref="VaeConfig.Sdxl"/>; 4-channel latent, scaling=0.13025).</param>
    /// <param name="config">AuraFlow configuration (use <see cref="AuraFlowConfig.V03"/> for fal/AuraFlow-v0.3).</param>
    /// <param name="schedulerShift">Flow-match scheduler shift. AuraFlow uses a static shift; default 1.73 matches the diffusers config.</param>
    public AuraFlowPipeline(IBackend backend, T5TextEncoder t5, AuraFlowTransformer transformer,
        VaeDecoder vaeDecoder, AuraFlowConfig config, float schedulerShift = 1.73f)
        : base(backend)
    {
        _t5 = t5;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
        _schedulerShift = schedulerShift;
    }

    /// <summary>Generates an image from pre-tokenized Pile-T5-XL input with optional CFG.</summary>
    /// <param name="promptTokenIdsT5">Prompt token IDs from Pile-T5-XL tokenizer (max length 256).</param>
    /// <param name="negativePromptTokenIdsT5">Negative prompt token IDs (same length as <paramref name="promptTokenIdsT5"/>).</param>
    /// <param name="promptAttentionMaskT5">Optional T5 attention mask for the prompt (1=attend, 0=pad).</param>
    /// <param name="negativeAttentionMaskT5">Optional T5 attention mask for the negative prompt.</param>
    /// <param name="request">Generation parameters.</param>
    /// <param name="onProgress">Optional progress callback.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsT5,
        int[] negativePromptTokenIdsT5,
        int[]? promptAttentionMaskT5,
        int[]? negativeAttentionMaskT5,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int latentH = request.Height / 8;
        int latentW = request.Width / 8;
        int steps = request.Steps;
        float cfgScale = request.CfgScale;

        Logs.Info($"AuraFlow: Generating {request.Width}x{request.Height} image, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode text with Pile-T5-XL ───────────────────────────────
        Logs.Info("Encoding text with Pile-T5-XL...");

        // Bulk-upload T5 weights once so the encoder's many kernel launches don't each pay a
        // per-op cache-miss H2D transfer. Backends without a weight cache no-op. Paired with
        // the FreeWeights below the VAE handoff so VRAM cycles cleanly.
        Backend.PreloadWeights(_t5.EnumerateWeights());

        bool useCfg = cfgScale > 1.0f;

        int[][] batchT5 = [promptTokenIdsT5];
        int[][]? batchMask = promptAttentionMaskT5 is not null ? [promptAttentionMaskT5] : null;
        Tensor condContext = _t5.Encode(Backend, batchT5, batchMask);

        Tensor? uncondContext = null;
        if (useCfg)
        {
            int[][] negBatchT5 = [negativePromptTokenIdsT5];
            int[][]? negBatchMask = negativeAttentionMaskT5 is not null ? [negativeAttentionMaskT5] : null;
            uncondContext = _t5.Encode(Backend, negBatchT5, negBatchMask);
        }

        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

        // ── 2. Create initial noise latent [1, 4, latentH, latentW] ────
        TensorShape latentShape = new TensorShape(1, _config.InChannels, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);

        // ── 3. Set up flow-match scheduler (static shift) ────────────────
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

        // ── 4. Denoising loop ────────────────────────────────────────────
        // Bulk-upload transformer weights before the denoise loop (touched every step, every
        // block — without this the first step would pay cache-miss overhead per parameter).
        // Paired with FreeWeights at the VAE handoff so VRAM cycles between generations.
        Backend.PreloadWeights(_transformer.EnumerateWeights());

        Logs.Info("Starting AuraFlow denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            Tensor noisePred;
            if (useCfg)
            {
                Tensor uncondNoise = _transformer.Forward(Backend, latent, t, uncondContext!);
                Tensor condNoise = _transformer.Forward(Backend, latent, t, condContext);
                noisePred = CfgHelper.ApplyCfg(uncondNoise, condNoise, cfgScale);
                uncondNoise.Dispose();
                condNoise.Dispose();
            }
            else
            {
                noisePred = _transformer.Forward(Backend, latent, t, condContext);
            }

            Tensor newLatent = new Tensor(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePred, latent, i);
            noisePred.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
            {
                Latent = latent,
                LatentArch = LatentArchitecture.AuraFlow,
            });
        }

        condContext.Dispose();
        uncondContext?.Dispose();

        AuraFlowTransformer.DumpFinalLatent(latent);

        // Free transformer + T5 weights from GPU before VAE decode (mirrors SD3 pattern;
        // the SDXL VAE im2col buffers are large and the transformer is huge — we OOM on
        // a 12 GB card without this eviction).
        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());
        Backend.FreeWeights(_t5.EnumerateWeights());

        // ── 5. VAE decode ────────────────────────────────────────────────
        // Tiled decode: caps im2col workspace at ~2.4 GB per tile.
        Logs.Verbose("Decoding latents to image (tiled F32 path)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(Backend, latent);
        latent.Dispose();
        vaeSw.Stop();
        Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"AuraFlow image generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, request.Width, request.Height, seed);
    }
}
