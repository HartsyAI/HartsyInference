using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Diffusion.Pipelines;

/// <summary>Lumina-Image-2.0 text-to-image pipeline (Alpha-VLLM, Apache 2.0). Accepts pre-computed Gemma-2 caption embeddings (text-encoder forward is owned by a separate component, since SharpInference's <see cref="Models.TextEncoders.LlamaStyleEncoder"/> does not yet implement Gemma-2-specific features like GeGLU MLP and attention soft-capping) and orchestrates the NextDiT transformer with a static-shift flow-match Euler scheduler.
/// <para>The diffusers pipeline (<c>pipeline_lumina2.py</c>) inverts the timestep before feeding it to the transformer (<c>1 - t / num_train_timesteps</c>) and negates the predicted velocity before the scheduler step — both behaviors are replicated here for parity. Default sampling: 30 steps, cfg_scale=4.0 with a negative prompt (matches the diffusers default). Only t2i is supported.</para>
/// </summary>
public sealed unsafe class Lumina2Pipeline : DiffusionPipelineBase
{
    private readonly Lumina2Transformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly Lumina2Config _config;

    /// <summary>Creates a Lumina-Image-2.0 pipeline (text-to-image only; no image-to-image).</summary>
    public Lumina2Pipeline(IBackend backend, Lumina2Transformer transformer, VaeDecoder vaeDecoder, Lumina2Config config)
        : base(backend)
    {
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-computed Gemma-2 caption embeddings. CFG is applied when <paramref name="cfgScale"/> &gt; 1.0 and a negative-prompt embedding is provided.</summary>
    /// <param name="captionEmbeddings">Last hidden state of Gemma-2-2B applied to the prompt: [B, capLen, 2304]. The Lumina 2.0 system prompt prefix should already be applied upstream.</param>
    /// <param name="request">Generation parameters (Width, Height, Steps, Seed).</param>
    /// <param name="cfgScale">Classifier-free guidance scale. Lumina 2.0's diffusers default is 4.0 with a negative prompt.</param>
    /// <param name="negativeCaptionEmbeddings">Negative-prompt embeddings for CFG. Required when <paramref name="cfgScale"/> &gt; 1.0.</param>
    /// <param name="onProgress">Optional progress callback per step.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor captionEmbeddings,
        TextToImageRequest request,
        float cfgScale = 4.0f,
        Tensor? negativeCaptionEmbeddings = null,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        if (cfgScale > 1.0f && negativeCaptionEmbeddings is null)
            throw new ArgumentException(
                "negativeCaptionEmbeddings is required when cfgScale > 1.0. Lumina 2.0's diffusers pipeline always uses a negative prompt at cfg=4.0.",
                nameof(negativeCaptionEmbeddings));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width;
        int height = request.Height;
        int latentH = height / _config.VaeDownscaleFactor;
        int latentW = width / _config.VaeDownscaleFactor;
        int steps = request.Steps;

        Logs.Info($"Lumina 2.0 t2i: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        TensorShape latentShape = new TensorShape(1, _config.InChannels, latentH, latentW);
        FlowMatchEulerDiscreteScheduler scheduler = new(_config.SchedulerShift);
        scheduler.SetTimesteps(steps);

        Tensor latent = BuildInitialLatent(scheduler, latentShape, seed);

        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float sigma = timesteps[i] / 1000.0f;

            // Lumina 2.0's diffusers pipeline inverts the timestep before feeding it to the transformer
            // (`current_timestep = 1 - t/num_train_timesteps`). See pipeline_lumina2.py:757.
            float invertedSigma = 1.0f - sigma;
            Tensor velocity = _transformer.Forward(Backend, latent, captionEmbeddings, invertedSigma);

            if (cfgScale > 1.0f)
            {
                Tensor uncondVelocity = _transformer.Forward(Backend, latent, negativeCaptionEmbeddings!, invertedSigma);
                Tensor combined = CfgHelper.ApplyCfg(uncondVelocity, velocity, cfgScale);
                uncondVelocity.Dispose();
                velocity.Dispose();
                velocity = combined;
            }

            // Lumina 2.0 negates the velocity before the scheduler step (pipeline_lumina2.py:771).
            NegateInPlace(velocity);

            Tensor newLatent = new Tensor(latentShape, DType.F32);
            scheduler.Step(newLatent, velocity, latent, i);
            velocity.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            Logs.Debug($"Lumina 2.0 step {i + 1}/{steps} (sigma={sigma:F4}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        Logs.Verbose("Decoding latents to image (tiled F32 path)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(Backend, latent);
        latent.Dispose();
        vaeSw.Stop();
        Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Lumina 2.0 t2i complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    /// <summary>Builds the initial latent: noise * initSigma. Lumina 2.0's flow-match scheduler at sigma_init typically yields ~1.0 so this is a near-identity scaling.</summary>
    private Tensor BuildInitialLatent(FlowMatchEulerDiscreteScheduler scheduler, TensorShape latentShape, int seed)
    {
        Tensor noise = SeedGenerator.CreateNoise(latentShape, seed);
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(latentShape, DType.F32);
            Backend.Scale(scaled, noise, initSigma);
            noise.Dispose();
            return scaled;
        }
        return noise;
    }

    private static void NegateInPlace(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long count = t.Shape.ElementCount;
        for (long i = 0; i < count; i++)
            p[i] = -p[i];
    }
}
