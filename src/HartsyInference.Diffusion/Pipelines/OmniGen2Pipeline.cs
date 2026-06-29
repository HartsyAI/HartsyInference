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

/// <summary>OmniGen 2 text-to-image pipeline. Caller supplies pre-computed Qwen2.5-VL caption embeddings
/// — OmniGen 2's text encoder is wired through the existing <see cref="Models.TextEncoders.LlamaStyleEncoder"/>
/// at the Qwen2.5-VL preset, but a working Qwen2.5-VL forward + embedding extraction is left to the
/// caller for now. Editing / multi-image-input paths are intentionally out of scope (t2i only).</summary>
public sealed class OmniGen2Pipeline : DiffusionPipelineBase
{
    private readonly OmniGen2Transformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly OmniGen2Config _config;

    /// <summary>Creates an OmniGen 2 t2i pipeline. Caller owns each component.</summary>
    public OmniGen2Pipeline(IBackend backend, OmniGen2Transformer transformer, VaeDecoder vaeDecoder, OmniGen2Config config)
        : base(backend)
    {
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-computed Qwen2.5-VL caption embeddings <c>[1, T, 2048]</c>.
    /// CFG dual-pass when guidance is active with a negative-prompt embedding.
    /// <para>OmniGen 2's defining feature is DUAL guidance: <paramref name="textGuidanceScale"/> (default 4.0,
    /// diffusers <c>text_guidance_scale</c>) and <paramref name="imageGuidanceScale"/> (default 1.0,
    /// diffusers <c>image_guidance_scale</c>). With an input image diffusers runs three forwards and combines
    /// <c>pred = uncond + image_guidance_scale*(text_only - uncond) + text_guidance_scale*(text_image - text_only)</c>.
    /// This pipeline is text-to-image only (no input image), so it reduces to standard CFG with
    /// <paramref name="textGuidanceScale"/>: <c>pred = uncond + text_guidance_scale*(cond - uncond)</c>.</para>
    /// <para><paramref name="cfgRange"/> = <c>(start, end)</c> as fractions of the schedule: CFG is only applied
    /// when <c>start &lt;= i/num_steps &lt;= end</c>; outside that window the bare conditional prediction is used.</para>
    /// <para><paramref name="cfgScale"/> is accepted for back-compat with the generic
    /// <see cref="TextToImageRequest"/> flow, but <paramref name="textGuidanceScale"/> is the effective guidance:
    /// when the caller leaves <paramref name="textGuidanceScale"/> at its default and passes a non-default
    /// <paramref name="cfgScale"/>, the <paramref name="cfgScale"/> value is used as the text guidance.</para></summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor captionEmbeddings,
        TextToImageRequest request,
        float cfgScale = 1.0f,
        Tensor? negativeCaptionEmbeddings = null,
        float textGuidanceScale = 4.0f,
        float imageGuidanceScale = 1.0f,
        (float start, float end)? cfgRange = null,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        // VALIDATION-PENDING: verify vs diffusers OmniGen2Pipeline.
        // Resolve the effective text guidance. textGuidanceScale (default 4.0) is the OmniGen 2 default and
        // takes priority. If the caller left it at the default but supplied a non-default generic cfgScale,
        // honour the generic value so the legacy request.CfgScale path keeps working.
        float effectiveTextGuidance =
            (Math.Abs(textGuidanceScale - 4.0f) < 1e-6f && cfgScale > 1.0f)
                ? cfgScale
                : textGuidanceScale;

        // cfg_range gates which steps apply CFG. Default = full schedule (0..1).
        (float rangeStart, float rangeEnd) = cfgRange ?? (0f, 1f);

        bool guidanceActive = CfgHelper.IsGuidanceActive(effectiveTextGuidance);

        if (guidanceActive && negativeCaptionEmbeddings is null)
            throw new ArgumentException(
                "negativeCaptionEmbeddings is required when text guidance > 1.0.",
                nameof(negativeCaptionEmbeddings));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, _, int width, int height) = GenerationDefaults.OmniGen2.Resolve(request);
        int latentH = height / 8;
        int latentW = width / 8;

        Logs.Info($"OmniGen 2 t2i: {width}x{height}, {steps} steps, textGuidance={effectiveTextGuidance}, " +
            $"imageGuidance={imageGuidanceScale}, cfgRange=[{rangeStart},{rangeEnd}], seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        TensorShape latentShape = new(1, _config.InChannels, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);

        FlowMatchEulerDiscreteScheduler scheduler = new(3.0f);
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        int textSeqLen = (int)captionEmbeddings.Shape[1];

        // Bulk-upload transformer weights before the denoise loop. Paired with FreeWeights
        // below the VAE handoff. No-op on backends without a weight cache.
        Backend.PreloadWeights(_transformer.EnumerateWeights());

        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            Tensor velocity = _transformer.Forward(Backend, latent, t, captionEmbeddings, textSeqLen);

            // VALIDATION-PENDING: verify vs diffusers OmniGen2Pipeline.
            // cfg_range gating: diffusers applies CFG only when start <= (i/num_steps) <= end, otherwise it
            // uses the bare conditional ("text_image" / cond) prediction. Match that fraction-of-schedule test.
            float schedFraction = steps > 1 ? (float)i / steps : 0f;
            bool inCfgRange = schedFraction >= rangeStart && schedFraction <= rangeEnd;

            if (guidanceActive && inCfgRange)
            {
                int negSeqLen = (int)negativeCaptionEmbeddings!.Shape[1];
                Tensor uncond = _transformer.Forward(Backend, latent, t, negativeCaptionEmbeddings, negSeqLen);

                // t2i path (no input image): reduces to standard CFG with the text guidance scale,
                // pred = uncond + text_guidance_scale * (cond - uncond).
                Tensor combined = CfgHelper.ApplyCfg(uncond, velocity, effectiveTextGuidance);
                uncond.Dispose();
                velocity.Dispose();
                velocity = combined;

                // VALIDATION-PENDING: verify vs diffusers OmniGen2Pipeline.
                // TODO(image-guidance): when input-image conditioning lands, run the full triple pass
                //   uncond     = forward(latent, negative)
                //   textOnly   = forward(latent, text, no-image)
                //   textImage  = forward(latent, text, image)
                // and combine:
                //   pred = uncond
                //        + imageGuidanceScale * (textOnly  - uncond)
                //        + textGuidanceScale  * (textImage - textOnly)
                // imageGuidanceScale (default 1.0) is wired through above for that path; t2i has no input
                // image, so only the text-guidance term is exercised here. Do NOT fabricate an image pass.
                _ = imageGuidanceScale;
            }

            Tensor newLatent = new(latentShape, DType.F32);
            scheduler.Step(newLatent, velocity, latent, i);
            velocity.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        Tensor decoded = _vaeDecoder.Decode(Backend, latent);
        latent.Dispose();
        byte[] rgb = ImagePostProcessor.TensorToRgbBytes(decoded);
        decoded.Dispose();

        sw.Stop();
        Logs.Info($"OmniGen 2 t2i complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgb, width, height, seed);
    }
}
