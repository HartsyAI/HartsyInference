using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Zeta-Chroma text-to-image pipeline (<c>lodestones/Zeta-Chroma</c> pixel-proto) — pixel-space, VAE-free. Accepts pre-computed Qwen3-4B caption embeddings (same upstream encoder path as <see cref="ZImagePipeline"/>) and runs the <see cref="ZetaChromaTransformer"/> directly on RGB in [-1, 1]. Sampling (ALL validation-gated — the model is mid-pretraining, see docs/Research/CHROMA_RADIANCE_ARCHITECTURE.md §Zeta-Chroma):
/// <list type="bullet">
///   <item><b>x0 prediction</b> — converted per step via <see cref="X0Prediction.ToVelocity"/> (<c>v = (x_t − x0) / t</c>, ComfyUI <c>NextDiTPixelSpace</c>), then Chroma-family cond-anchored CFG on v (<see cref="CfgHelper.ApplyCfgCondAnchored"/>, <c>cond + scale·(cond − uncond)</c> — NOT Z-Image's non-standard cond-baseline formula; item 8).</item>
///   <item><b>Timestep inversion</b> — the transformer is conditioned on <c>1 − sigma</c>, inheriting Z-Image's convention (item 7).</item>
///   <item><b>Flow-match Euler</b>, static shift 3.0 (item 9), default 50 steps, CFG 5.0.</item>
///   <item><b>Resolution must be divisible by the pixel patch size</b> (32 reported) — no pad/crop path until the training resolution behavior is known.</item>
/// </list>
///
/// <para>Creates a Zeta-Chroma pipeline. The Qwen3 caption encoder is owned by the caller (as with Z-Image).</para></summary>
public sealed class ZetaChromaPipeline(IBackend backend, ZetaChromaTransformer transformer, ZetaChromaConfig config)
    : DiffusionPipelineBase(backend)
{
    private readonly ZetaChromaTransformer _transformer = transformer;
    private readonly ZetaChromaConfig _config = config;

    /// <summary>Generates an image from pre-computed Qwen3 caption embeddings. API mirrors <see cref="ZImagePipeline.GenerateFromEmbeddings"/>. Img2img / inpaint is pixel-space (no VAE): pass an <see cref="ImageToImageRequest"/> and the source pixels are noised directly at <c>sigma[startStep]</c> (VALIDATION-PENDING alongside the rest of the sampling recipe — mid-pretraining checkpoint).</summary>
    /// <param name="captionEmbeddings">Qwen3-4B last-hidden-state for the prompt [B, capLen, 2560].</param>
    /// <param name="request">Generation parameters. Width/Height must be divisible by the 32-px patch.</param>
    /// <param name="cfgScale">CFG scale (5.0 recommended; 1.0 disables the second pass).</param>
    /// <param name="negativeCaptionEmbeddings">Negative-prompt embeddings, required when <paramref name="cfgScale"/> &gt; 1.</param>
    /// <param name="onProgress">Optional progress callback per step.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor captionEmbeddings,
        TextToImageRequest request,
        float cfgScale = 5.0f,
        Tensor? negativeCaptionEmbeddings = null,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        // Wrap-pad every conv backend for this call so the output tiles seamlessly; restores on dispose.
        using IDisposable seamlessScope = BeginSeamlessTiling(request.SeamlessTiling);

        if (cfgScale > 1.0f && negativeCaptionEmbeddings is null)
            throw new ArgumentException(
                "negativeCaptionEmbeddings is required when cfgScale > 1.0.", nameof(negativeCaptionEmbeddings));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width ?? GenerationDefaults.Chroma.Width;
        int height = request.Height ?? GenerationDefaults.Chroma.Height;
        int steps = request.Steps ?? _config.DefaultSteps;
        int patch = _transformer.PatchSize;
        bool useCfg = cfgScale > 1.0f;

        if (width % patch != 0 || height % patch != 0)
            throw new ArgumentException(
                $"Zeta-Chroma is pixel-space with a {patch}-px patch; width/height ({width}x{height}) must be " +
                $"divisible by {patch}.", nameof(request));

        // Img2img / inpaint: pixel-space means no VAE — the source image IS the clean sample. Since
        // Zeta-Chroma has no pad/crop path, the source already matches the (patch-aligned) pixel shape.
        Img2ImgSetup.Plan plan = Img2ImgSetup.Prepare(request, height, width, steps);
        bool isImg2Img = request is ImageToImageRequest;
        if (plan.PassThrough)
        {
            Logs.Info("Strength=0; passing source through unchanged");
            return (ImagePostProcessor.TensorToRgbBytes(((ImageToImageRequest)request).SourceImage), width, height, seed);
        }
        int startStep = plan.StartStep;
        Tensor? maskPixel = plan.MaskPixel;
        bool isMaskedInpaint = maskPixel is not null;

        string opMode = isMaskedInpaint ? $"inpaint (start={startStep}/{steps})"
                      : isImg2Img ? $"img2img (start={startStep}/{steps})"
                      : "txt2img";
        Logs.Info($"Zeta-Chroma {opMode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed} " +
            "(mid-pretraining checkpoint — output quality is validation-gated)");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Static-shift flow-match Euler scheduler ──
        TensorShape pixelShape = new TensorShape(1, 3, height, width);
        // Reference dynamic shift (FLUX-style): mu linear in token count, exp(mu) ≈ 1.88 at 1024²
        // (1024 tokens) — the static 3.0 starved the low-sigma tail where patch texture harmonizes.
        int tokenCount = (height / 32) * (width / 32);
        float mu = 0.5f + 0.65f * (tokenCount - 256) / 3840.0f;
        float dynamicShift = MathF.Exp(mu);
        FlowMatchEulerDiscreteScheduler scheduler = new(dynamicShift);
        scheduler.SetTimesteps(steps);

        // ── 2. Initial pixel sample ──
        // T2I: pure noise scaled by initSigma. Img2img: AddNoise(source, noise) at sigma[startStep]
        // directly on the pixels (no VAE encode). Masked inpaint keeps the source for per-step blending.
        Tensor pixels = TakeOrCreateNoise(request, pixelShape, seed);
        if (!pixels.Shape.Equals(pixelShape))
            throw new ArgumentException($"InitialNoise shape {pixels.Shape} != expected {pixelShape}.");
        Tensor? sourcePixels = null;
        if (isImg2Img)
        {
            sourcePixels = ((ImageToImageRequest)request).SourceImage;
            Tensor noised = new Tensor(pixelShape, DType.F32);
            scheduler.AddNoise(noised, sourcePixels, pixels, startStep);
            pixels.Dispose();
            pixels = noised;
        }
        else
        {
            float initSigma = scheduler.InitialNoiseSigma;
            if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
            {
                Tensor scaled = new Tensor(pixelShape, DType.F32);
                Backend.Scale(scaled, pixels, initSigma);
                pixels.Dispose();
                pixels = scaled;
            }
        }

        // ── 3. Denoising loop ──
        Backend.PreloadWeights(_transformer.EnumerateWeights());

        // Sampler seam. Zeta-Chroma is flow-matching, so it had no user-selectable sampler at all before this: the
        // predictor converts each x0 pass to velocity and applies the family's own CFG, and the sampler integrates in
        // the same velocity domain the host scheduler step did. Masked inpaint keeps the reference branch, which
        // rebuilds the sample on the host from the SCHEDULER's sigmas every step and never consults the sampler.
        bool samplerDriven = !isMaskedInpaint;
        ISampler sampler = FlowMatchSampling.Resolve(request.Scheduler, scheduler, seed, "Zeta-Chroma",
            startsFromNoisedInit: startStep > 0);
        bool nonDefaultSampler = FlowMatchSampling.IsNonDefault(request.Scheduler);
        if (!samplerDriven && nonDefaultSampler)
        {
            throw new NotSupportedException(
                $"Sampler/schedule '{request.Scheduler}' runs only on Zeta-Chroma's sampler-driven path, and this "
                + "generation fell back to the reference loop (masked inpaint). Drop the sampler selection, or "
                + "drop the feature that forced the fallback.");
        }

        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        // The predictor closure needs the timestep table, and `timesteps` is a ReadOnlySpan (a ref local) that a
        // lambda cannot capture. One small array copy per generation.
        float[] timestepTable = timesteps.ToArray();

        DelegateDenoisePredictor predictor = new DelegateDenoisePredictor(
            PredictionType.FlowVelocity,
            (x, s, stepIndex) =>
            {
                // On-schedule sigmas reuse the loop's own `timesteps[i]/1000` expression: the two are equal
                // mathematically, but the F32 round trip through x1000 is not exact, and substituting one for the
                // other would shift every existing generation by an ulp of conditioning.
                float t = stepIndex < steps && s == scheduler.SigmaAt(stepIndex) ? timestepTable[stepIndex] / 1000.0f
                    : s;
                // Z-Image lineage: the transformer conditions on the INVERTED timestep (1 - sigma); the
                // transformer multiplies by t_scale=1000 internally. Validation-gated (research doc item 7).
                float invertedSigma = 1.0f - t;

                // Model predicts x0; convert each pass to velocity BEFORE the CFG combine. The conversion is a
                // function of the sample being evaluated, so it runs against `x` and this evaluation's own t — a
                // sub-step's x0 residual belongs to ITS sample, not the step's.
                Tensor condX0 = _transformer.Forward(Backend, x, captionEmbeddings, invertedSigma);
                Tensor condV = X0Prediction.ToVelocity(condX0, x, t);
                condX0.Dispose();
                if (!useCfg)
                {
                    return new DenoisePrediction(condV, condV);
                }
                Tensor uncondX0 = _transformer.Forward(Backend, x, negativeCaptionEmbeddings!, invertedSigma);
                Tensor uncondV = X0Prediction.ToVelocity(uncondX0, x, t);
                uncondX0.Dispose();

                // VALIDATION-PENDING: standard uncond-anchored CFG on velocity — lodestones sampling.py::denoise_cfg
                // uses it; the Chroma cond-anchored form over-guided by one unit and amplified the per-patch x0 tile
                // texture. Kept as the HOST combine rather than handed to the fused kernel as a raw pair: ApplyCfg's
                // u + s·(c − u) equals the kernel's s·c + (1 − s)·u mathematically but not in F32.
                Tensor combined = CfgHelper.ApplyCfg(uncondV, condV, cfgScale);
                uncondV.Dispose();
                condV.Dispose();
                return new DenoisePrediction(combined, combined);
            });
        if (samplerDriven)
        {
            sampler.Reset(pixelShape);
        }

        for (int i = startStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float sigma = timesteps[i] / 1000.0f;

            if (samplerDriven)
            {
                sampler.Step(Backend, pixels, predictor, i);
            }
            else
            {
                // Z-Image lineage: the transformer conditions on the INVERTED timestep (1 - sigma); the
                // transformer multiplies by t_scale=1000 internally. Validation-gated (research doc item 7).
                float invertedSigma = 1.0f - sigma;

                Tensor condX0 = _transformer.Forward(Backend, pixels, captionEmbeddings, invertedSigma);
                Tensor velocity = X0Prediction.ToVelocity(condX0, pixels, sigma);
                condX0.Dispose();

                if (useCfg)
                {
                    Tensor uncondX0 = _transformer.Forward(Backend, pixels, negativeCaptionEmbeddings!, invertedSigma);
                    Tensor uncondV = X0Prediction.ToVelocity(uncondX0, pixels, sigma);
                    uncondX0.Dispose();

                    // VALIDATION-PENDING: standard uncond-anchored CFG (see the predictor above); verify vs reference.
                    Tensor combined = CfgHelper.ApplyCfg(uncondV, velocity, cfgScale);
                    uncondV.Dispose();
                    velocity.Dispose();
                    velocity = combined;
                }

                Tensor newPixels = new Tensor(pixelShape, DType.F32);
                scheduler.Step(newPixels, velocity, pixels, i);
                velocity.Dispose();
                pixels.Dispose();
                pixels = newPixels;
            }

            // Masked-inpaint blend in pixel space: keep unmasked region on the source's
            // flow-matching trajectory by re-noising the source at the next step's sigma.
            // Final step blends with the clean source — no further denoising follows.
            if (isMaskedInpaint && sourcePixels is not null)
            {
                int nextStep = i + 1;
                Tensor noisedSource;
                if (nextStep < steps)
                {
                    Tensor freshNoise = SeedGenerator.CreateNoise(pixelShape, seed + nextStep);
                    noisedSource = new Tensor(pixelShape, DType.F32);
                    scheduler.AddNoise(noisedSource, sourcePixels, freshNoise, nextStep);
                    freshNoise.Dispose();
                }
                else
                {
                    noisedSource = sourcePixels;
                }
                MaskBlendUtilities.BlendChannelsInPlace(pixels, noisedSource, maskPixel!);
                if (noisedSource != sourcePixels) noisedSource.Dispose();
            }

            stepSw.Stop();
            Logs.Debug($"Zeta-Chroma step {i + 1}/{steps} (sigma={sigma:F4}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
            {
                Latent = pixels,
                LatentArch = LatentArchitecture.ZetaChroma,
            });
        }

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        // Recomposite for masked inpaint: guarantees the unmasked region is byte-exact to the source
        // (consistent with the latent-space pipelines; there is no VAE drift to suppress here).
        if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
        {
            MaskBlendUtilities.BlendChannelsInPlace(pixels, ((ImageToImageRequest)request).SourceImage, maskPixel!);
        }

        // ── 4. Direct RGB conversion — no VAE ──
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(pixels);
        pixels.Dispose();

        sw.Stop();
        Logs.Info($"Zeta-Chroma {opMode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }
}
