using System.Diagnostics;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Fibo 2.0 editing (ComfyUI <c>TextEncodeFibo2</c> with images plus the denoiser's <c>ref_latents</c>; diffusers
/// <c>BriaFibo2Pipeline</c> with <c>image</c>): the source and reference images condition every step as clean tokens next to
/// the noisy image, and the prompt reaches the model through the vision tower too.</summary>
public sealed unsafe partial class Fibo2Pipeline
{
    /// <summary>Edits <paramref name="edit"/>'s images by the instruction in its prompt. Everything else is as
    /// <see cref="GenerateFromTokens"/>: Euler over the shifted sigmas, and the negative branch (the null caption, text only) runs
    /// only when <paramref name="guidanceScale"/> is above 1, as in ComfyUI and diffusers; the images condition both passes.</summary>
    /// <param name="edit">The prompt, images and the components that read them.</param>
    /// <param name="unconditionalTokens">Template-wrapped null-caption ids; encoded only when guidance is active.</param>
    /// <param name="width">Output width in pixels; a multiple of 16. The caller sizes the images for it.</param>
    /// <param name="height">Output height in pixels; a multiple of 16.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateEdit(Fibo2EditInputs edit, int[] unconditionalTokens, int width,
        int height, int steps, float guidanceScale, float sigmaShift, int? seed, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(edit);
        if (edit.ReferencePixels.Count == 0 || edit.VisionImages.Count != edit.ReferencePixels.Count)
        {
            throw new ArgumentException($"Fibo 2.0 editing needs one vision copy per image; got {edit.VisionImages.Count} vision copies "
                + $"for {edit.ReferencePixels.Count} images.", nameof(edit));
        }
        ValidateRequest(edit.ConditionalTokens, unconditionalTokens, width, height, steps, guidanceScale, sigmaShift);
        int resolvedSeed = seed ?? SeedGenerator.RandomSeed();
        int latentHeight = height / VaeSpatialScale;
        int latentWidth = width / VaeSpatialScale;

        bool guided = CfgHelper.IsGuidanceActive(guidanceScale);
        List<Tensor> references = [];
        Tensor? conditional = null;
        Tensor? unconditional = null;
        Tensor? latent = null;
        try
        {
            // Encoder phases first: the vision-conditioned prompt, then the reference latents.
            using (Tensor taps = edit.VisionEncoder.EncodeMultiLayer(Backend, edit.ConditionalTokens, edit.VisionImages, TapLayers))
                conditional = Fibo2TextBundles.Build(taps, (int)(taps.Shape[2] / TapLayers.Length));
            if (guided)
                unconditional = EncodeBundles(unconditionalTokens);
            foreach (Tensor pixels in edit.ReferencePixels)
                references.Add(EncodeReference(edit, pixels));

            FlowMatchEulerDiscreteScheduler scheduler = new(sigmaShift);
            scheduler.SetTimesteps(steps);
            latent = SeedGenerator.CreateNoise(new TensorShape(1, _config.InChannels, latentHeight, latentWidth), resolvedSeed);
            float initialSigma = scheduler.InitialNoiseSigma;
            if (MathF.Abs(initialSigma - 1f) > 1e-6f)
                ScaleInPlace(latent, initialSigma);

            for (int i = 0; i < steps; i++)
            {
                Stopwatch stepWatch = Stopwatch.StartNew();
                Tensor velocity = Denoise(latent, scheduler.SigmaAt(i), conditional, unconditional, guidanceScale, references);
                Tensor next = new(latent.Shape, DType.F32);
                scheduler.Step(next, velocity, latent, i);
                velocity.Dispose();
                latent.Dispose();
                latent = next;
                stepWatch.Stop();
                onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepWatch.Elapsed.TotalMilliseconds));
            }

            using Tensor decoded = DecodeLatent(latent);
            return (ImagePostProcessor.TensorToRgbBytes(decoded), (int)decoded.Shape[3], (int)decoded.Shape[2], resolvedSeed);
        }
        finally
        {
            latent?.Dispose();
            conditional?.Dispose();
            unconditional?.Dispose();
            foreach (Tensor reference in references)
                reference.Dispose();
        }
    }

    /// <summary>One guided velocity with the reference latents conditioning each pass: <c>uncond + g · (cond − uncond)</c> when CFG
    /// is active, otherwise the conditional velocity alone.</summary>
    private Tensor Denoise(Tensor latent, float sigma, Tensor conditional, Tensor? unconditional, float guidanceScale,
        IReadOnlyList<Tensor> references)
    {
        Tensor conditionalVelocity = _transformer.Forward(Backend, latent, sigma, conditional, references);
        if (unconditional is null)
            return conditionalVelocity;

        Tensor unconditionalVelocity = _transformer.Forward(Backend, latent, sigma, unconditional, references);
        Tensor guided = CfgHelper.ApplyCfg(unconditionalVelocity, conditionalVelocity, guidanceScale);
        unconditionalVelocity.Dispose();
        conditionalVelocity.Dispose();
        return guided;
    }

    /// <summary>An image <c>[1, 3, H, W]</c> in [-1, 1] as a reference latent: the VAE posterior mean, 2×2 patchified, BatchNorm
    /// normalised with the VAE's statistics and unpatchified, which is the layout the denoiser's own latent has (diffusers
    /// <c>prepare_image_latents</c>, ComfyUI <c>vae.encode</c> then <c>process_latent_in</c>).</summary>
    private Tensor EncodeReference(Fibo2EditInputs edit, Tensor pixels)
    {
        using Tensor vaeLatent = edit.VaeEncoder.Encode(Backend, pixels);
        int patchVolume = _config.InChannels * _config.PatchSize * _config.PatchSize;
        using Tensor patched = Flux2LatentOps.Patchify(vaeLatent, patchVolume, _config.PatchSize);
        using Tensor normalized = Flux2LatentOps.BnNormalize(patched, _bnMean, _bnVar, _bnEps);
        return Flux2LatentOps.Unpatchify(normalized, _config.InChannels, _config.PatchSize);
    }
}
