using System.Diagnostics;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Fibo 2.0 image-to-image and mask inpaint. This is the model-agnostic latent path the init image and mask of a host UI
/// drive: the source is VAE-encoded into the denoiser's normalised latent, noised to the sigma at the strength's start step on the
/// shifted schedule, and sampled from there; a mask keeps the unmasked region on the source's own noise trajectory at every step.
/// Bria's masked edit (white = regenerate, with <c>&lt;image_k&gt;</c> references) is a different feature: the model is conditioned
/// on the image and mask. That is reference editing and is not what this file does.</summary>
public sealed unsafe partial class Fibo2Pipeline
{
    /// <summary>One velocity evaluation at <paramref name="sigma"/>; the caller owns the returned tensor.</summary>
    internal delegate Tensor VelocityFunction(Tensor latent, float sigma);

    /// <summary>The per-step mask blend of a masked inpaint: after each Euler step the unmasked region is replaced with the source
    /// noised to the next sigma, and with the clean source after the last step. Mask value 1 regenerates, 0 keeps the source.</summary>
    internal sealed class LatentInpaint(FlowMatchEulerDiscreteScheduler scheduler, Tensor sourceLatent, Tensor latentMask, int seed)
    {
        /// <summary>Blends <paramref name="latent"/> in place for the state after step <paramref name="nextStep"/> − 1.</summary>
        public void Apply(Tensor latent, int nextStep, int steps)
        {
            if (nextStep >= steps)
            {
                MaskBlendUtilities.BlendChannelsInPlace(latent, sourceLatent, latentMask);
                return;
            }
            using Tensor noise = SeedGenerator.CreateNoise(sourceLatent.Shape, seed + nextStep);
            using Tensor noised = new(sourceLatent.Shape, DType.F32);
            scheduler.AddNoise(noised, sourceLatent, noise, nextStep);
            MaskBlendUtilities.BlendChannelsInPlace(latent, noised, latentMask);
        }
    }

    /// <summary>Validates the init image and mask and derives the start step; plain text-to-image gets the no-op plan.</summary>
    /// <exception cref="InvalidOperationException">An init image arrived but no VAE encoder was loaded (a decode-only VAE file).</exception>
    private Img2ImgSetup.Plan PrepareInit(ImageToImageRequest? img2img, int height, int width, int steps)
    {
        if (img2img is null)
            return Img2ImgSetup.TextToImage;
        if (_vaeEncoder is null)
            throw new InvalidOperationException("Fibo 2.0 img2img needs the FLUX.2 VAE encoder, but the loaded VAE file has no encoder weights.");
        return Img2ImgSetup.Prepare(img2img, height, width, steps);
    }

    /// <summary>Encodes an image <c>[1, 3, H, W]</c> to the normalised <c>[1, 32, H/8, W/8]</c> latent the denoiser runs on: the exact
    /// inverse of <see cref="DecodeLatent"/>.</summary>
    private Tensor EncodeInitLatent(Tensor image)
    {
        using Tensor raw = _vaeEncoder!.Encode(Backend, image);
        return NormalizeVaeLatent(raw, _config, _bnMean, _bnVar, _bnEps);
    }

    /// <summary>VAE latent to denoiser latent: 2×2 patchify, BatchNorm normalisation on the 128 channels, unpatchify.</summary>
    internal static Tensor NormalizeVaeLatent(Tensor vaeLatent, Fibo2Config config, Tensor bnMean, Tensor bnVar, float bnEps)
    {
        using Tensor patched = Flux2LatentOps.Patchify(vaeLatent, config.InChannels * config.PatchSize * config.PatchSize, config.PatchSize);
        using Tensor normalized = Flux2LatentOps.BnNormalize(patched, bnMean, bnVar, bnEps);
        return Flux2LatentOps.Unpatchify(normalized, config.InChannels, config.PatchSize);
    }

    /// <summary>The latent the loop starts from. Text-to-image scales the noise by the schedule's initial sigma; img2img mixes
    /// <c>(1 − sigma)·source + sigma·noise</c> at the sigma of <paramref name="startStep"/>. Takes ownership of <paramref name="noise"/>.</summary>
    internal static Tensor BuildStartLatent(FlowMatchEulerDiscreteScheduler scheduler, Tensor noise, Tensor? sourceLatent, int startStep)
    {
        if (sourceLatent is null)
        {
            float initialSigma = scheduler.InitialNoiseSigma;
            if (MathF.Abs(initialSigma - 1f) > 1e-6f)
                ScaleInPlace(noise, initialSigma);
            return noise;
        }
        Tensor mixed = new(noise.Shape, DType.F32);
        scheduler.AddNoise(mixed, sourceLatent, noise, startStep);
        noise.Dispose();
        return mixed;
    }

    /// <summary>The Euler loop from <paramref name="startStep"/> to <paramref name="steps"/>. <paramref name="latent"/> is replaced
    /// in place at each step (the old tensor is disposed), so the caller's variable always names the live tensor, also when a step throws.
    /// Progress counts the steps actually run.</summary>
    internal static void SampleFlow(FlowMatchEulerDiscreteScheduler scheduler, ref Tensor? latent, int startStep, int steps,
        VelocityFunction velocity, LatentInpaint? inpaint, Action<GenerationProgress>? onProgress)
    {
        for (int i = startStep; i < steps; i++)
        {
            Stopwatch stepWatch = Stopwatch.StartNew();
            using Tensor velocityTensor = velocity(latent!, scheduler.SigmaAt(i));
            Tensor next = new(latent!.Shape, DType.F32);
            scheduler.Step(next, velocityTensor, latent, i);
            inpaint?.Apply(next, i + 1, steps);
            latent.Dispose();
            latent = next;
            stepWatch.Stop();
            onProgress?.Invoke(new GenerationProgress(i - startStep + 1, steps - startStep, stepWatch.Elapsed.TotalMilliseconds));
        }
    }
}
