using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>ComfyUI's <c>euler_cfg_pp</c>: the Euler step re-anchored on the unconditional prediction (CFG++), so it
/// needs the pipeline to hand back a real conditional/unconditional pair.</summary>
public sealed class EulerCfgPlusPlusSampler : SamplerBase
{
    /// <summary>Creates the sampler over a resolved sigma array.</summary>
    public EulerCfgPlusPlusSampler(float[] sigmas, SamplerOptions? options = null)
        : base(sigmas, 0, options)
    {
    }

    /// <inheritdoc/>
    public override string Name => "euler_cfg_pp";

    /// <inheritdoc/>
    protected override void StepLocal(IBackend backend, Tensor z, IDenoisePredictor predictor, int i, int stepIndex)
    {
        float sigma = Sigma(i);
        float sigmaNext = Sigma(i + 1);
        bool paired = SamplerMath.TryPredictDenoisedPair(backend, predictor, z, sigma, stepIndex, out Tensor denoised,
            out Tensor? uncondDenoised);
        using Tensor combined = denoised;
        using Tensor? uncond = uncondDenoised;
        if (!paired && i == 0)
        {
            throw new NotSupportedException(
                "euler_cfg_pp needs the unconditional prediction, and this model returned an already-combined or "
                + "guidance-free result. Raise CFG above 1 on a model with a negative-prompt branch, or pick another sampler.");
        }
        if (sigmaNext == 0f)
        {
            backend.Scale(z, combined, 1.0f);
            return;
        }
        double alphaS = sigma * Math.Exp(Lambda(sigma));
        double alphaT = sigmaNext * Math.Exp(Lambda(sigmaNext));
        double ratio = (double)sigmaNext / sigma;
        SamplerOps.MixInto(backend, z, combined, uncond ?? combined, (float)ratio, (float)alphaT, (float)(-ratio * alphaS));
    }
}
