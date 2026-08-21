using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Schedulers;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>A pipeline's denoise step, reduced to the one thing a sampler needs: evaluate the model at
/// <c>(x, sigma)</c> and hand back the raw prediction pair.
///
/// <para>Each pipeline implements this as a thin closure over the loop body it already has. Everything family-specific
/// stays behind it and is invisible to samplers — input scaling, batched cond/uncond forwards, ControlNet residual
/// injection, regional attention bias, block streaming, DiT sharding, step-cache. A sampler never learns which
/// architecture it is driving.</para>
///
/// <para><b>The sampler calls this, not the other way round.</b> Second-order samplers (heun, dpm_2, dpmpp_2s_ancestral,
/// dpmpp_sde, seeds_2/3) evaluate the model a second time at an intermediate sigma inside a single step. An interface
/// that handed the sampler pre-computed predictions could not express them at all.</para></summary>
public interface IDenoisePredictor
{
    /// <summary>What <see cref="Predict"/>'s output means. Samplers past plain Euler convert it to a denoised (x0)
    /// estimate via <see cref="SamplerMath.ToDenoised"/> before applying their own math.</summary>
    PredictionType Prediction { get; }

    /// <summary>Evaluates the model at noise level <paramref name="sigma"/>. The caller owns the returned pair and must
    /// dispose it.</summary>
    /// <param name="x">Current latent. Implementations must NOT mutate it — samplers reuse it across the sub-evaluations
    /// of one step, and the fused Euler op writes to it only after the last read.</param>
    /// <param name="sigma">Noise level to evaluate at. For a sub-step this is an intermediate value that appears nowhere
    /// in the schedule, so implementations must map sigma to their conditioning timestep rather than index a table.</param>
    /// <param name="stepIndex">Schedule index the evaluation belongs to, for progress reporting and step-cache keying.
    /// Several evaluations within one step share it.</param>
    DenoisePrediction Predict(Tensor x, float sigma, int stepIndex);
}
