using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>One sampling algorithm — the integrator half of sampling, paired with <see cref="SigmaSchedule"/>'s spacing
/// half. Implemented once and available to every model family, because the only thing it touches is
/// <see cref="IDenoisePredictor"/>.
///
/// <para>A sampler owns its resolved sigma array, any multi-step history buffers, and how many times it evaluates the
/// model per step. It does not own the latent: <see cref="Step"/> mutates the caller's tensor in place, which is what
/// keeps the latent GPU-resident across the loop.</para></summary>
public interface ISampler
{
    /// <summary>Canonical lowercase name, matching ComfyUI's sampler vocabulary (<c>euler</c>, <c>euler_ancestral</c>,
    /// <c>dpmpp_2m</c>, …).</summary>
    string Name { get; }

    /// <summary>Total steps in the resolved schedule.</summary>
    int StepCount { get; }

    /// <summary>Prepares for a run over latents of <paramref name="latentShape"/>, clearing any history from a previous
    /// generation. Must be called before the first <see cref="Step"/>; cheap and idempotent for stateless samplers.</summary>
    void Reset(TensorShape latentShape);

    /// <summary>Advances <paramref name="z"/> from <c>sigma[stepIndex]</c> to <c>sigma[stepIndex + 1]</c>, in place,
    /// evaluating the model through <paramref name="predictor"/> as many times as the algorithm requires.</summary>
    void Step(IBackend backend, Tensor z, IDenoisePredictor predictor, int stepIndex);
}
