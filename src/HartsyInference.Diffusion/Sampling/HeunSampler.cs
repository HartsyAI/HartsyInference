using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>k-diffusion's <c>heun</c> — Euler's predictor followed by a corrector that averages the derivative at both
/// ends of the step. Second order, so it costs TWO model evaluations per step: a 20-step heun run is ~40 forwards, and
/// should be compared against a 40-step euler run rather than a 20-step one.
///
/// <para>This is the sampler that justifies <see cref="ISampler"/> owning the evaluation loop. The corrector needs the
/// model at the step's END sigma, on a latent that does not exist until the predictor has run — a design handing
/// samplers pre-computed predictions could not express it at all.</para>
///
/// <para>The final step has <c>sigmaNext == 0</c>, where the corrector's <c>d = (x − denoised)/sigma</c> would divide
/// by zero; k-diffusion degrades to plain Euler there, and so does this.</para></summary>
public sealed class HeunSampler : ISampler
{
    private readonly float[] _sigmas;

    /// <inheritdoc/>
    public string Name => "heun";

    /// <inheritdoc/>
    public int StepCount => _sigmas.Length - 1;

    /// <summary>Creates the sampler over a resolved sigma array.</summary>
    public HeunSampler(float[] sigmas)
    {
        ArgumentNullException.ThrowIfNull(sigmas);
        if (sigmas.Length < 2)
        {
            throw new ArgumentException($"Need at least 2 sigmas; got {sigmas.Length}.", nameof(sigmas));
        }
        _sigmas = sigmas;
    }

    /// <inheritdoc/>
    public void Reset(TensorShape latentShape)
    {
        // Stateless across steps: the corrector's second evaluation lives entirely inside one Step call.
    }

    /// <inheritdoc/>
    public void Step(IBackend backend, Tensor z, IDenoisePredictor predictor, int stepIndex)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(z);
        ArgumentNullException.ThrowIfNull(predictor);
        float sigma = _sigmas[stepIndex];
        float sigmaNext = _sigmas[stepIndex + 1];
        float dt = sigmaNext - sigma;

        // d = (z − denoised)/sigma
        using Tensor denoised = SamplerMath.PredictDenoised(backend, predictor, z, sigma, stepIndex);
        using Tensor d = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, d, z, denoised, 1.0f / sigma, -1.0f / sigma);

        if (sigmaNext <= 0f)
        {
            SamplerOps.MixInto(backend, z, d, 1.0f, dt);
            return;
        }

        // Predictor: z2 = z + d·dt, evaluated at the step's end sigma.
        using Tensor z2 = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, z2, z, d, 1.0f, dt);
        using Tensor denoised2 = SamplerMath.PredictDenoised(backend, predictor, z2, sigmaNext, stepIndex);

        // Corrector: average the two derivatives, then take the full step with it.
        using Tensor d2 = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, d2, z2, denoised2, 1.0f / sigmaNext, -1.0f / sigmaNext);
        SamplerOps.MixInto(backend, d2, d, 0.5f, 0.5f);
        SamplerOps.MixInto(backend, z, d2, 1.0f, dt);
    }
}
