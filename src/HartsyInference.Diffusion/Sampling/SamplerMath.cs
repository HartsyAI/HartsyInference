using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Schedulers;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>The two conversions every sampler past plain Euler needs: CFG-combining a prediction pair, and turning the
/// combined raw prediction into a denoised (x0) estimate.
///
/// <para><b>Pinning the domain matters.</b> The dpmpp / uni_pc / deis / ipndm families are all written in terms of x0,
/// but families return three different things — epsilon (SD1.5/SDXL), v (some finetunes), flow velocity (every DiT).
/// Implementing a sampler against whichever one its author had in mind produces correct images on that family and mud
/// on the others, with no error anywhere. Every sampler therefore goes through <see cref="ToDenoised"/> and never reads
/// a raw prediction directly.</para></summary>
public static class SamplerMath
{
    /// <summary>Combines a prediction pair by CFG into <paramref name="output"/>:
    /// <c>v = w·cond + (1−w)·uncond</c>. Skips the arithmetic entirely at <c>w == 1</c> or when the pair is the
    /// guidance-free aliased form.</summary>
    public static void CombineCfg(IBackend backend, Tensor output, in DenoisePrediction prediction)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(output);
        float guidance = prediction.Guidance;
        if (ReferenceEquals(prediction.Cond, prediction.Uncond) || guidance == 1.0f)
        {
            backend.Scale(output, prediction.Cond, 1.0f);
            return;
        }
        backend.AffineMix(output, prediction.Cond, prediction.Uncond, guidance, 1.0f - guidance);
    }

    /// <summary>Converts a CFG-combined raw prediction at noise level <paramref name="sigma"/> into the denoised (x0)
    /// estimate, writing into <paramref name="denoised"/>. ComfyUI's <c>ModelSampling.calculate_denoised</c>.
    ///
    /// <para><see cref="PredictionType.Epsilon"/> and <see cref="PredictionType.FlowVelocity"/> share the formula
    /// <c>x0 = x − sigma·pred</c> — for epsilon because <c>x = x0 + sigma·eps</c> by construction, and for flow matching
    /// because <c>x_t = (1−sigma)·x0 + sigma·noise</c> with <c>v = noise − x0</c> rearranges to the same thing. They stay
    /// separate enum members because their sigma domains differ, not because this step does.</para>
    ///
    /// <para><see cref="PredictionType.VPrediction"/> needs the VP alphas: with <c>alpha = 1/sqrt(sigma²+1)</c>,
    /// <c>x0 = alpha²·(x − sigma·v·sqrt(sigma²+1))</c>… expressed here in the equivalent single-pass form
    /// <c>x0 = (x − sigma·sqrt(sigma²+1)·v) / (sigma²+1)</c>.</para></summary>
    public static void ToDenoised(IBackend backend, Tensor denoised, Tensor x, Tensor prediction, float sigma,
        PredictionType predictionType)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(denoised);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(prediction);
        switch (predictionType)
        {
            case PredictionType.Epsilon:
            case PredictionType.FlowVelocity:
                backend.AffineMix(denoised, x, prediction, 1.0f, -sigma);
                break;
            case PredictionType.NegatedFlowVelocity:
                // Same relation with the model's sign convention flipped: x0 = x + sigma·(−v).
                backend.AffineMix(denoised, x, prediction, 1.0f, sigma);
                break;
            case PredictionType.VPrediction:
                float sigmaSqPlus1 = (sigma * sigma) + 1.0f;
                backend.AffineMix(denoised, x, prediction, 1.0f / sigmaSqPlus1,
                    -sigma * MathF.Sqrt(sigmaSqPlus1) / sigmaSqPlus1);
                break;
            default:
                throw new NotSupportedException($"Unhandled prediction type {predictionType} in the sampler core.");
        }
    }

    /// <summary>Evaluates the model and returns the denoised (x0) estimate in a freshly allocated tensor — the shape
    /// every non-Euler sampler wants. Disposes the prediction pair before returning.</summary>
    public static Tensor PredictDenoised(IBackend backend, IDenoisePredictor predictor, Tensor x, float sigma, int stepIndex)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(predictor);
        ArgumentNullException.ThrowIfNull(x);
        Tensor combined = new Tensor(x.Shape, DType.F32);
        Tensor denoised = new Tensor(x.Shape, DType.F32);
        try
        {
            using DenoisePrediction prediction = predictor.Predict(x, sigma, stepIndex);
            CombineCfg(backend, combined, prediction);
            ToDenoised(backend, denoised, x, combined, sigma, predictor.Prediction);
            return denoised;
        }
        catch
        {
            denoised.Dispose();
            throw;
        }
        finally
        {
            combined.Dispose();
        }
    }

    /// <summary>Evaluates the model and returns both the CFG-combined and the unconditional denoised estimates, for
    /// CFG++ samplers. Returns false, with <paramref name="uncondDenoised"/> null, when the pair carries no separate
    /// unconditional branch.</summary>
    public static bool TryPredictDenoisedPair(IBackend backend, IDenoisePredictor predictor, Tensor x, float sigma,
        int stepIndex, out Tensor denoised, out Tensor? uncondDenoised)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(predictor);
        ArgumentNullException.ThrowIfNull(x);
        Tensor combined = new Tensor(x.Shape, DType.F32);
        denoised = new Tensor(x.Shape, DType.F32);
        uncondDenoised = null;
        try
        {
            using DenoisePrediction prediction = predictor.Predict(x, sigma, stepIndex);
            CombineCfg(backend, combined, prediction);
            ToDenoised(backend, denoised, x, combined, sigma, predictor.Prediction);
            if (ReferenceEquals(prediction.Cond, prediction.Uncond))
            {
                return false;
            }
            uncondDenoised = new Tensor(x.Shape, DType.F32);
            ToDenoised(backend, uncondDenoised, x, prediction.Uncond, sigma, predictor.Prediction);
            return true;
        }
        catch
        {
            denoised.Dispose();
            uncondDenoised?.Dispose();
            throw;
        }
        finally
        {
            combined.Dispose();
        }
    }

    /// <summary>Whether <paramref name="type"/> is a flow (ComfyUI <c>CONST</c>) model, whose half-log-SNR is the logit form.</summary>
    public static bool IsFlow(PredictionType type) => type is PredictionType.FlowVelocity or PredictionType.NegatedFlowVelocity;

    /// <summary>ComfyUI's <c>sigma_to_half_log_snr</c>: <c>log((1−σ)/σ)</c> for flow models, <c>−log σ</c> otherwise.</summary>
    public static double HalfLogSnr(double sigma, bool flow) => flow ? Math.Log((1.0 - sigma) / sigma) : -Math.Log(sigma);

    /// <summary>ComfyUI's <c>half_log_snr_to_sigma</c>, the inverse of <see cref="HalfLogSnr"/>.</summary>
    public static double HalfLogSnrToSigma(double lambda, bool flow) => flow ? 1.0 / (1.0 + Math.Exp(lambda)) : Math.Exp(-lambda);

    /// <summary><c>e^x − 1</c>, accurate near zero.</summary>
    public static double Expm1(double x) => Math.Abs(x) < 1e-5 ? x + (x * x * 0.5) + (x * x * x / 6.0) : Math.Exp(x) - 1.0;

    /// <summary><c>h·φ₁(h) = e^h − 1</c>.</summary>
    public static double PhiOne(double h) => Expm1(h);

    /// <summary><c>h·φ₂(h) = (e^h − 1 − h)/h</c>.</summary>
    public static double PhiTwo(double h) => (PhiOne(h) - h) / h;

    /// <summary>Double-precision ComfyUI <c>get_ancestral_step</c>; <paramref name="eta"/> 0 returns <c>(to, 0)</c>.</summary>
    public static (double SigmaDown, double SigmaUp) AncestralStep(double sigmaFrom, double sigmaTo, double eta)
    {
        if (eta == 0.0)
        {
            return (sigmaTo, 0.0);
        }
        double up = Math.Min(sigmaTo, eta * Math.Sqrt(sigmaTo * sigmaTo * ((sigmaFrom * sigmaFrom) - (sigmaTo * sigmaTo)) / (sigmaFrom * sigmaFrom)));
        return (Math.Sqrt((sigmaTo * sigmaTo) - (up * up)), up);
    }

    /// <summary>ComfyUI's rectified-flow ancestral split (<c>*_RF</c> samplers): the down-step sigma, the rescale
    /// <c>alpha_next/alpha_down</c> and the renoise coefficient.</summary>
    public static (double SigmaDown, double Rescale, double Renoise) FlowAncestralStep(double sigma, double sigmaNext, double eta)
    {
        double sigmaDown = sigmaNext * (1.0 + (((sigmaNext / sigma) - 1.0) * eta));
        double alphaNext = 1.0 - sigmaNext;
        double alphaDown = 1.0 - sigmaDown;
        double renoise = Math.Sqrt(Math.Max(0.0, (sigmaNext * sigmaNext) - (sigmaDown * sigmaDown * alphaNext * alphaNext / (alphaDown * alphaDown))));
        return (sigmaDown, alphaNext / alphaDown, renoise);
    }

    /// <summary>The ancestral split of a step: how much of the jump is deterministic (<c>sigmaDown</c>) versus resampled
    /// noise (<c>sigmaUp</c>), for a given <paramref name="eta"/>. k-diffusion's <c>get_ancestral_step</c>.</summary>
    public static (float SigmaDown, float SigmaUp) AncestralStep(float sigma, float sigmaNext, float eta = 1.0f)
    {
        if (sigmaNext <= 0f)
        {
            return (sigmaNext, 0f);
        }
        float ratio = (sigmaNext * sigmaNext) / (sigma * sigma);
        float sigmaUp = MathF.Min(sigmaNext, eta * MathF.Sqrt((sigmaNext * sigmaNext) * (1f - ratio)));
        float sigmaDown = MathF.Sqrt(MathF.Max(0f, (sigmaNext * sigmaNext) - (sigmaUp * sigmaUp)));
        return (sigmaDown, sigmaUp);
    }
}
