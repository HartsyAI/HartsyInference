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
    public static void CombineCfg(IBackend backend, Tensor output, in DenoisePrediction prediction, float guidance)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(output);
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
            CombineCfg(backend, combined, prediction, predictor.GuidanceScale);
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
