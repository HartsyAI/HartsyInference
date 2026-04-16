using SharpInference.Core.Tensors;

namespace SharpInference.Core.Schedulers;

/// <summary>Interface for diffusion noise schedulers. Controls the denoising process by determining noise levels at each step and computing the update rule.</summary>
public interface IScheduler
{
    /// <summary>Name of this scheduler (e.g., "euler", "dpm++2m", "ddim").</summary>
    string Name { get; }

    /// <summary>Number of inference steps configured.</summary>
    int NumInferenceSteps { get; }

    /// <summary>The computed timestep schedule.</summary>
    ReadOnlySpan<float> Timesteps { get; }

    /// <summary>Configures the scheduler for the given number of inference steps. Must be called before <see cref="Step"/>.</summary>
    void SetTimesteps(int numInferenceSteps);

    /// <summary>Performs one denoising step: takes the model output and current noisy sample, returns the denoised (or less noisy) sample for the next step.</summary>
    /// <param name="output">Tensor to write the result into.</param>
    /// <param name="modelOutput">The denoiser's prediction for this timestep.</param>
    /// <param name="sample">The current noisy sample.</param>
    /// <param name="stepIndex">Index into the timestep schedule (0-based).</param>
    void Step(Tensor output, Tensor modelOutput, Tensor sample, int stepIndex);

    /// <summary>Adds noise to a clean sample according to the noise level at the given timestep. Used for img2img (start denoising from a partially noisy image).</summary>
    /// <param name="output">Tensor to write the noisy sample into.</param>
    /// <param name="sample">The clean sample.</param>
    /// <param name="noise">Random noise tensor (same shape as sample).</param>
    /// <param name="stepIndex">Index into the timestep schedule.</param>
    void AddNoise(Tensor output, Tensor sample, Tensor noise, int stepIndex);

    /// <summary>Returns the initial noise scale factor. For most schedulers this is 1.0, but some (like Karras) scale the initial noise.</summary>
    float InitialNoiseSigma { get; }

    /// <summary>Returns the scale factor to apply to the model input at the given step. For Euler schedulers this is 1/sqrt(sigma^2 + 1), for most others it's 1.0 (no scaling).</summary>
    /// <param name="stepIndex">Index into the timestep schedule (0-based).</param>
    float ScaleModelInput(int stepIndex) => 1.0f;
}
