namespace HartsyInference.Diffusion.Schedulers;

/// <summary>Stable Audio Open 1.0's continuous v-prediction (k-diffusion <c>VDenoiser</c>, σ_data = 1), as
/// <c>stable_audio_tools.inference.generation.generate_diffusion_cond</c> drives it at the model card's settings:
/// <c>get_sigmas_polyexponential(steps, sigma_min 0.3, sigma_max 500, rho 1)</c>, the model fed
/// <c>x / √(σ²+1)</c> at time <c>atan(σ)·2/π</c>, and the output read as <c>v</c>. <see cref="Sampling.SamplerMath.ToDenoised"/>
/// turns <c>v</c> into the denoised estimate (<c>x/(σ²+1) − σ·v/√(σ²+1)</c>), so any sampler can step it.</summary>
public static class StableAudioEdm
{
    /// <summary>The model card's lowest noise level.</summary>
    public const float DefaultSigmaMin = 0.3f;

    /// <summary>The model card's highest noise level.</summary>
    public const float DefaultSigmaMax = 500f;

    /// <summary>k-diffusion's <c>get_sigmas_polyexponential</c>: <paramref name="steps"/> levels from
    /// <paramref name="sigmaMax"/> down to <paramref name="sigmaMin"/>, log-spaced along <c>ramp^rho</c>, then a final 0.
    /// With <paramref name="rho"/> 1 (stable-audio-tools' default) the spacing is exponential.</summary>
    public static float[] Sigmas(int steps, float sigmaMin = DefaultSigmaMin, float sigmaMax = DefaultSigmaMax, double rho = 1.0)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(steps, 1);
        if (!(sigmaMin > 0 && sigmaMax > sigmaMin))
            throw new ArgumentException($"Need 0 < sigmaMin < sigmaMax; got {sigmaMin}, {sigmaMax}.");
        float[] sigmas = new float[steps + 1];
        double logMin = Math.Log(sigmaMin), logMax = Math.Log(sigmaMax);
        for (int i = 0; i < steps; i++)
        {
            // torch.linspace(1, 0, n) ** rho
            double ramp = steps == 1 ? 1.0 : 1.0 - (double)i / (steps - 1);
            sigmas[i] = (float)Math.Exp(Math.Pow(ramp, rho) * (logMax - logMin) + logMin);
        }
        sigmas[steps] = 0f;
        return sigmas;
    }

    /// <summary><c>VDenoiser.sigma_to_t</c>: the model's time input for noise level <paramref name="sigma"/>, in [0, 1).</summary>
    public static float TimestepOf(float sigma) => (float)(Math.Atan(sigma) / Math.PI * 2.0);

    /// <summary><c>c_in</c>: the factor the noisy latent is scaled by before the model sees it.</summary>
    public static float InputScale(float sigma) => (float)(1.0 / Math.Sqrt((double)sigma * sigma + 1.0));
}
