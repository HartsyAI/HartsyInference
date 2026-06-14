namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>RoPE inverse-frequency scaling strategy. <see cref="None"/> = standard RoPE used by Llama / Qwen / Mistral / Gemma. <see cref="Yarn"/> = YaRN-extended RoPE used by GPT-OSS (and Llama-3.1-extended-context variants) — interpolates between extrapolation and interpolation across a smooth wavelength ramp so the model trained at <c>OriginalMaxPosition</c> can attend to <c>OriginalMaxPosition × YarnFactor</c> positions without catastrophic position-index drift.</summary>
public enum RopeScalingType
{
    /// <summary>Standard RoPE — inv_freq[k] = 1 / theta^(2k/D).</summary>
    None,

    /// <summary>YaRN scaling — inv_freq blends between extrapolation (high frequencies, kept original)
    /// and interpolation (low frequencies, divided by <c>YarnFactor</c>), with a smooth ramp determined
    /// by <c>YarnBetaFast</c> / <c>YarnBetaSlow</c> over <c>OriginalMaxPosition</c>.</summary>
    Yarn,
}
