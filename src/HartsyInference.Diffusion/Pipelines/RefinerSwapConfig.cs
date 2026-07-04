using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Optional add-on for <see cref="SdxlPipeline.GenerateFromTokens"/> that turns
/// the standard base-only denoise loop into a Comfy-style "StepSwap": run base UNet for
/// the first <c>(1 - Strength)</c> fraction of steps, then mid-loop swap to the refiner
/// UNet for the remainder. Both halves share the same latent state — no VAE round-trip
/// in between (which is what <see cref="SdxlRefinerPipeline"/>'s PostApply mode does).
///
/// <para>The refiner UNet has a different shape than base (4 levels, CrossAttentionDim=1280
/// from CLIP-G alone, AdmInChannels=2560 with aesthetic_score conditioning). The pipeline
/// slices CLIP-G out of the base's concatenated text embedding for the refiner phase and
/// rebuilds ADM scalars with the user's aesthetic scores. Caller is responsible for
/// constructing the refiner UNet with <see cref="UNetConfig.SdxlRefiner"/> and loading
/// its weights before passing here.</para>
///
/// <para><b>Why this beats PostApply:</b> PostApply re-encodes the base output via VAE,
/// which is lossy and slower. StepSwap keeps the latent in float space across the
/// hand-off, so the refiner sees exactly what the base produced — same latent norm,
/// same noise schedule position. Visible quality difference at low refiner strengths
/// (0.15–0.25) where PostApply's VAE round-trip muddies fine details.</para></summary>
public sealed record RefinerSwapConfig
{
    /// <summary>Loaded SDXL refiner UNet, configured with <see cref="UNetConfig.SdxlRefiner"/>. Caller owns the lifetime; the pipeline only borrows it during one <c>GenerateFromTokens</c> call.</summary>
    public required UNet RefinerUnet { get; init; }

    /// <summary>Fraction of <i>total</i> steps the refiner runs (at the END of the schedule). <c>0.2</c> means the last 20% of steps use the refiner; <c>0</c> disables the swap entirely (treated as no refiner). Comfy and Swarm both call this <c>RefinerControl</c> / <c>Refiner Strength</c>.</summary>
    public required float Strength { get; init; }

    /// <summary>Aesthetic score for the cond branch (positive prompt). SDXL refiner was trained with values around <c>6.0</c> on the conditioning side; higher pushes outputs toward the model's "more aesthetic" prior.</summary>
    public float AestheticScore { get; init; } = 6.0f;

    /// <summary>Aesthetic score for the uncond branch (negative prompt). Trained around <c>2.5</c>; CFG steers away from this so lower values amplify the "more aesthetic" pull on the conditional side.</summary>
    public float NegativeAestheticScore { get; init; } = 2.5f;

    /// <summary>Optional CLIP-G penultimate hidden states <c>[2, S, 1280]</c> (uncond, cond) encoded from a
    /// separate <c>&lt;refiner&gt;</c> prompt. When set, the refiner phase cross-attends to this instead of the
    /// base prompt's CLIP-G conditioning (SwarmUI's <c>&lt;refiner&gt;</c> prompt syntax — a distinct prompt for the
    /// refine stage). Null → the refiner reuses the base prompt's CLIP-G (previous behavior). The caller owns
    /// the tensor's lifetime; the pipeline only borrows it for the one <c>GenerateFromTokens</c> call.</summary>
    public Tensor? RefinerConditioning { get; init; }
}
