using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>One IP-Adapter's contribution to a generation: the loaded adapter (per-cross-attn-layer K_ip / V_ip), the projected image-prompt tokens (computed once per generation outside the denoise loop, possibly averaged across multiple reference images), a strength multiplier, weight-type semantics, and a step-fraction window during which the adapter is active.
///
/// <para>The pipeline consumes this record once per generation and builds a per-step, per-cross-attention-layer scale array based on:
/// <list type="bullet">
///   <item><see cref="Scale"/> as the base multiplier.</item>
///   <item><see cref="WeightType"/> selecting a per-cross-attn-layer weighting profile ("standard" = uniform, "prompt is more important" = ramp down toward decoder, "style transfer" = only style-relevant blocks). All values are heuristics — Cubiq's IPAdapterPlus distinguishes 15+ weight schedules; we map Swarm's 3-option dropdown to documented approximations.</item>
///   <item><see cref="StartFraction"/> / <see cref="EndFraction"/> — the adapter contributes nothing for steps outside <c>[start, end]</c> of the schedule (multiplies all layer scales by 0 in that range). Lets the user use IPA only at the beginning (composition phase) or only at the end (style refinement phase).</item>
/// </list></para>
///
/// <para>Caller is responsible for:
/// <list type="bullet">
///   <item>Encoding the reference image(s) through CLIP-Vision into the right form (CLS-projected for standard, penultimate hidden states for Plus). For multi-image input, average the projected tokens (or the raw vision outputs) into one set of <see cref="ImageTokens"/> — diffusers does this; multi-image IPA is single-conditioning-with-averaged-tokens, not per-image-stacking.</item>
///   <item>Running <see cref="IpAdapter.ProjectImage"/> on that to produce <see cref="ImageTokens"/>.</item>
///   <item>Disposing the returned tokens after the generation completes.</item>
/// </list></para>
/// </summary>
public sealed record IpAdapterConditioning
{
    /// <summary>The loaded IP-Adapter (already weights-loaded; pipelines borrow its per-layer K_ip/V_ip during the denoise loop).</summary>
    public required IpAdapter Adapter { get; init; }

    /// <summary>Pre-projected image-prompt tokens from <see cref="IpAdapter.ProjectImage"/>: shape <c>[B, numTokens, crossAttnDim]</c>. Owned by caller. When the user supplied multiple reference images, the resolver averages their projected tokens into this single tensor before passing through.</summary>
    public required Tensor ImageTokens { get; init; }

    /// <summary>IPA strength multiplier. <c>0</c> = adapter contributes nothing, <c>1.0</c> = full strength (default training scale). Comfy / A1111 expose this as "IP-Adapter Weight" or "IP-Adapter Strength".</summary>
    public float Scale { get; init; } = 1.0f;

    /// <summary>Weight-type selector: one of <c>"standard"</c> (uniform per-layer scale), <c>"prompt is more important"</c> (down-weight encoder/mid layers so the prompt drives subject/composition while IPA contributes mainly to style refinement at decoder layers), or <c>"style transfer"</c> (zero out encoder-side, apply scale only at the layers most associated with style — typically mid + first up block for SDXL). The exact per-layer profile is computed inside <see cref="HartsyInference.Diffusion.Pipelines.SdxlPipeline"/> when the adapter is wired up; values other than the three known strings fall back to "standard".</summary>
    public string WeightType { get; init; } = "standard";

    /// <summary>Step-fraction at which the adapter starts contributing. <c>0.0</c> = from the very first step; <c>0.5</c> = only after halfway through the schedule. Must be less than <see cref="EndFraction"/>. Implemented by zeroing the per-layer scale array on steps outside <c>[start, end]</c>.</summary>
    public float StartFraction { get; init; } = 0.0f;

    /// <summary>Step-fraction at which the adapter stops contributing. <c>1.0</c> = through the last step; <c>0.5</c> = only for the first half. Must be greater than <see cref="StartFraction"/>.</summary>
    public float EndFraction { get; init; } = 1.0f;
}
