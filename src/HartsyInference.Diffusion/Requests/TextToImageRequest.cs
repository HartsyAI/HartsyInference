using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Requests;

/// <summary>Request parameters for text-to-image generation.</summary>
public record TextToImageRequest
{
    /// <summary>The text prompt to generate an image from.</summary>
    public required string Prompt { get; init; }

    /// <summary>Optional negative prompt for classifier-free guidance.</summary>
    public string NegativePrompt { get; init; } = "";

    /// <summary>Number of denoising steps; <c>null</c> resolves via <c>Steps ?? modelDefault.Steps</c> (see <see cref="GenerationDefaults"/>).</summary>
    public int? Steps { get; init; }

    /// <summary>Classifier-free guidance scale (higher = more prompt adherence); <c>null</c> = model-specific default (<see cref="GenerationDefaults"/>).</summary>
    public float? CfgScale { get; init; }

    /// <summary>Output image width in pixels. Must be divisible by 8. <c>null</c> = model-specific default.</summary>
    public int? Width { get; init; }

    /// <summary>Output image height in pixels. Must be divisible by 8. <c>null</c> = model-specific default.</summary>
    public int? Height { get; init; }

    /// <summary>Random seed for reproducibility. Null = random.</summary>
    public int? Seed { get; init; }

    /// <summary>Scheduler to use. Null = default (Euler).</summary>
    public string? Scheduler { get; init; }

    /// <summary>"CLIP skip" — how many layers from the end of the CLIP text encoder to take hidden states from. 1 = standard last layer (default), 2 = penultimate (common for SD1.5 anime checkpoints). Null = 1. Only honored by pipelines whose text encoder is CLIP-final-layer based (SD 1.5); SDXL already uses penultimate by spec.</summary>
    public int? ClipSkip { get; init; }

    /// <summary>CFG-Rescale strength, 0..1 (0/null = off); see <see cref="Utilities.CfgHelper.ApplyCfgRescale"/>. Only consumed by pipelines that wire it in (SDXL as of 2026-08-10); ignored elsewhere.</summary>
    public float? CfgRescale { get; init; }

    /// <summary>TCFG (Tangential Damping CFG) toggle; see <see cref="Utilities.CfgHelper.ApplyTcfg"/>. Composes with <see cref="CfgRescale"/> (TCFG combine runs first, rescale applies to its output). Only consumed by pipelines that wire it in (SDXL as of 2026-08-11); ignored elsewhere.</summary>
    public bool? Tcfg { get; init; }

    /// <summary>Seamless-tileable axis: <c>null</c>/<c>"false"</c> = off, <c>"true"</c> = both axes, <c>"X-Only"</c>/<c>"Y-Only"</c> = one axis; same vocabulary as SwarmUI core's <c>SeamlessTileable</c> param (shared, carries its own <c>"seamless"</c> feature flag). When set, every conv in the request pads that axis with wrapped edge pixels instead of zeros so the output tiles continuously along it. Only consumed by pipelines that wire it in (SDXL as of 2026-08-11); ignored elsewhere.</summary>
    public string? SeamlessTiling { get; init; }

    /// <summary>Optional pre-built initial noise tensor. When non-null, overrides the seed-based noise generator — used for cross-runtime parity tests where the same noise tensor must flow into both PyTorch and HartsyInference (PyTorch's <c>torch.Generator.manual_seed</c> and HartsyInference's <c>SeedGenerator</c> use different RNGs and don't agree bit-for-bit on the same seed). Pipeline takes ownership and disposes after use. Shape must match the pipeline's expected initial latent shape (txt2img path; for img2img use <see cref="ImageToImageRequest.SourceImage"/>).</summary>
    public Tensor? InitialNoise { get; init; }

    /// <summary>Second seed for variation-seed blending (ComfyUI's SwarmKSampler semantics): the starting noise becomes <c>slerp(noise(Seed), noise(VariationSeed), VariationSeedStrength)</c>; negative draws a random variation seed. Only consulted when <see cref="VariationSeedStrength"/> &gt; 0. Blended in whatever space the pipeline seeds its own initial noise, which is what makes this architecture-agnostic (SD's 4×H/8 latents, DiT 16-channel latents, Chroma-Radiance pixels, Lens/Flux2 packed sequences alike). Ignored when <see cref="InitialNoise"/> is injected and on img2img.</summary>
    public long VariationSeed { get; init; } = -1;

    /// <summary>Blend factor for <see cref="VariationSeed"/>: 0 (default) = base seed exactly, 1 = the variation seed's noise entirely. Slerp keeps unit-Gaussian variance at every intermediate value.</summary>
    public double VariationSeedStrength { get; init; }
}
