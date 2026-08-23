namespace HartsyInference.ModelAssets.Registry;

/// <summary>Diffusion model architecture families that <see cref="ModelArchitectureDetector"/> can recognize from a checkpoint's tensor-key layout. This is intentionally a coarse family identifier (it does not distinguish, say, Flux dev from Flux schnell) — its job is to pick the right pipeline constructor / checkpoint converter, not to capture every variant.</summary>
public enum ModelArchitecture
{
    /// <summary>Could not be matched to a known family.</summary>
    Unknown = 0,

    /// <summary>Stable Diffusion 1.x (LDM single-file: <c>cond_stage_model.</c> + <c>model.diffusion_model.</c>).</summary>
    StableDiffusion15,

    /// <summary>SDXL base (dual CLIP: <c>conditioner.embedders.0/1</c> + ADM <c>label_emb</c>).</summary>
    Sdxl,

    /// <summary>SDXL refiner (single CLIP-G embedder, ADM <c>label_emb</c>, no <c>conditioner.embedders.1</c>).</summary>
    SdxlRefiner,

    /// <summary>Stable Diffusion 3 / 3.5 (MMDiT <c>joint_blocks</c> / <c>text_encoders.*</c>).</summary>
    StableDiffusion3,

    /// <summary>Flux.1 dev/schnell (<c>double_blocks</c> + <c>single_blocks</c>).</summary>
    Flux1,

    /// <summary>Flux.2 (single-stream modulation keys, e.g. <c>double_stream_modulation_img</c>).</summary>
    Flux2,

    /// <summary>AuraFlow (<c>double_layers</c> / <c>single_layers</c>).</summary>
    AuraFlow,

    /// <summary>Chroma (Flux-lineage, distilled, radiance/nerf keys).</summary>
    Chroma,
}
