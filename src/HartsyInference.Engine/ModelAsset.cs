namespace HartsyInference.Engine;

/// <summary>One downloadable file a model needs (the transformer/checkpoint, a text encoder, or a VAE), with the
/// HuggingFace source and the models-root-relative folder it belongs in. A catalog entry lists all the assets that
/// make a model runnable, so selecting it can fetch the complete set — SwarmUI-style — instead of the user hunting
/// down each component.</summary>
public sealed record ModelAsset
{
    /// <summary>HuggingFace repo id, e.g. "Comfy-Org/Krea-2".</summary>
    public required string Repo { get; init; }

    /// <summary>Path of the file within the repo, e.g. "diffusion_models/krea2_turbo_fp8_scaled.safetensors".</summary>
    public required string RepoPath { get; init; }

    /// <summary>Destination folder relative to the models root, e.g. "Stable-Diffusion/Krea2" or "VAE/QwenImage".</summary>
    public required string TargetSubdir { get; init; }

    /// <summary>Human label for what this file is: "transformer", "text encoder", or "vae".</summary>
    public required string Role { get; init; }

    /// <summary>Canonical save name when it differs from the repo file name (a side model is often published under
    /// a different name, e.g. save "t5xxl_fp8_e4m3fn.safetensors" as "t5xxl_enconly.safetensors"). May contain a
    /// nested subfolder segment (e.g. "Flux/ae.safetensors"). Null keeps the repo file name.</summary>
    public string? TargetName { get; init; }

    /// <summary>SHA-256 hex hash for post-download verification; null/empty skips verification (discouraged — a
    /// size-correct but byte-corrupt file loads as NaN).</summary>
    public string? Sha256 { get; init; }

    /// <summary>The on-disk file name (<see cref="TargetName"/> when set, else the basename of <see cref="RepoPath"/>).</summary>
    public string FileName => TargetName ?? Path.GetFileName(RepoPath);
}
