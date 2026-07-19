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

    /// <summary>The on-disk file name (basename of <see cref="RepoPath"/>).</summary>
    public string FileName => Path.GetFileName(RepoPath);
}
