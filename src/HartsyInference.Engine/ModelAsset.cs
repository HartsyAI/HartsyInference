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

    /// <summary>Human label for what this file is: "transformer", "text encoder", "vae", or <see cref="LowNoiseExpertRole"/>.</summary>
    public required string Role { get; init; }

    /// <summary>The <see cref="Role"/> of a mixture-of-experts entry's second transformer (Wan 2.2 A14B's low-noise expert). It
    /// downloads with the set, and the engine passes it as the swap model when the caller names none.</summary>
    public const string LowNoiseExpertRole = "low-noise expert";

    /// <summary>Canonical save name when it differs from the repo file name (a side model is often published under
    /// a different name, e.g. save "t5xxl_fp8_e4m3fn.safetensors" as "t5xxl_enconly.safetensors"). May contain a
    /// nested subfolder segment (e.g. "Flux/ae.safetensors"). Null keeps the repo file name.</summary>
    public string? TargetName { get; init; }

    /// <summary>SHA-256 hex hash for post-download verification; null/empty skips verification (discouraged — a
    /// size-correct but byte-corrupt file loads as NaN).</summary>
    public string? Sha256 { get; init; }

    /// <summary>Direct https URL of a file that is not in a Hugging Face repository (a GitHub release asset). When set, the file is fetched from here, never with the Hugging Face token, and <see cref="Sha256"/> is required.
    /// <see cref="Repo"/> and <see cref="RepoPath"/> then only name the source for messages.</summary>
    public string? Url { get; init; }

    /// <summary>The safetensors repack of this (pickle) file that a weights mirror hosts, or null when none is. Fetched from the mirror named by <c>paths.weightsMirrorUrl</c> first; the upstream file stays the fallback.</summary>
    public AssetRepack? Repack { get; init; }

    /// <summary>Audio cache category to fetch into, when it is not the one this model's modality implies.
    /// SheetSage2 transcribes, so its modality says <c>"stt"</c>, but its weights ship inside the YuE2 repo and
    /// belong beside it under <c>"music"</c> — and the loader looks there, so a prefetch into <c>"stt"</c> would
    /// download the file a second time and still leave an offline run without it.</summary>
    public string? AudioCategory { get; init; }

    /// <summary>Names this asset was saved under before <see cref="TargetName"/> was changed, in the same
    /// models-root-relative form. An install that already holds the file under one of these keeps using it
    /// instead of re-downloading gigabytes to the new name — and, just as importantly, recipes that resolve
    /// this asset through the strict non-downloading overload keep finding it. Empty for an asset whose name
    /// never moved.</summary>
    public IReadOnlyList<string> LegacyTargetNames { get; init; } = [];

    /// <summary>The on-disk file name (<see cref="TargetName"/> when set, else the basename of <see cref="RepoPath"/>).</summary>
    public string FileName => TargetName ?? Path.GetFileName(RepoPath);
}
