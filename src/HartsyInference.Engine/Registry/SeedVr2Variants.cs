namespace HartsyInference.Engine.Registry;

/// <summary>The SeedVR2 assets the catalog's restore variants share, so each variant lists only its own transformer.</summary>
internal static class SeedVr2Variants
{
    /// <summary>Comfy-Org's repack repo: the same NaDiT keys as numz, with the layouts ComfyUI loads.</summary>
    public const string ComfyOrgRepo = "Comfy-Org/SeedVR2";

    /// <summary>The shared VAE. Comfy-Org's copy has the same bytes as numz's, so the name and hash stay the same.</summary>
    public static ModelAsset Vae() => new()
    {
        Repo = ComfyOrgRepo, RepoPath = "vae/seedvr2_ema_vae_fp16.safetensors",
        TargetSubdir = "VAE/SeedVR2", TargetName = "seedvr2_ema_vae_fp16.safetensors", Role = "vae",
        Sha256 = "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1",
    };

    /// <summary>The frozen positive embedding every variant reads, from the upstream ByteDance-Seed release.</summary>
    public static ModelAsset Embeddings() => new()
    {
        Repo = "ByteDance-Seed/SeedVR2-3B", RepoPath = "pos_emb.pt",
        TargetSubdir = "Video/SeedVr2", Role = "embeddings",
        Sha256 = "fa07a14844314772266b66c3b95deb0027696d8fe7065721263db5176f45d799",
    };

    /// <summary>The assets of a Comfy-Org variant: its transformer in diffusion_models/, the shared VAE and the embedding.</summary>
    public static ModelAsset[] ComfyOrg(string file, string sha256) =>
    [
        new() { Repo = ComfyOrgRepo, RepoPath = "diffusion_models/" + file, TargetSubdir = "diffusion_models", Role = "transformer", Sha256 = sha256 },
        Vae(),
        Embeddings(),
    ];
}
