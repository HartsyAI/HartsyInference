namespace HartsyInference.Engine.Registry;

/// <summary>The <c>moss-vl-realtime</c> catalog row: OpenMOSS's MOSS-VL-Realtime (Apache-2.0), an 11B cross-attention VLM. The
/// default download is the community Q4_K GGUF, which carries the vision tower inside it, plus the tokenizer and JSON configs from
/// the official repository (the GGUF embeds no tokenizer). Hashes were checked against the pinned revisions on 2026-11-09.</summary>
internal static class MossVlCatalog
{
    /// <summary>Catalog id of the model.</summary>
    internal const string Id = "moss-vl-realtime";

    private const string Folder = "LLM/moss-vl-realtime";
    private const string OfficialRepo = "OpenMOSS-Team/MOSS-VL-Realtime-SGLANG";
    private const string OfficialRevision = "3d6407cc65c3a32722e852d56d563ce087f98fdf";
    private const string GgufRepo = "SkyContact/MOSS-VL-Realtime-GGUF";
    private const string GgufRevision = "dca2156a7a19c3daa4841c368bc8944e60d20b4f";

    /// <summary>Builds the entry.</summary>
    internal static CatalogEntry Build() => new()
    {
        // ValidationPending: the GGUF generates correct replies end to end on Vulkan (CLI and tests) and its layers match the BF16
        // reference within 5e-3; the realtime session, video input and CUDA runs are still being built and checked.
        Id = Id, Modality = Modality.Text, DisplayName = "MOSS-VL-Realtime (11B)",
        Architecture = "48-layer Qwen3 decoder with 12 gated cross-attention layers; Qwen3-VL ViT with a unified deepstack merger",
        Status = ModelStatus.ValidationPending, CliDrivable = true, HuggingFaceRepo = OfficialRepo,
        Assets =
        [
            new() { Repo = GgufRepo, RepoPath = "Moss-VL-Realtime-q4k.gguf", TargetSubdir = Folder, Role = "transformer",
                Sha256 = "605470f81a2ee13aab8f22c36d89873302feb1affb0f19f593d9916a982374f4" },
            new() { Repo = OfficialRepo, RepoPath = "tokenizer.json", TargetSubdir = Folder, Role = "tokenizer",
                Sha256 = "e7bbd0f9784004df51aca562befc3c7a8f294b4045aa8685536c35804c9aa493" },
            new() { Repo = OfficialRepo, RepoPath = "config.json", TargetSubdir = Folder, Role = "config",
                Sha256 = "1da9331956a8e58edb261b85624ba191a06ea6ac69d5f8debb0ef4c7fb257bd6" },
            new() { Repo = OfficialRepo, RepoPath = "preprocessor_config.json", TargetSubdir = Folder, Role = "config",
                Sha256 = "94f62f9fd2d76b66f0a388d688c26ba639d555d4fa69c18efaa483a7f9ed62a7" },
            new() { Repo = OfficialRepo, RepoPath = "video_preprocessor_config.json", TargetSubdir = Folder, Role = "config",
                Sha256 = "f6828441758b19187559988fabac496c478fd3a6b684b1dd674f9f01fd5fcba4" },
            new() { Repo = OfficialRepo, RepoPath = "chat_template.json", TargetSubdir = Folder, Role = "chat template",
                Sha256 = "d272c68bd115dca51b7e918dabac29b23510e3a25c4c5d9bd410b96eb11bf26e" },
        ],
        Variants =
        [
            new CatalogVariant
            {
                Id = "gguf-q4k", DisplayName = "GGUF Q4_K_M (community, vision tower inside)", Flavor = null,
                Components = CatalogComponents.Backbone | CatalogComponents.Vision,
                Source = new CatalogShardSet
                {
                    Repo = GgufRepo, Revision = GgufRevision, TargetSubdir = Folder,
                    FileCount = 1, TotalBytes = 8_423_234_016, Files = ["Moss-VL-Realtime-q4k.gguf"],
                },
                Notes = "Text projections Q4_K/Q6_K, cross-attention and head Q6_K, vision F16. The converter pads the vision qkv "
                    + "(q/k per rotary half 36->40 rows, v per head 72->80); MossVlVisionTower.OpenGguf removes the padding. No tokenizer "
                    + $"is embedded: tokenizer.json comes from {OfficialRepo}.",
            },
            new CatalogVariant
            {
                Id = "bf16", DisplayName = "Official BF16 safetensors", Flavor = null,
                Components = CatalogComponents.Backbone | CatalogComponents.Vision,
                Source = new CatalogShardSet
                {
                    Repo = OfficialRepo, Revision = OfficialRevision, TargetSubdir = Folder,
                    IndexFile = "model.safetensors.index.json", FileCount = 5, TotalBytes = 22_672_852_616,
                },
                Notes = "Needs about 21 GB of device memory at 16 bits (CUDA keeps BF16; Vulkan casts once to F16), or about 42 GB of "
                    + "host RAM on the CPU, where projections are widened to F32. Apache-2.0.",
            },
        ],
    };
}
