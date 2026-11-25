namespace HartsyInference.Engine.Registry;

/// <summary>The Tencent Hy4 Flash catalog rows: the instruct model, its FP8 build and the pre-trained base, each pinned to the revision its files were inspected at.</summary>
internal static class Hy4FlashCatalog
{
    /// <summary>Catalog id of the BF16 instruct model.</summary>
    internal const string Id = "hy4-flash";

    /// <summary>Catalog id of the FP8 instruct model.</summary>
    internal const string FP8Id = "hy4-flash-fp8";

    /// <summary>Catalog id of the pre-trained base model.</summary>
    internal const string BaseId = "hy4-flash-base";

    /// <summary>The BF16 instruct row. It has no <see cref="CatalogEntry.Assets"/>, so the CLI never offers the 287 to 440 GB downloads.</summary>
    internal static CatalogEntry Instruct() =>
        Row(Id, "Hy4 Flash", "tencent/Hy4-flash", "eccd3123eec73f2040f4c9ed0de1883593470510", "official", "Official (BF16)", 440_520_833_910,
            "Instruct model; text and image input. The text path runs the checkpoint on the host reference as F32 and only checkpoints up to 1 GiB load there, so "
            + "this one is refused by name; the device path is not built.");

    /// <summary>The FP8 instruct row: the routed experts and attention projections in FP8 with BF16 block scales.</summary>
    internal static CatalogEntry Fp8() =>
        Row(FP8Id, "Hy4 Flash FP8", "tencent/Hy4-flash-FP8", "8a1b19890d3533aefc730f8b41f4b55f50cc31c0", "official", "Official (FP8 e4m3, 128x128 blocks)",
            287_073_129_974, "Instruct model with the routed experts and attention projections in FP8 (BF16 block scales). Same refusal as the BF16 build on the host reference.");

    /// <summary>The pre-trained base row.</summary>
    internal static CatalogEntry Base() =>
        Row(BaseId, "Hy4 Flash Base", "tencent/Hy4-flash-base", "95b9f8239e2f63cb837c91704f070ae47332ebdf", "official", "Official (BF16)", 438_829_461_044,
            "Pre-trained base: no chat roles in its chat_template.jinja, so the instruct template is used as the checkpoint ships it. Same refusal as the BF16 build on the host reference.");

    private static CatalogEntry Row(string id, string displayName, string repo, string revision, string variantId, string variantName, long bytes, string notes) => new()
    {
        // Hy4Flash has the config reader, key map, the host reference text forward matching the Hugging Face classes on a tiny config (see PARITY_VERIFICATION),
        // and the Text path wiring. No real checkpoint has run, so Structural.
        Id = id, Modality = Modality.Text, DisplayName = displayName,
        Architecture = "42 layers: gated MLA (indexed layers with a compressed group indexer, sliding layers), identity hyper-connections (4 streams), "
            + "latent MoE (512 routed experts, 16 active, clamped SwiGLU) with Engram n-gram memories on four layers; 1M-token context",
        Status = ModelStatus.Structural, CliDrivable = false, HuggingFaceRepo = repo,
        Variants =
        [
            new CatalogVariant
            {
                Id = variantId, DisplayName = variantName, Flavor = null,
                Components = CatalogComponents.Backbone,
                Source = new CatalogShardSet
                {
                    Repo = repo, Revision = revision,
                    TargetSubdir = $"LLM/{id}", IndexFile = "model.safetensors.index.json",
                    FileCount = 148, TotalBytes = bytes,
                },
                // Licence: Tencent Hy License 1.0 (Apache-2.0 derived). Section 10: a MaaS operator needs a separate written licence above US$50M worldwide gross
                // revenue in any 12 months or a product above 100M monthly active users; no territory limit. Internal use is exempt.
                Notes = notes + " Licence: Tencent Hy License 1.0, section 10 (separate licence for MaaS operators above US$50M revenue or 100M MAU).",
            },
        ],
    };
}
