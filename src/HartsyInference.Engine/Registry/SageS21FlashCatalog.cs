namespace HartsyInference.Engine.Registry;

/// <summary>The <c>sage-s2.1-flash</c> catalog row: the Shanghai AI Laboratory (InternLM) Sage S2.1 Flash checkpoints, pinned to the revisions their headers, index and configs were read at.</summary>
internal static class SageS21FlashCatalog
{
    /// <summary>Catalog id of the model.</summary>
    internal const string Id = "sage-s2.1-flash";

    /// <summary>Builds the entry. It has no <see cref="CatalogEntry.Assets"/>, so the CLI never offers the 329 to 643 GB download for it.</summary>
    internal static CatalogEntry Build() => new()
    {
        // The decoder matches transformers' glm5_next on a tiny random model (docs/Checklists/PARITY_VERIFICATION.md); no real weights have run
        // (320B total, 18B active), hence Structural and not CLI-drivable.
        Id = Id, Modality = Modality.Text, DisplayName = "Sage S2.1 Flash",
        Architecture = "45 layers: 34 Kimi Delta Attention and 11 NoPE MLA layers with a compressed-pool DSA indexer, mHC (hc 4), 3 dense layers then "
            + "clamped-SwiGLU sigmoid MoE (288 routed, 8 active, 1 shared), MTP layer (not run), vision tower and scientific-data projectors (not wired to the Text path)",
        Status = ModelStatus.Structural, CliDrivable = false, HuggingFaceRepo = "internlm/Sage-S2.1-Flash",
        Variants =
        [
            new CatalogVariant
            {
                Id = "bf16", DisplayName = "Official BF16", Flavor = null,
                Components = CatalogComponents.Backbone,
                Refusals = Refusals(),
                Source = new CatalogShardSet
                {
                    Repo = "internlm/Sage-S2.1-Flash", Revision = "eacc61b34a1af2ecb0c764e099152c46e076169b",
                    TargetSubdir = $"LLM/{Id}", IndexFile = "model.safetensors.index.json",
                    FileCount = 120, TotalBytes = 642_658_755_280,
                },
                Notes = Notes("120 safetensors shards sum to 642,658,755,280 bytes per the HF tree API; the index metadata.total_size (642,653,338,488) is smaller and is not used."),
            },
            new CatalogVariant
            {
                Id = "fp8", DisplayName = "Official FP8 (e4m3, 128x128 blocks)", Flavor = null,
                Components = CatalogComponents.Backbone,
                Refusals = Refusals(),
                Source = new CatalogShardSet
                {
                    Repo = "internlm/Sage-S2.1-Flash-FP8", Revision = "5717cf2fb5c2a54e9c123334f78029d5eadc09cd",
                    TargetSubdir = $"LLM/{Id}-fp8", IndexFile = "model.safetensors.index.json",
                    FileCount = 62, TotalBytes = 328_344_148_976,
                },
                Notes = Notes("62 safetensors shards sum to 328,344,148,976 bytes per the HF tree API; the index metadata.total_size (328,333,456,248) is smaller and is not used. "
                    + "Dense, shared and routed FFN weights and the DSA q_a, q_b, kv_a and o projections are FP8 with F32 weight_scale_inv grids; everything else stays BF16 or F32."),
            },
        ],
    };

    private static Dictionary<CatalogComponents, string> Refusals() => new()
    {
        [CatalogComponents.Vision] = "Sage S2.1 Flash image input is refused: the vision tower and the scientific projectors exist in the engine but the Text path does not splice their rows yet.",
    };

    private static string Notes(string sizes) => sizes
        + " The repo ships tokenizer.json, tokenization_glm5.py, chat_template.jinja and 8 scientific projectors under projectors/sage-s2.1-flash/ (the scientific encoders "
        + "themselves, such as CHGNet and ESM-2, are separate dependencies and are not shipped or ported). "
        + "License MIT per the model card (\"The code and model weights in this repository are released under the MIT License.\"); the repo has no LICENSE file, "
        + "and the separate Sage-SciPerception repo states \"License metadata has not yet been specified for this release\".";
}
