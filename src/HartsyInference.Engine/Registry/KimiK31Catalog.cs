namespace HartsyInference.Engine.Registry;

/// <summary>The <c>kimi-k3.1-pro</c> catalog row: Moonshot AI's Kimi K3.1 Pro checkpoint, pinned to the revision its files were read at.</summary>
internal static class KimiK31Catalog
{
    /// <summary>Catalog id of the model.</summary>
    internal const string Id = "kimi-k3.1-pro";

    /// <summary>Builds the entry. It has no <see cref="CatalogEntry.Assets"/>, so the CLI never offers the ~1.5 TB download for it.</summary>
    internal static CatalogEntry Build() => new()
    {
        // The text decoder is wired to the Text path and has no real-weight run, hence Structural and not CLI-drivable. The repo ships no
        // tokenizer.json, so the Text path refuses the checkpoint by name until one is supplied.
        Id = Id, Modality = Modality.Text, DisplayName = "Kimi K3.1 Pro",
        Architecture = "93 layers: KDA linear attention and NoPE MLA with DSA indexer (24 full layers), dense layer 0, "
            + "latent SiTU MoE (896 routed, 16 active, 2 shared), attention residuals; vision tower not run",
        Status = ModelStatus.Structural, CliDrivable = false, HuggingFaceRepo = "moonshotai/Kimi-K3.1-Pro",
        Variants =
        [
            new CatalogVariant
            {
                Id = "official", DisplayName = "Official (MXFP4 routed experts, BF16 dense)", Flavor = null,
                Components = CatalogComponents.Backbone,
                Refusals = new Dictionary<CatalogComponents, string>
                {
                    [CatalogComponents.Vision] = "Kimi K3.1 image input is refused: the Text path runs no vision tower (vision_tower.*, mm_projector.*).",
                },
                Source = new CatalogShardSet
                {
                    Repo = "moonshotai/Kimi-K3.1-Pro", Revision = "d9e99b0792a5e93b78ea9bea62a68167e36ddd9e",
                    TargetSubdir = $"LLM/{Id}", IndexFile = "model.safetensors.index.json",
                    FileCount = 96, TotalBytes = 1_561_217_377_280,
                },
                Notes = "96 safetensors shards; 1,561,217,377,280 bytes summed from the shard headers (docs/Research/KIMI_K31_ARCHITECTURE.md). "
                    + "The HF API reports usedStorage 1,561,355,732,732 and a safetensors total of 2,780,110,363,392; neither is used here. "
                    + "The repo ships tiktoken.model and tokenization_kimi.py and no tokenizer.json or chat_template.jinja. "
                    + "License 'Kimi K3.1-Pro License' (Moonshot AI, 2026, license name kimi-k3.1-pro, LICENSE file): use, copy, modify, merge, "
                    + "publish, distribute, sublicense and sell are granted, with the notice kept in copies; a model-as-a-service sale that lets "
                    + "third parties direct inputs needs a separate agreement only above US$20M aggregate revenue in any consecutive 12 months; "
                    + "products above 100M monthly active users or US$20M monthly revenue must show 'Kimi K3.1-Pro' in the UI; "
                    + "the software is provided as is.",
            },
        ],
    };
}
