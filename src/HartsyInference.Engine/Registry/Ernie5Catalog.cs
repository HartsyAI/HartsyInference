namespace HartsyInference.Engine.Registry;

/// <summary>The <c>ernie-5.5-flash-base</c> catalog row: Baidu's ERNIE-5.5-Flash-Base pretraining checkpoint, pinned to the revision its header was inspected at.</summary>
internal static class Ernie5Catalog
{
    /// <summary>Catalog id of the model.</summary>
    internal const string Id = "ernie-5.5-flash-base";

    /// <summary>Builds the entry. It has no <see cref="CatalogEntry.Assets"/>, so the CLI never offers the ~288 GB download for it.</summary>
    internal static CatalogEntry Build() => new()
    {
        // Ernie5Model implements the architecture, MTP layer included, on the host and the Text path serves it on the CPU, but no
        // real-weight run has verified it, hence Structural.
        Id = Id, Modality = Modality.Text, DisplayName = "ERNIE-5.5-Flash-Base",
        Architecture = "37 HCA + 6 DSA attention layers, mHC (hc 4), SiTU-GLU latent MoE (512 routed experts, 10 active), multimax head; "
            + "1 MTP layer (host reference; not used by the Text path)",
        Status = ModelStatus.Structural, CliDrivable = false, HuggingFaceRepo = "baidu/ERNIE-5.5-Flash-Base",
        Variants =
        [
            new CatalogVariant
            {
                Id = "official", DisplayName = "Official (FP8 e4m3, 128x128 blocks)", Flavor = null,
                Components = CatalogComponents.Backbone,
                Source = new CatalogShardSet
                {
                    Repo = "baidu/ERNIE-5.5-Flash-Base", Revision = "028544b2ffceea189b2d8768ed2b9d09b06f2343",
                    TargetSubdir = $"LLM/{Id}", IndexFile = "model.safetensors.index.json",
                    FileCount = 128, TotalBytes = 288_434_400_632,
                },
                Notes = "128 safetensors shards sum to 288,434,400,632 bytes (268.6 GiB) per the HF tree API; the index metadata.total_size "
                    + "(288,416,689,496) is 17,711,136 bytes smaller and is not used. FP8 weights carry F32 [ceil(out/128), ceil(in/128)] weight_scale_inv grids "
                    + "(three shard headers inspected); HfQuantCheckpoint reads them directly, so no QuantFlavor applies. "
                    + "License 'ernie-5.5-flash-base-model-license-v1.0' grants use only outside the EU, UK and South Korea, and needs separate "
                    + "authorization for commercial use above US$50M annual revenue (LICENSE 1.11, 2, 4.3, 5.3).",
            },
        ],
    };
}
