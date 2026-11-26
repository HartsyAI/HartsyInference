namespace HartsyInference.Engine.Registry;

/// <summary>The <c>maxwell-v1-flash</c> catalog row: Maxwell-V1.0-Flash, a 35B / 3B-active MoE with mHC residual streams and block-sparse GQA attention,
/// served from its official FP8 checkpoint at one pinned revision.</summary>
internal static class MaxwellCatalog
{
    /// <summary>Catalog id of the model.</summary>
    internal const string Id = "maxwell-v1-flash";

    /// <summary>Builds the entry. Drivable from the CLI and the native API through TextService, which runs the real FP8 checkpoint on CUDA. Still
    /// <see cref="ModelStatus.ValidationPending"/>: no reference tokens have been compared on the real weights (see MODEL_STATUS_LLM.md).</summary>
    internal static CatalogEntry Build() => new()
    {
        Id = Id, Modality = Modality.Text, DisplayName = "Maxwell-V1.0-Flash",
        Architecture = "45 layers, hidden 2048; block-sparse GQA (32 query / 4 KV heads, 128-token blocks, 8 prefix + 8 local + 16 remote, per-layer indexer); "
            + "mHC with 4 residual streams; sigmoid MoE (256 routed, top-12, one shared expert, 3 dense layers); 1 MTP layer (not loaded)",
        Status = ModelStatus.ValidationPending, CliDrivable = true, HuggingFaceRepo = "maxwell-model/Maxwell-V1-Flash",
        Variants =
        [
            new CatalogVariant
            {
                Id = "official-fp8", DisplayName = "Official (FP8 e4m3, 128x128 blocks)", Flavor = null,
                Components = CatalogComponents.Backbone,
                Source = new CatalogShardSet
                {
                    Repo = "maxwell-model/Maxwell-V1-Flash-FP8", Revision = "6f91969de0d39c9fdbb98306e8826ba4f661125b",
                    TargetSubdir = $"LLM/{Id}", Files = ["model.safetensors"], FileCount = 1, TotalBytes = 38_588_885_068,
                    SupportFiles =
                    [
                        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
                        "chat_template.jinja", "LICENSE",
                    ],
                    Sha256ByFile = new Dictionary<string, string>
                    {
                        ["model.safetensors"] = "3b8b158c1a0656bda3642904a0c5040da7d901fec6167ebff26bf41d69c741ff",
                    },
                },
                Notes = "One model.safetensors of 38,588,885,068 bytes (F8_E4M3 weights with F32 128x128 weight_scale_inv grids; 1,015 tensors kept BF16 "
                    + "per modules_to_not_convert, expert_bias F32). Licence: Apache License 2.0, full text in LICENSE, no territory or commercial limit. "
                    + "The BF16 repo maxwell-model/Maxwell-V1-Flash (73.6 GB) is the upstream reference and is not in the catalog.",
            },
        ],
    };
}
