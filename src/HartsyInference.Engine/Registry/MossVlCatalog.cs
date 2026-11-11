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
    private const string Nf4Repo = "OpenMOSS-Team/MOSS-VL-Realtime-NF4";
    private const string Nf4Revision = "e0aa38db02c9cc0653ad85fed4e6f21db88049ba";
    private const string Fp8Repo = "OpenMOSS-Team/MOSS-VL-Realtime-FP8";
    private const string Fp8Revision = "c2f10d6ad3b7b00a53b77753889d20c393957bf9";
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
                    Repo = OfficialRepo, Revision = OfficialRevision, TargetSubdir = Folder + "-bf16",
                    IndexFile = "model.safetensors.index.json", FileCount = 5, TotalBytes = 22_672_852_616,
                    SupportFiles = SupportFiles,
                    Sha256ByFile = new Dictionary<string, string>
                    {
                        ["model.safetensors.index.json"] = "25dc84c05625b956cfdb5fb5dd571ed4c725d9f42f42affd223d220d23bd0726",
                        ["model-00001-of-00005.safetensors"] = "3fe2d46a92e3c036e8dfbb2519f65415f8f1ab4f731ffdcb9c8bbff4e66fc3d6",
                        ["model-00002-of-00005.safetensors"] = "18f3579907933bad721053b3ac405c51045db4539fedcc7ee3c8db79e79f70e5",
                        ["model-00003-of-00005.safetensors"] = "cecf1e02afe5c7250bb43d6b6e99c7cb430a02afc07a2d697ab1dc4b5f784567",
                        ["model-00004-of-00005.safetensors"] = "57d794a266a1283d209d054114d9888a6a6b36a0af87fbf314c92b76c89dc294",
                        ["model-00005-of-00005.safetensors"] = "34009172ba732eb447caf1293126bba35eb934a622677ac7c24bb2d6af462b0d",
                        ["tokenizer.json"] = "e7bbd0f9784004df51aca562befc3c7a8f294b4045aa8685536c35804c9aa493",
                        ["config.json"] = "1da9331956a8e58edb261b85624ba191a06ea6ac69d5f8debb0ef4c7fb257bd6",
                        ["preprocessor_config.json"] = "94f62f9fd2d76b66f0a388d688c26ba639d555d4fa69c18efaa483a7f9ed62a7",
                        ["video_preprocessor_config.json"] = "f6828441758b19187559988fabac496c478fd3a6b684b1dd674f9f01fd5fcba4",
                        ["chat_template.json"] = "d272c68bd115dca51b7e918dabac29b23510e3a25c4c5d9bd410b96eb11bf26e",
                        ["generation_config.json"] = "b83f040874812bdf0aafa9d566049271036f1ab73d860cef53a0989a109627a5",
                    },
                },
                Notes = "Needs about 21 GB of device memory at 16 bits (CUDA keeps BF16; Vulkan casts once to F16), or about 42 GB of "
                    + "host RAM on the CPU, where projections are widened to F32. Folder download, own folder: it cannot share one "
                    + "with the GGUF, since a folder holding both a GGUF and a shard index is ambiguous. Apache-2.0.",
            },
            new CatalogVariant
            {
                Id = "nf4", DisplayName = "Official NF4 (bitsandbytes 4-bit)", Flavor = null,
                Components = CatalogComponents.Backbone | CatalogComponents.Vision,
                Source = new CatalogShardSet
                {
                    Repo = Nf4Repo, Revision = Nf4Revision, TargetSubdir = Folder + "-nf4",
                    IndexFile = "model.safetensors.index.json", FileCount = 3, TotalBytes = 11_841_998_136,
                    SupportFiles = SupportFiles,
                    Sha256ByFile = new Dictionary<string, string>
                    {
                        ["model.safetensors.index.json"] = "1d50cc23c4f8f2d2d64f908b1f4d5e0960355130982c43614c450ca487f12960",
                        ["model-00001-of-00003.safetensors"] = "b0219c7d3d132e72d80cc6f0d04aa20e98edce88dea6d282ace84539c81e8c12",
                        ["model-00002-of-00003.safetensors"] = "12114b03fc2c0735059a5617610848d480b5bfb04fe13569fa7e24776b8b4ee1",
                        ["model-00003-of-00003.safetensors"] = "db7736eefc60465956ece6b3748a5175b60b764375d50d8c13bf67f947a18704",
                        ["tokenizer.json"] = "e7bbd0f9784004df51aca562befc3c7a8f294b4045aa8685536c35804c9aa493",
                        ["config.json"] = "933f503aca4aba8424b99d902fae8557381950816cec082bfead1b88d731bdad",
                        ["preprocessor_config.json"] = "b4553631864a28c1a478e81171e8d311c27dee24ea0930e9bf0c68cbe4953201",
                        ["video_preprocessor_config.json"] = "bbe22d3f192cfbc2cec69ddec52b4b70683c1240eda333df6974a08b92e05232",
                        ["chat_template.json"] = "d272c68bd115dca51b7e918dabac29b23510e3a25c4c5d9bd410b96eb11bf26e",
                        ["generation_config.json"] = "45c0897c93421c07d90db183e0f44745ebb5b73a222c15b3962751d26c50143c",
                    },
                },
                Notes = "The 4-bit projections load as Q4_K (Q4_0 through the stack gave garbage; see docs/MOSS_VL_ARCHITECTURE.md) after a "
                    + "one-time conversion into a quant cache. On a 12 GB card pair it with WeightPrecision q8_0 for the layers the "
                    + "checkpoint leaves in BF16 (about 28.8 tok/s on Vulkan).",
            },
            new CatalogVariant
            {
                Id = "fp8", DisplayName = "Official FP8 (per-channel)", Flavor = null,
                Components = CatalogComponents.Backbone | CatalogComponents.Vision,
                Source = new CatalogShardSet
                {
                    Repo = Fp8Repo, Revision = Fp8Revision, TargetSubdir = Folder + "-fp8",
                    IndexFile = "model.safetensors.index.json", FileCount = 4, TotalBytes = 15_729_921_752,
                    SupportFiles = SupportFiles,
                    Sha256ByFile = new Dictionary<string, string>
                    {
                        ["model.safetensors.index.json"] = "2a8a2eec9467083ef3cdf18b240411be5535d48754c9538b79976b8108026338",
                        ["model-00001-of-00004.safetensors"] = "1e8bd10851f55f846e8943cf9e1ab0651ddd96a3a3d5e74dd06a19c876d384ba",
                        ["model-00002-of-00004.safetensors"] = "17467922bd278d495c0716005f61f7d6f1d7e952603a78fa76ab7be9eca04421",
                        ["model-00003-of-00004.safetensors"] = "d89d911bc0fe79b2f22fddd94f39606c6359cad26fd32a227de11aede6672ad4",
                        ["model-00004-of-00004.safetensors"] = "717d11d69c19883857e6a9d20fb442eb3a0ef7dda52122dab9f0b98d8deed0ca",
                        ["tokenizer.json"] = "e7bbd0f9784004df51aca562befc3c7a8f294b4045aa8685536c35804c9aa493",
                        ["config.json"] = "799222d847abe928e7e94296f6badfe6557c2f5111b26222036c22a590765548",
                        ["preprocessor_config.json"] = "b4553631864a28c1a478e81171e8d311c27dee24ea0930e9bf0c68cbe4953201",
                        ["video_preprocessor_config.json"] = "bbe22d3f192cfbc2cec69ddec52b4b70683c1240eda333df6974a08b92e05232",
                        ["chat_template.json"] = "d272c68bd115dca51b7e918dabac29b23510e3a25c4c5d9bd410b96eb11bf26e",
                        ["generation_config.json"] = "45c0897c93421c07d90db183e0f44745ebb5b73a222c15b3962751d26c50143c",
                    },
                },
                Notes = "Per-channel FP8 (compressed-tensors) projections load as Q8_0 after a one-time conversion into a quant cache. "
                    + "On a 12 GB card pair it with WeightPrecision q4_k (about 9.7 tok/s on Vulkan).",
            },
        ],
    };

    /// <summary>The JSON a folder load needs beside the weights: configuration, tokenizer, processors and the chat template.</summary>
    private static readonly string[] SupportFiles =
    [
        "config.json", "tokenizer.json", "preprocessor_config.json", "video_preprocessor_config.json", "chat_template.json",
        "generation_config.json",
    ];
}
