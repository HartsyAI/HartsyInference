using HartsyInference.ModelAssets.Quant;

namespace HartsyInference.Engine.Registry;

/// <summary>The <c>deepseek-v4.1-flash</c> catalog row: one model, six derivatives, each pinned to the commit its files were inspected at.</summary>
internal static class DeepSeekV41Catalog
{
    /// <summary>Catalog id of the model.</summary>
    internal const string Id = "deepseek-v4.1-flash";

    private const CatalogComponents Everything =
        CatalogComponents.Backbone | CatalogComponents.Draft | CatalogComponents.Vision | CatalogComponents.Engram;

    /// <summary>The refusal the MLX variant carries; a test pins it to what the checkpoint scan reports for the real repo's gaps.</summary>
    internal const string MlxDraftRefusal =
        "DSpark speculative decoding refused: mtp.2 lacks 14 of 128 routed experts (9, 87, 88-99).";

    /// <summary>Builds the entry. It has no <see cref="CatalogEntry.Assets"/>, so the CLI never offers a ~475 GiB download for it.</summary>
    internal static CatalogEntry Build() => new()
    {
        // Model class is not wired (PR 7 ships the config, loader and catalog rows only), hence Structural.
        Id = Id, Modality = Modality.Text, DisplayName = "DeepSeek-V4.1-Flash",
        Architecture = "sparse latent attention + mHC + Engram + 384-expert MoE with 3 draft layers",
        Status = ModelStatus.Structural, CliDrivable = false, HuggingFaceRepo = "deepseek-ai/DeepSeek-V4.1-Flash",
        Variants =
        [
            new CatalogVariant
            {
                Id = "official", DisplayName = "Official (FP8 + FP4 experts)", Flavor = QuantFlavor.Official, Components = Everything,
                Source = ShardSet(
                    "official", "deepseek-ai/DeepSeek-V4.1-Flash", "dba1be0a40aa45a94ad051997016db3960a90277", 48, 510_296_708_312),
                Notes = "Header inventory sums to 510,286,023,000 tensor bytes over 96,085 tensors, matching metadata.total_size.",
            },
            new CatalogVariant
            {
                Id = "nvfp4", DisplayName = "NVIDIA ModelOpt NVFP4", Flavor = QuantFlavor.NvidiaNvfp4, Components = Everything,
                Source = ShardSet(
                    "nvfp4", "nvidia/DeepSeek-V4.1-Flash-NVFP4", "3431dde3247c13b5957f682b1e3c6fcae2566079", 48, 527_293_384_576),
                Notes = "The index copies the official metadata.total_size (510,286,023,000), which its 527 GB of shards exceed.",
            },
            new CatalogVariant
            {
                Id = "quark-mxfp4", DisplayName = "AMD Quark MXFP4", Flavor = QuantFlavor.AmdQuark, Components = Everything,
                Source = ShardSet(
                    "quark-mxfp4", "amd/DeepSeek-V4.1-Flash-Quark-MXFP4", "0d56db2133ce922656be8025eac1e8450243aed5", 48, 509_582_097_952),
                Notes = "Same 96,085 tensor names as official; scales are <weight>_scale.",
            },
            new CatalogVariant
            {
                Id = "exl3-2.0bpw", DisplayName = "EXL3 2.0 bpw (mcg, Viterbi, MXFP8 lm_head)", Flavor = QuantFlavor.Exl3,
                Components = Everything,
                Source = ShardSet(
                    "exl3-2.0bpw", "sfxnz/DeepSeek-V4.1-Flash-EXL3", "982b70452f399814f56b46272fd30394ae10d58c", 48, 357_466_041_064),
                Notes = "Branch 2.0bpw-mcg-viterbi-lmhead-mxfp8. The head is lm_head.weight; the index declares total_size 0.",
            },
            new CatalogVariant
            {
                Id = "mlx-4bit", DisplayName = "MLX 4-bit affine (group size 64)", Flavor = QuantFlavor.Mlx,
                Components = CatalogComponents.Backbone | CatalogComponents.Vision | CatalogComponents.Engram,
                Source = ShardSet(
                    "mlx-4bit", "mlx-community/DeepSeek-V4.1-Flash-MLX-4bit", "100694c1c65d34e331f4f0c2841212f0e849a0ea", 75, 486_051_783_624),
                Refusals = new Dictionary<CatalogComponents, string> { [CatalogComponents.Draft] = MlxDraftRefusal },
                Notes = "75 shards are indexed; four stale of-00048 files sit beside them and are not fetched, and the index total_size "
                    + "(552,845,699,976) still counts them.",
            },
            GgufVariant("dwarfstar-q2", "DwarfStar GGUF Q2", ["DeepSeek-V4.1-Flash-Q2.gguf"], 365_713_686_528 + 970_555_552),
            GgufVariant("dwarfstar-q4", "DwarfStar GGUF Q4 (two raw parts)",
                ["DeepSeek-V4.1-Flash-Q4.gguf.part1", "DeepSeek-V4.1-Flash-Q4.gguf.part2"], 480_000_000_000 + 38_596_067_328 + 970_555_552),
        ],
    };

    // The official derivative lands where ModelResolver looks for the catalog id; the others sit beside it.
    private static string Subdir(string variantId) => variantId == "official" ? $"LLM/{Id}" : $"LLM/{Id}-{variantId}";

    private static CatalogShardSet ShardSet(string variantId, string repo, string revision, int shards, long bytes) => new()
    {
        Repo = repo, Revision = revision, TargetSubdir = Subdir(variantId),
        IndexFile = "model.safetensors.index.json", FileCount = shards, TotalBytes = bytes,
    };

    private static CatalogVariant GgufVariant(string id, string name, string[] parts, long bytes) => new()
    {
        Id = id, DisplayName = name, Flavor = null,
        Components = CatalogComponents.Backbone | CatalogComponents.Vision | CatalogComponents.Engram,
        VisionSeparateFile = true,
        Refusals = new Dictionary<CatalogComponents, string>
        {
            [CatalogComponents.Draft] = "DSpark speculative decoding unavailable: the DwarfStar files carry no draft (mtp) layers.",
        },
        Source = new CatalogShardSet
        {
            Repo = "antirez/deepseek-v4.1-flash-gguf", Revision = "dd8a266f7145edc19e2334b46e19b6821f221dc7",
            TargetSubdir = Subdir(id),
            FileCount = parts.Length + 1, TotalBytes = bytes, Files = [.. parts, "DeepSeek-V4.1-Flash-Vision.gguf"],
        },
        Notes = parts.Length > 1 ? "The Q4 file is split into raw parts; concatenate part1 then part2 to form the GGUF." : null,
    };
}
