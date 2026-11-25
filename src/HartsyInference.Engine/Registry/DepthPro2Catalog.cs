namespace HartsyInference.Engine.Registry;

/// <summary>
/// The <c>depthpro2-s|b|l|h</c> rows: Apple's Depth Pro 2 metric depth (<c>apple-aiml-research/ml-depth-pro-2</c> at aa389cbb), one ungated F32 safetensors
/// repo per size, each row also carrying the focal estimator (<c>apple/DepthPro2-focal</c>) it needs when the caller gives no focal length. The sizes are the
/// variants of <see cref="Vision.DepthPro2Variants"/>; a row downloads only its own size and the estimator. The estimator is a required asset on purpose
/// (<c>MissingAssets</c> and <c>hartsy pull</c> fetch it with the row): a request without a focal length, the default, needs it. The sha256 pins verify the bytes, since an asset
/// cannot name a revision (S at 903a1c1ade918831d138adcc4aedbcc8f4da0078).
/// <para>Licence: the weights ship under the "Apple Machine Learning Research Model License Agreement", use "exclusively for Research Purposes" (non-commercial
/// research, excluding "any commercial exploitation, product development or use in any commercial product or service"); it is personal, non-transferable and
/// revocable, and a redistributed copy needs the licence and an attribution notice. The repos' ACKNOWLEDGEMENTS also list the DINOv3 License; treat both as
/// applying. The code is under Apple's permissive sample-code licence. Recorded, not enforced: a licence never blocks a model here.</para>
/// </summary>
internal static class DepthPro2Catalog
{
    /// <summary>The role of the focal estimator asset in each row.</summary>
    internal const string FocalRole = "focal-estimator";

    /// <summary>The focal estimator every size shares.</summary>
    internal static ModelAsset Focal() => new()
    {
        Repo = "apple/DepthPro2-focal", RepoPath = "model.safetensors", TargetSubdir = "Vision/DepthPro2-focal", Role = FocalRole,
        Sha256 = "8f567d1c97ff916137288668f6bd306ce2094da2e2b852b0e4539dbb139c8c86",
    };

    // S is Verified: it matches the reference on CPU F32, and CUDA and Vulkan match the CPU lane. B, L and H+ have not run at all.

    /// <summary>The S row.</summary>
    internal static CatalogEntry Small() =>
        Row("depthpro2-s", "S", "S", "ViT-S/16", "583c4d630779fd231f969c738733624963ad34bd3c29f9c64088257e3436f97c", ModelStatus.Verified);

    /// <summary>The B row.</summary>
    internal static CatalogEntry Base() => Row("depthpro2-b", "B", "B", "ViT-B/16", "244ff85e0d9bde6e0f985de1c61420abc4c2110265b495c974556225ac2b60ed");

    /// <summary>The L row.</summary>
    internal static CatalogEntry Large() => Row("depthpro2-l", "L", "L", "ViT-L/16", "d799d8a91f2e8dc91715f9b33a405a1ff0727212041d206ebbb04e6c1424ff9f");

    /// <summary>The H+ row (repo <c>apple/DepthPro2-H</c>).</summary>
    internal static CatalogEntry HugePlus() => Row("depthpro2-h", "H", "H+", "ViT-H+/16", "cfc532336019abd2128b04be2c2967175522e46de69222ef72b87c6675ee3b17");

    private static CatalogEntry Row(string id, string repoSize, string size, string backbone, string sha256,
        ModelStatus status = ModelStatus.ValidationPending) => new()
    {
        Id = id, Modality = Modality.Vision, DisplayName = $"Depth Pro 2 {size} (metric depth, research-only weights)",
        Architecture = $"DINOv3 {backbone} (timm layout) + reassemble/fusion decoder + inverse-depth head; separate ViT-S/16 focal estimator",
        Status = status, CliDrivable = true, HuggingFaceRepo = "apple/DepthPro2-" + repoSize,
        Assets =
        [
            new() { Repo = "apple/DepthPro2-" + repoSize, RepoPath = "model.safetensors", TargetSubdir = "Vision/DepthPro2-" + repoSize, Role = "transformer",
                Sha256 = sha256 },
            Focal(),
        ],
    };
}
