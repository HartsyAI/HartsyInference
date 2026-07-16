namespace HartsyInference.Vision.Annotators;

/// <summary>Checkpoint identity for the anime-style lineart Generator variants comfyui_controlnet_aux's
/// <c>LineartDetector</c> ships (identical architecture — <see cref="LineartGenerator"/> — different
/// training): <c>sk_model.pth</c> (realistic/detailed) and <c>sk_model2.pth</c> (coarse).</summary>
public sealed record LineartPreset
{
    /// <summary>Hub-style identifier.</summary>
    public required string Name { get; init; }

    /// <summary>HuggingFace repo the checkpoint lives in (non-gated).</summary>
    public required string HuggingFaceRepo { get; init; }

    /// <summary>Checkpoint file name inside the repo.</summary>
    public required string CheckpointFile { get; init; }

    /// <summary>SHA-256 of the checkpoint file, for download verification.</summary>
    public required string Sha256 { get; init; }

    /// <summary>Residual block count (3 for both released variants).</summary>
    public int ResidualBlocks { get; init; } = 3;

    /// <summary><c>lllyasviel/Annotators/sk_model.pth</c> — the default detailed lineart model (17MB).</summary>
    public static LineartPreset Realistic => new()
    {
        Name = "lllyasviel/Annotators/sk_model",
        HuggingFaceRepo = "lllyasviel/Annotators",
        CheckpointFile = "sk_model.pth",
        Sha256 = "c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6",
    };

    /// <summary><c>lllyasviel/Annotators/sk_model2.pth</c> — the coarse variant (17MB).</summary>
    public static LineartPreset Coarse => new()
    {
        Name = "lllyasviel/Annotators/sk_model2",
        HuggingFaceRepo = "lllyasviel/Annotators",
        CheckpointFile = "sk_model2.pth",
        Sha256 = "30a534781061f34e83bb9406b4335da4ff2616c95d22a585c1245aa8363e74e0",
    };
}
