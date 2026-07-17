namespace HartsyInference.Vision.Annotators;

/// <summary>Checkpoint identity for the UperNet-ConvNeXt semantic-segmentation annotator (the transformers
/// reference the diffusers ecosystem documents for <c>control_v11p_sd15_seg</c> conditioning). The
/// architecture is fixed (see <see cref="UperNetSegModel"/>); this record pins the download for the
/// extension's annotator auto-downloader.</summary>
public sealed record UperNetSegPreset
{
    /// <summary>Hub-style identifier.</summary>
    public required string Name { get; init; }

    /// <summary>HuggingFace repo the checkpoint lives in (non-gated).</summary>
    public required string HuggingFaceRepo { get; init; }

    /// <summary>Checkpoint file name inside the repo.</summary>
    public required string CheckpointFile { get; init; }

    /// <summary>Local file name to store the checkpoint under (the repo file name is a generic
    /// <c>pytorch_model.bin</c>).</summary>
    public required string LocalFileName { get; init; }

    /// <summary>SHA-256 of the checkpoint file, for download verification.</summary>
    public required string Sha256 { get; init; }

    /// <summary><c>openmmlab/upernet-convnext-small</c> ADE20K (327MB F32 pickle) — the variant the
    /// diffusers seg-ControlNet examples use.</summary>
    public static UperNetSegPreset ConvNextSmall => new()
    {
        Name = "openmmlab/upernet-convnext-small",
        HuggingFaceRepo = "openmmlab/upernet-convnext-small",
        CheckpointFile = "pytorch_model.bin",
        LocalFileName = "upernet_convnext_small.bin",
        Sha256 = "76c163aa531ab7edfb3a77bbcc039e340645aa0ffe2b0ffcfc68755f550c76ea",
    };
}
