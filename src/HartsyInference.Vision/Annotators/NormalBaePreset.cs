namespace HartsyInference.Vision.Annotators;

/// <summary>Checkpoint identity for the NormalBAE surface-normal annotator ("Estimating and Exploiting the
/// Aleatoric Uncertainty in Surface Normal Estimation", the ScanNet-trained NNET all normalbae ControlNets
/// condition on). Architecture is fixed (see <see cref="NormalBaeModel"/>); this record pins the download
/// for the extension's annotator auto-downloader.</summary>
public sealed record NormalBaePreset
{
    /// <summary>Hub-style identifier.</summary>
    public required string Name { get; init; }

    /// <summary>HuggingFace repo the checkpoint lives in (non-gated).</summary>
    public required string HuggingFaceRepo { get; init; }

    /// <summary>Checkpoint file name inside the repo.</summary>
    public required string CheckpointFile { get; init; }

    /// <summary>SHA-256 of the checkpoint file, for download verification.</summary>
    public required string Sha256 { get; init; }

    /// <summary><c>lllyasviel/Annotators/scannet.pt</c> — the only released variant (291MB).</summary>
    public static NormalBaePreset Default => new()
    {
        Name = "lllyasviel/Annotators/scannet",
        HuggingFaceRepo = "lllyasviel/Annotators",
        CheckpointFile = "scannet.pt",
        Sha256 = "03dbf1600c51ee3d45c29f77b77bf1a3b7a24c3452dba62a4ae658f37330c209",
    };
}
