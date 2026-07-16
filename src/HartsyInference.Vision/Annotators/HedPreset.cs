namespace HartsyInference.Vision.Annotators;

/// <summary>Checkpoint identity for the ControlNetHED soft-edge annotator (lllyasviel's Apache-2 re-release of
/// the "network-bsds500" HED model that all softedge_hed / scribble_hed ControlNets were trained against).
/// The architecture is fixed (see <see cref="HedModel"/>); this record pins the download for the extension's
/// annotator auto-downloader.</summary>
public sealed record HedPreset
{
    /// <summary>Hub-style identifier.</summary>
    public required string Name { get; init; }

    /// <summary>HuggingFace repo the checkpoint lives in (non-gated).</summary>
    public required string HuggingFaceRepo { get; init; }

    /// <summary>Checkpoint file name inside the repo.</summary>
    public required string CheckpointFile { get; init; }

    /// <summary>SHA-256 of the checkpoint file, for download verification.</summary>
    public required string Sha256 { get; init; }

    /// <summary><c>lllyasviel/Annotators/ControlNetHED.pth</c> — the only released variant (28MB).</summary>
    public static HedPreset Default => new()
    {
        Name = "lllyasviel/Annotators/ControlNetHED",
        HuggingFaceRepo = "lllyasviel/Annotators",
        CheckpointFile = "ControlNetHED.pth",
        Sha256 = "5ca93762ffd68a29fee1af9d495bf6aab80ae86f08905fb35472a083a4c7a8fa",
    };
}
