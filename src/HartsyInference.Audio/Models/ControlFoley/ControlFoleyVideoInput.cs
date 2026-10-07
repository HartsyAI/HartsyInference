namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Preprocessed video streams (official <c>MediaClipData</c>): float frames <c>[N, 3, S, S]</c> per stream.</summary>
public sealed record ControlFoleyVideoInput
{
    /// <summary>Usable duration in seconds after truncating to the shortest stream.</summary>
    public required double TotalDuration { get; init; }

    /// <summary>CLIP frames, RGB in 0..1 (the CLIP tower normalises itself).</summary>
    public required float[] Clip { get; init; }

    /// <summary>Number of CLIP frames.</summary>
    public required int ClipFrames { get; init; }

    /// <summary>CAV-MAE frames, ImageNet-normalised.</summary>
    public required float[] Visual { get; init; }

    /// <summary>Number of CAV-MAE frames.</summary>
    public required int VisualFrames { get; init; }

    /// <summary>Synchformer frames, normalised with mean/std 0.5.</summary>
    public required float[] Sync { get; init; }

    /// <summary>Number of Synchformer frames.</summary>
    public required int SyncFrames { get; init; }
}
