namespace HartsyInference.Engine.Requests;

/// <summary>One independently windowed MiniMax-H3 Fun ControlNet-Union stream.</summary>
public sealed record VideoControl
{
    /// <summary>Local control-branch checkpoint path.</summary>
    public required string Model { get; init; }

    /// <summary>Already-preprocessed control video.</summary>
    public required VideoClip Video { get; init; }

    /// <summary>How the host produced <see cref="Video"/>; used for validation and execution metadata only.</summary>
    public VideoControlKind Kind { get; init; } = VideoControlKind.Custom;

    /// <summary>Residual strength.</summary>
    public double Strength { get; init; } = 1.0;

    /// <summary>Inclusive normalized denoise start.</summary>
    public double Start { get; init; }

    /// <summary>Inclusive normalized denoise end.</summary>
    public double End { get; init; } = 1.0;

    /// <summary>Continuous white-is-visible mask for <see cref="VideoControlKind.Inpaint"/>.</summary>
    public VideoClip? VisibilityMask { get; init; }

    /// <summary>Masked source video for <see cref="VideoControlKind.Inpaint"/>.</summary>
    public VideoClip? MaskedSource { get; init; }
}
