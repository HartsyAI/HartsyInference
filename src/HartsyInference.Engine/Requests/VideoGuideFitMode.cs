namespace HartsyInference.Engine.Requests;

/// <summary>How visual guide media is fitted to the aligned MiniMax-H3 target geometry.</summary>
public enum VideoGuideFitMode
{
    /// <summary>Preserve legacy behavior: resize and center-crop to cover the target.</summary>
    Cover = 0,

    /// <summary>Resize inside the target and pad the uncovered area.</summary>
    Contain = 1,

    /// <summary>Resize directly to the target dimensions without preserving aspect ratio.</summary>
    Stretch = 2,
}
