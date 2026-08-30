namespace HartsyInference.Engine.Planning;

/// <summary>The MiniMax-H3 conditioning contract a checkpoint was trained or distilled to serve.</summary>
public enum VideoTaskFamily
{
    /// <summary>The checkpoint structure does not distinguish a more specific task.</summary>
    Unknown = 0,

    /// <summary>Text-only conditioning with jointly generated video and audio.</summary>
    T2Va = 1,

    /// <summary>First/last-frame conditioning with jointly generated video and audio.</summary>
    Fl2Va = 2,

    /// <summary>Reference media conditioning with jointly generated video and audio.</summary>
    Ref2Va = 3,

    /// <summary>First/last-frame and reference conditioning may be combined in one packed sequence.</summary>
    Hybrid = 4,
}
