namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>H3 trunk family an acceleration adapter was trained against.</summary>
public enum MiniMaxH3PddTask
{
    /// <summary>The file does not carry a trustworthy task binding and must be confirmed by a hash-bound profile.</summary>
    Unknown,

    /// <summary>First/last-frame and text-to-video/audio H3 trunk.</summary>
    Fl2Va,

    /// <summary>Reference-conditioned video/audio H3 trunk.</summary>
    Ref2Va,
}
