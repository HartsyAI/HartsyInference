namespace HartsyInference.Engine.Planning;

/// <summary>Whether a planning diagnostic is informational, cautionary, or prevents execution.</summary>
public enum VideoPlanIssueSeverity
{
    /// <summary>Non-actionable provenance or resolution information.</summary>
    Info = 0,

    /// <summary>The request remains executable but deserves operator attention.</summary>
    Warning = 1,

    /// <summary>The request must not construct a model or begin denoising.</summary>
    Error = 2,
}
