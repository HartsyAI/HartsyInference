namespace HartsyInference.Engine.Requests;

/// <summary>Caller policy for a checkpoint profile with learned video sparse attention.</summary>
public enum SparseAttentionPolicy
{
    /// <summary>Require sparse execution whenever the detected profile carries VSA gates.</summary>
    Auto = 0,
    /// <summary>Explicitly require a supported native sparse backend.</summary>
    Require = 1,
    /// <summary>Disable sparse execution; rejected unless the profile explicitly certifies dense equivalence.</summary>
    Disable = 2,
}
