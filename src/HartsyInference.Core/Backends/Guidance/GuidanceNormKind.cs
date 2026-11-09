namespace HartsyInference.Core.Backends.Guidance;

/// <summary>Over which elements a guided prediction's norm is matched to the conditional prediction's.</summary>
public enum GuidanceNormKind
{
    /// <summary>No rescale: the guided prediction is used as formed.</summary>
    None,

    /// <summary>One ratio over the whole tensor (Lance's renormalized guidance).</summary>
    Global,

    /// <summary>One ratio per row of the last dimension (Lumina 2's normalized guidance).</summary>
    PerRow,
}
