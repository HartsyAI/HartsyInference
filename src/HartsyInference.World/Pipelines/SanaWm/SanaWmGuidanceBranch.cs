namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>Which caption a batch item of a SANA-WM forward is conditioned on.</summary>
public enum SanaWmGuidanceBranch
{
    /// <summary>The negative prompt, without the CHI prefix.</summary>
    Unconditional,

    /// <summary>The CHI-prefixed prompt.</summary>
    Conditional,
}
