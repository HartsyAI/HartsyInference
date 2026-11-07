namespace HartsyInference.Core.Backends.Gating;

/// <summary>Which half of a gated linear unit's packed last axis holds the gate.</summary>
public enum GluGateHalf
{
    /// <summary><c>[gate | value]</c>: the gate leads (the <c>[gate | up]</c> FFN projections).</summary>
    First,

    /// <summary><c>[value | gate]</c>: the gate trails (the GEGLU and SiTU projections).</summary>
    Second,
}
