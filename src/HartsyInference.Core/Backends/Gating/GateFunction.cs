namespace HartsyInference.Core.Backends.Gating;

/// <summary>What a <see cref="GroupScaleSpec"/> gate passes through before it scales its group.</summary>
public enum GateFunction
{
    /// <summary>The stored gate as it is: a mask or a routing weight.</summary>
    Identity,

    /// <summary><c>2 · sigmoid(g)</c>: a gate around identity (LTX-2's per-head output gate).</summary>
    TwoSigmoid,
}
