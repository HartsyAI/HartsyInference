namespace HartsyInference.ThreeD.Pipelines;

/// <summary>Upstream's pipeline types that this port runs. The number is the final dual-grid resolution; the cascades first sample
/// the 512 shape flow, then refine at that resolution with the 1024 flow. The direct <c>1024</c> type is not exposed.</summary>
public enum Trellis2PipelineType
{
    /// <summary>Upstream <c>512</c>: the 512 flow on the 32 grid, decoded at 512.</summary>
    Res512 = 512,

    /// <summary>Upstream <c>1024_cascade</c>, the default.</summary>
    Cascade1024 = 1024,

    /// <summary>Upstream <c>1536_cascade</c>: the 1024 flow on up to a 96 grid, decoded at 1536 (or lower when the token cap bites).</summary>
    Cascade1536 = 1536,
}
