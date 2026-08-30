namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>Execution settings checked again at the PDD boundary after profile planning.</summary>
public sealed record MiniMaxH3PddExecutionSettings
{
    /// <summary>Number of transformer evaluations.</summary>
    public required int Nfe { get; init; }

    /// <summary>Sampler name; the released adapters require plain Euler.</summary>
    public required string Sampler { get; init; }

    /// <summary>Classifier-free guidance scale.</summary>
    public required float CfgScale { get; init; }

    /// <summary>Video flow-shift value.</summary>
    public required double VideoFlowShift { get; init; }

    /// <summary>Audio flow-shift value.</summary>
    public required double AudioFlowShift { get; init; }

    /// <summary>Adapter strength; partial PDD heads are not a published recipe.</summary>
    public required float Strength { get; init; }

    /// <summary>Whether another baked or adapter-based distillation path is active.</summary>
    public bool HasOtherDistillation { get; init; }

    /// <summary>Whether VSA sparse attention is active.</summary>
    public bool HasVsa { get; init; }

    /// <summary>Whether step/block caching is active.</summary>
    public bool HasStepCache { get; init; }

    /// <summary>Whether the request uses the hybrid FL+reference packing contract.</summary>
    public bool HasHybrid { get; init; }
}
