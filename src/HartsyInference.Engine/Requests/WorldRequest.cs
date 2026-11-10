using HartsyInference.Core.Memory.Vram;

namespace HartsyInference.Engine.Requests;

/// <summary>Native interactive-world request: opens a stateful session seeded by a prompt and/or an initial frame.</summary>
/// <remarks>Every optional field is checked against the model's <see cref="Worlds.WorldCapabilities"/> by
/// <see cref="Worlds.WorldPlanResolver"/> before a weight is loaded. The CLI and the HTTP route both send this shape.</remarks>
public sealed record WorldRequest
{
    /// <summary>The seeding prompt.</summary>
    public string Prompt { get; init; } = "";

    /// <summary>Optional initial frame the world starts from.</summary>
    public ImageData? InitImage { get; init; }

    /// <summary>Number of denoising steps per generated frame; 0 uses the model default.</summary>
    public int Steps { get; init; }

    /// <summary>RNG seed; negative means a random seed.</summary>
    public long Seed { get; init; } = -1;

    /// <summary>Per-request VRAM lever overrides; null follows the backend's policy.</summary>
    public VramOverrides? Vram { get; init; }

    /// <summary>Frames to roll out, counting the seed frame; null uses the model default.</summary>
    public int? Frames { get; init; }

    /// <summary>Output frame rate; null uses the model default.</summary>
    public int? Fps { get; init; }

    /// <summary>Classifier-free guidance scale; null uses the model default.</summary>
    public double? Guidance { get; init; }

    /// <summary>Flow-matching sigma shift; null uses the model default.</summary>
    public double? FlowShift { get; init; }

    /// <summary>Sampler name; null uses the model's own sampler.</summary>
    public string? Sampler { get; init; }

    /// <summary>Precision the model runs at; null uses the model's first declared precision.</summary>
    public string? Precision { get; init; }

    /// <summary>Whether to run the model's refiner stage.</summary>
    public bool Refiner { get; init; }

    /// <summary>Whether to stream partial frames while each one denoises. Only models that declare progressive
    /// preview support accept it; frames are otherwise delivered as each rollout completes.</summary>
    public bool ProgressivePreview { get; init; }

    /// <summary>Runs the conditional and unconditional guidance passes as two calls per step instead of one batch-2 call.
    /// Only models that declare <see cref="Worlds.WorldCapabilities.SupportsSequentialCfg"/> accept it. Off by default.</summary>
    public bool SequentialCfg { get; init; }

    /// <summary>Typed per-frame camera input; null for an action-driven rollout.</summary>
    public WorldCameraPath? Camera { get; init; }
}
