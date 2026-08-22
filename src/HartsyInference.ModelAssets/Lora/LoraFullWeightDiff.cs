using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Lora;

/// <summary>One full-weight delta from a Comfy-style LoRA (<c>.diff</c> = weight, <c>.diff_b</c> = bias). Not low-rank: the tensor IS the delta, applied as W' = W + strength·diff. Like LoraLayer's matrices, the tensor is borrowed from the parent LoraFile's mmap and remains valid only for the file's lifetime.</summary>
public sealed class LoraFullWeightDiff
{
    /// <summary>Canonical weight-dictionary key, including the trailing .weight or .bias.</summary>
    public required string TargetKey { get; init; }

    /// <summary>Which model component this diff applies to.</summary>
    public required LoraTarget Target { get; init; }

    /// <summary>The delta tensor, same shape as the target weight.</summary>
    public required Tensor Diff { get; init; }

    /// <summary>True for a <c>.diff_b</c> bias delta, false for a <c>.diff</c> weight delta.</summary>
    public required bool IsBias { get; init; }
}
