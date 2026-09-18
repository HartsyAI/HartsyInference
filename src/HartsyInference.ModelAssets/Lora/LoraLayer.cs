using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Lora;

/// <summary>One LoRA weight delta bound to a canonical target key. The delta's matrices are typically borrowed from the parent LoraFile's safetensors mmap and remain valid only for the file's lifetime.</summary>
public sealed class LoraLayer
{
    /// <summary>Canonical weight-dictionary key (the same string the model class passes to weights[…] in its LoadWeights method, including the trailing .weight).</summary>
    public required string TargetKey { get; init; }

    /// <summary>Which model component this layer applies to.</summary>
    public required LoraTarget Target { get; init; }

    /// <summary>The decomposition that produces this layer's ΔW.</summary>
    public required LoraDelta Delta { get; init; }

    /// <summary>Decomposition variant, derived from <see cref="Delta"/>.</summary>
    public LoraVariant Variant => Delta.Variant;

    /// <summary>Down-projection (A) matrix with shape [rank, in_dim]. Standard and DoRA layers only.</summary>
    public Tensor LoraDown => Standard.Down;

    /// <summary>Up-projection (B) matrix with shape [out_dim, rank]. Standard and DoRA layers only.</summary>
    public Tensor LoraUp => Standard.Up;

    /// <summary>Per-layer alpha scaling factor; equals Rank when the file stores no explicit alpha (treat scale = alpha / rank). Standard and DoRA layers only.</summary>
    public float Alpha => Standard.Alpha;

    /// <summary>Intrinsic rank — equals LoraDown.Shape[0]. Standard and DoRA layers only.</summary>
    public int Rank => Standard.Rank;

    /// <summary>The low-rank pair behind this layer, or a named failure for a decomposition that has none.</summary>
    private StandardLoraDelta Standard => Delta as StandardLoraDelta
        ?? throw new HartsyInferenceException(
            $"LoRA layer '{TargetKey}' is a {Delta.Variant} adapter, which has no down/up pair. Consume its "
            + $"{nameof(LoraDelta.ComputeF32)} delta instead of the low-rank matrices.");
}
