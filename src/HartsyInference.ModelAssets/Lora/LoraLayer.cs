using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Lora;

/// <summary>One LoRA-decomposed weight delta. The down/up tensors are typically borrowed from the parent LoraFile's safetensors mmap and remain valid only for the file's lifetime.</summary>
public sealed class LoraLayer
{
    /// <summary>Canonical weight-dictionary key (the same string the model class passes to weights[…] in its LoadWeights method, including the trailing .weight).</summary>
    public required string TargetKey { get; init; }

    /// <summary>Which model component this layer applies to.</summary>
    public required LoraTarget Target { get; init; }

    /// <summary>Down-projection (A) matrix with shape [rank, in_dim].</summary>
    public required Tensor LoraDown { get; init; }

    /// <summary>Up-projection (B) matrix with shape [out_dim, rank].</summary>
    public required Tensor LoraUp { get; init; }

    /// <summary>Per-layer alpha scaling factor; defaults to Rank when the file does not store an explicit alpha (treat scale = alpha / rank).</summary>
    public required float Alpha { get; init; }

    /// <summary>Intrinsic rank — equals LoraDown.Shape[0].</summary>
    public required int Rank { get; init; }

    /// <summary>Decomposition variant. v1 only emits StandardLora.</summary>
    public required LoraVariant Variant { get; init; }
}
