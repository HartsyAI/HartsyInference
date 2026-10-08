using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Moe;

/// <summary>How an expert's gate, up and down projections are stored.</summary>
public enum ExpertWeightLayout
{
    /// <summary>Separate gate, up and down tensors.</summary>
    SplitGateUp,

    /// <summary>One fused <c>[2·I, H]</c> gate/up tensor plus down.</summary>
    FusedGateUp,
}

/// <summary>Shape and storage of one expert. Heterogeneous expert sizes are expressed by different descriptors, not by a uniform assumption.</summary>
public sealed record ExpertDescriptor
{
    /// <summary>Creates an expert shape.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A dimension is not positive.</exception>
    public ExpertDescriptor(int hiddenSize, int intermediateSize, DType weightDType, ExpertWeightLayout layout = ExpertWeightLayout.SplitGateUp)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(intermediateSize);
        HiddenSize = hiddenSize;
        IntermediateSize = intermediateSize;
        WeightDType = weightDType;
        Layout = layout;
    }

    /// <summary>Model width H the expert reads and writes.</summary>
    public int HiddenSize { get; }

    /// <summary>Inner width I of the expert FFN.</summary>
    public int IntermediateSize { get; }

    /// <summary>Storage type of the three projection matrices (block-quantized types allowed).</summary>
    public DType WeightDType { get; }

    /// <summary>Tensor layout.</summary>
    public ExpertWeightLayout Layout { get; }

    /// <summary>Payload bytes of the three projections, excluding any scale companions the checkpoint stores separately.</summary>
    public long PayloadBytes => WeightDType.ComputeByteCount(3L * HiddenSize * IntermediateSize);
}
