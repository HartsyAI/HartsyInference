using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.MemoryManagement;

/// <summary>A "block" of weights streamed to/from device memory as a unit (transformer layer, UNet down/up block, VAE encoder stage).</summary>
public interface IStreamingBlock
{
    /// <summary>Sum of <c>tensor.ElementCount * dtype.SizeInBytes</c> across this block's weights; sizes the controller's prefetch window.</summary>
    long EstimatedWeightBytes { get; }

    /// <summary>Weight tensors that must be resident on device while this block's forward pass runs.</summary>
    /// <remarks>Order is irrelevant, but the same tensor references must come back on every call — the streaming
    /// controller uses reference equality to track residency.</remarks>
    IEnumerable<Tensor> EnumerateWeights();
}
