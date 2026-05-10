using SharpInference.Core.Tensors;

namespace SharpInference.Core.MemoryManagement;

/// <summary>A "block" of weights streamed to/from device memory as a unit (transformer layer, UNet down/up block, VAE encoder stage). Implementers must return the same tensor references on every <see cref="EnumerateWeights"/> call (the controller uses reference equality for residency tracking).</summary>
public interface IStreamingBlock
{
    /// <summary>Sum of <c>tensor.ElementCount * dtype.SizeInBytes</c> across all weights in this block. Used by the controller's budget heuristic to size its prefetch window.</summary>
    long EstimatedWeightBytes { get; }

    /// <summary>Returns the weight tensors that must be resident on device while this block's forward pass runs. Order is irrelevant; the same references must come back on every call.</summary>
    IEnumerable<Tensor> EnumerateWeights();
}
