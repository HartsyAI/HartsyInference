using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.MemoryManagement;

/// <summary>A "block" of weights streamed to/from device memory as a unit (transformer layer, UNet down/up block, VAE encoder stage).</summary>
public interface IStreamingBlock
{
    /// <summary>Sum of <c>tensor.ElementCount * dtype.SizeInBytes</c> across this block's weights; sizes the controller's prefetch window.</summary>
    long EstimatedWeightBytes { get; }

    /// <summary>Weight tensors that must be resident on device while this block's forward pass runs.</summary>
    /// <remarks>Order is irrelevant, but the same tensor references must come back on every call — the streaming
    /// controller uses reference equality to track residency.
    /// <para><b>This is what makes the LoRA ordering rule load-bearing for streaming.</b> A LoRA merge REPLACES
    /// entries in the weight dictionary, so it has to run before the denoiser captures them
    /// (<c>BuildAndApply</c> then <c>LoadWeights</c>, the rule every recipe already states). Merging afterwards
    /// would leave the blocks enumerating pre-merge tensors: the controller would stream the unmodified weights
    /// while the layer computed against something else, and the LoRA would silently not apply — visible only as
    /// output that ignores it, never as an error.</para></remarks>
    IEnumerable<Tensor> EnumerateWeights();
}
