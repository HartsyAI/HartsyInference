using SharpInference.Core.Tensors;

namespace SharpInference.Core.MemoryManagement;

/// <summary>
/// A "block" of weights that can be streamed to/from device memory as a unit.
/// Implemented by transformer layers, UNet down/up blocks, VAE encoder stages,
/// etc. — any sequential model whose weights cleanly partition into independent
/// subgroups large enough to make per-block H2D bandwidth amortise the per-call
/// CUDA overhead.
///
/// <para>Models hand a list of these to <see cref="BlockStreamingController"/>
/// which then orchestrates prefetch / eviction during the forward pass.</para>
///
/// <para><b>Implementer responsibilities:</b></para>
/// <list type="bullet">
/// <item><description><see cref="EnumerateWeights"/> must return the same tensor
/// references on every call (the controller uses reference equality to track which
/// tensors are uploaded/evicted).</description></item>
/// <item><description><see cref="EstimatedWeightBytes"/> should be computed once
/// at construction or first call and cached — the controller may query it many times.</description></item>
/// <item><description>Blocks are independent: block N's forward must not assume
/// block M's weights are resident. The controller manages residency.</description></item>
/// </list>
/// </summary>
public interface IStreamingBlock
{
    /// <summary>Sum of <c>tensor.ElementCount * dtype.SizeInBytes</c> across all
    /// weights in this block. Used by the controller's budget heuristic to size
    /// its prefetch window and decide when to fall back to synchronous loads.</summary>
    long EstimatedWeightBytes { get; }

    /// <summary>Returns the weight tensors that must be resident on device while
    /// this block's forward pass runs. Order is irrelevant; same references
    /// must come back on every call.</summary>
    IEnumerable<Tensor> EnumerateWeights();
}
