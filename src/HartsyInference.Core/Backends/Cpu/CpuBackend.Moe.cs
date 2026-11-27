using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Backends.Routing;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Cpu;

public sealed partial class CpuBackend
{
    /// <inheritdoc />
    public void Softplus(Tensor output, Tensor input)
    {
        ThrowIfDisposed();
        SoftplusReference.Apply(output, input);
    }

    /// <inheritdoc />
    public void MoeRoute(Tensor topkIdx, Tensor topkWeight, Tensor logits, in MoeRouteArgs args,
        Tensor? bias = null, Tensor? altBias = null, Tensor? tokenKinds = null)
    {
        ThrowIfDisposed();
        MoeReference.Route(topkIdx, topkWeight, logits, args, bias, altBias, tokenKinds);
    }

    /// <inheritdoc />
    public void MoeBuildDispatch(Tensor counts, Tensor offsets, Tensor permutedToken, Tensor pairSlot, Tensor topkIdx,
        int numExperts)
    {
        ThrowIfDisposed();
        MoeReference.BuildDispatch(counts, offsets, permutedToken, pairSlot, topkIdx, numExperts);
    }

    /// <inheritdoc />
    public void MoeCombine(Tensor output, Tensor expertOut, Tensor pairSlot, Tensor topkWeight, int k, bool accumulate)
    {
        ThrowIfDisposed();
        MoeReference.Combine(output, expertOut, pairSlot, topkWeight, k, accumulate);
    }

}
