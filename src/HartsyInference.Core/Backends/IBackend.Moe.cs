using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Mixture-of-experts routing and top-k primitives; backends without a kernel inherit the throwing defaults.
public partial interface IBackend
{
    /// <summary>Softplus, <c>ln(1 + e^x)</c>, elementwise over F32; <paramref name="output"/> may alias <paramref name="input"/>.</summary>
    void Softplus(Tensor output, Tensor input) =>
        throw NotSupportedPrimitive(nameof(Softplus));

    /// <summary>Per token: score the router logits, pick the top-k experts with an optional bias and group limit, and
    /// write the expert ids and combine weights. Layout: <c>logits [T,E]</c>, <c>topkIdx</c>/<c>topkWeight</c> <c>[T,k]</c>.</summary>
    /// <param name="bias">Selection-only bias added to the scores (the weights stay unbiased); null for none.</param>
    /// <param name="altBias">Bias used instead of <paramref name="bias"/> for tokens whose kind is non-zero (image tokens).</param>
    /// <param name="tokenKinds">I32 kind per token, required with <paramref name="altBias"/>.</param>
    void MoeRoute(Tensor topkIdx, Tensor topkWeight, Tensor logits, in MoeRouteArgs args,
        Tensor? bias = null, Tensor? altBias = null, Tensor? tokenKinds = null) =>
        throw NotSupportedPrimitive(nameof(MoeRoute));

    /// <summary>Histogram, exclusive scan and stable expert-major permutation of the token-to-expert pairs.</summary>
    /// <param name="counts">I32 [E] rows per expert.</param>
    /// <param name="offsets">I32 [E+1] exclusive scan of <paramref name="counts"/>.</param>
    /// <param name="permutedToken">I32 [T*k] token id of each expert-major row; -1 past <c>offsets[E]</c>.</param>
    /// <param name="pairSlot">I32 [T*k] expert-major row of each (token, slot) pair; -1 when its expert is out of range.</param>
    void MoeBuildDispatch(Tensor counts, Tensor offsets, Tensor permutedToken, Tensor pairSlot, Tensor topkIdx,
        int numExperts) =>
        throw NotSupportedPrimitive(nameof(MoeBuildDispatch));

    /// <summary>Weighted, deterministic sum of each token's k expert rows: <c>output[t] (+)= sum_j w[t,j] * expertOut[pairSlot[t,j]]</c>.</summary>
    void MoeCombine(Tensor output, Tensor expertOut, Tensor pairSlot, Tensor topkWeight, int k, bool accumulate) =>
        throw NotSupportedPrimitive(nameof(MoeCombine));

    /// <summary>Top-k over the last dimension; see <see cref="TopKReference.Apply"/> for the exact ordering and padding rules.</summary>
    /// <param name="validLengths">Optional I32 per-row count of leading entries that may be chosen.</param>
    /// <param name="sortByIndex">Order the result by ascending index instead of descending value.</param>
    void TopKLastDim(Tensor values, Tensor indices, Tensor input, int k, Tensor? validLengths = null,
        bool sortByIndex = false) =>
        throw NotSupportedPrimitive(nameof(TopKLastDim));

    private NotSupportedException NotSupportedPrimitive(string name) =>
        new NotSupportedException($"{name} is not implemented on the {Device} backend.");
}
