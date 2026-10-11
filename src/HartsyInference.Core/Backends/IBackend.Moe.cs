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

    /// <summary>True when <see cref="MoeExpertGateUp"/> and <see cref="MoeExpertDown"/> run experts stored as <paramref name="expertType"/>,
    /// so a routed-expert stage can stay on the device from the router to the combine.</summary>
    bool SupportsMoeExpertIndexed(DType expertType) => false;

    /// <summary>Gate and up projections of every routed pair with the activation applied: for row <c>r = token * topk + slot</c>,
    /// <c>act[r] = act(gate[e] . x[token]) * (up[e] . x[token])</c> with <c>e = topkIdx[r]</c> read on the device. The experts are
    /// one device allocation per projection, back to back in expert order.</summary>
    /// <param name="act">F32 <c>[1, tokens * topk, N]</c>.</param>
    /// <param name="x">F32 <c>[1, tokens, K]</c>.</param>
    /// <param name="gateExperts">Per-expert <c>[N, K]</c> weights, resident and contiguous; so are <paramref name="upExperts"/>.</param>
    /// <param name="topkIdx">I32 <c>[tokens, topk]</c> expert ids, as <see cref="MoeRoute"/> writes them.</param>
    /// <param name="gelu">tanh-GELU instead of SiLU.</param>
    void MoeExpertGateUp(Tensor act, Tensor x, IReadOnlyList<Tensor> gateExperts, IReadOnlyList<Tensor> upExperts, Tensor topkIdx,
        int topk, bool gelu) =>
        throw NotSupportedPrimitive(nameof(MoeExpertGateUp));

    /// <summary>Down projection of every routed pair: <c>slotOut[r] = down[topkIdx[r]] . act[r]</c>.</summary>
    /// <param name="slotOut">F32 <c>[1, tokens * topk, N]</c>.</param>
    /// <param name="act">F32 <c>[1, tokens * topk, K]</c>, as <see cref="MoeExpertGateUp"/> wrote it.</param>
    void MoeExpertDown(Tensor slotOut, Tensor act, IReadOnlyList<Tensor> downExperts, Tensor topkIdx, int topk) =>
        throw NotSupportedPrimitive(nameof(MoeExpertDown));

    /// <summary>Weighted sum of each token's routed rows with the optional shared-expert row:
    /// <c>output[t] = sigmoid(sharedGateLogit[t]) * shared[t] + sum_j topkWeight[t,j] * slotOut[t*topk+j]</c>, in slot order.</summary>
    /// <param name="shared">F32 <c>[1, tokens, N]</c>, or null.</param>
    /// <param name="sharedGateLogit">F32 <c>[tokens]</c> pre-sigmoid gate of the shared row, or null for an ungated shared row.</param>
    void MoeCombineSlots(Tensor output, Tensor slotOut, Tensor topkWeight, Tensor? shared, Tensor? sharedGateLogit, int topk) =>
        throw NotSupportedPrimitive(nameof(MoeCombineSlots));

    /// <summary>True when the three expert groups are resident on the device as contiguous stacks the expert-indexed and grouped ops can address
    /// (<see cref="IBackend.PreloadWeightGroups"/> places them so). A layer whose experts are lazily uploaded, offloaded or re-placed keeps the host-routed path.</summary>
    bool MoeExpertsResident(IReadOnlyList<Tensor> gateExperts, IReadOnlyList<Tensor> upExperts, IReadOnlyList<Tensor> downExperts) => false;

    /// <summary>True when <see cref="MoeExpertsGrouped"/> and <see cref="MoeCombinePairs"/> run experts stored as <paramref name="expertType"/>.</summary>
    bool SupportsMoeExpertsGrouped(DType expertType) => false;

    /// <summary>The per-expert GEMMs of a large MoE batch, with the routing already on the device: for every expert <c>e</c> with rows
    /// <c>offsets[e]..offsets[e+1]</c> of the expert-major order, <c>expertOut[rows] = down[e] . (act(gate[e] . x[perm]) * (up[e] . x[perm]))</c>.
    /// Nothing is read back; the caller reads <paramref name="offsets"/> from <see cref="MoeBuildDispatch"/> once per layer.</summary>
    /// <param name="expertOut">F32 <c>[1, rows, hidden]</c>, rows in expert-major order.</param>
    /// <param name="x">F32 <c>[1, tokens, hidden]</c>.</param>
    /// <param name="permutedToken">I32 <c>[tokens * topk]</c> from <see cref="MoeBuildDispatch"/>: the token behind each expert-major row.</param>
    /// <param name="offsets">Host copy of the I32 <c>[E + 1]</c> exclusive scan from <see cref="MoeBuildDispatch"/>.</param>
    void MoeExpertsGrouped(Tensor expertOut, Tensor x, Tensor permutedToken, ReadOnlySpan<int> offsets, IReadOnlyList<Tensor> gateExperts,
        IReadOnlyList<Tensor> upExperts, IReadOnlyList<Tensor> downExperts, bool gelu) =>
        throw NotSupportedPrimitive(nameof(MoeExpertsGrouped));

    /// <summary>Weighted sum of each token's expert-major rows through the pair map of <see cref="MoeBuildDispatch"/>, plus the optional
    /// shared-expert row: <c>output[t] = sigmoid(sharedGateLogit[t]) * shared[t] + sum_j topkWeight[t,j] * expertOut[pairSlot[t,j]]</c>.</summary>
    void MoeCombinePairs(Tensor output, Tensor expertOut, Tensor pairSlot, Tensor topkWeight, Tensor? shared, Tensor? sharedGateLogit, int topk) =>
        throw NotSupportedPrimitive(nameof(MoeCombinePairs));

    private NotSupportedException NotSupportedPrimitive(string name) =>
        new NotSupportedException($"{name} is not implemented on the {Device} backend.");
}
