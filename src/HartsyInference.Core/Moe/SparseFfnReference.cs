using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Moe;

/// <summary>
/// Reference forward for one routed sparse layer, composed from the backend op contracts: <see cref="MoeReference.Route"/>,
/// <see cref="MoeReference.BuildDispatch"/>, the program oracle, and <see cref="MoeReference.Combine"/>. Shared experts are
/// not part of this path yet; a layer with a shared group is rejected rather than silently dropped.
/// </summary>
public static class SparseFfnReference
{
    /// <summary>Runs a routed layer over <paramref name="tokens"/> rows on the host.</summary>
    /// <param name="layer">Validated sparse layer; its routed group must match <paramref name="experts"/>.</param>
    /// <param name="prefill">Selects the prefill top-k (true) or the decode top-k (false).</param>
    /// <param name="x"><c>tokens × H</c> inputs.</param>
    /// <param name="tokens">Token rows.</param>
    /// <param name="routerLogits"><c>tokens × E</c> router logits.</param>
    /// <param name="experts">One weight set per routed expert, index-aligned with the router.</param>
    /// <param name="selectionBias">Per-expert selection bias (<c>e_score_correction_bias</c>), or empty. Required when the router has one.</param>
    /// <param name="y"><c>tokens × H</c> outputs, overwritten.</param>
    /// <param name="altSelectionBias">Second selection bias used for tokens whose kind is non-zero. Required with token-kind routing.</param>
    /// <param name="tokenKinds">Per-token kind, one entry per token. Required with token-kind routing.</param>
    public static void Run(MoeLayerDescriptor layer, bool prefill, ReadOnlySpan<float> x, int tokens,
        ReadOnlySpan<float> routerLogits, IReadOnlyList<F32ExpertWeights> experts, Span<float> y, ReadOnlySpan<float> selectionBias = default,
        ReadOnlySpan<float> altSelectionBias = default, ReadOnlySpan<int> tokenKinds = default)
    {
        ArgumentNullException.ThrowIfNull(layer);
        ArgumentNullException.ThrowIfNull(experts);
        layer.Validated();
        if (layer.Shared is not null) throw new NotSupportedException("The reference layer path does not run shared experts yet.");
        if (layer.Router.BiasSpace == SelectionBiasSpace.Logit)
            throw new NotSupportedException(
                "Flat logit-space selection (production SigmoidLogitAdd) is not in MoeReference yet; the backend selects on the score.");
        if (!layer.Router.CanLowerToBackend)
            throw new NotSupportedException(
                $"{layer.Router.NumExperts} experts exceed the backend router limit; the reference has no host router for this width yet.");
        int e = layer.ExpertCount, h = layer.Routed.Shape.HiddenSize;
        if (experts.Count != e) throw new ArgumentException($"Layer routes {e} experts; {experts.Count} weight sets were supplied.", nameof(experts));
        if (x.Length != (long)tokens * h) throw new ArgumentException($"x must hold {tokens} rows of {h}.", nameof(x));
        if (y.Length != (long)tokens * h) throw new ArgumentException($"y must hold {tokens} rows of {h}.", nameof(y));
        if (routerLogits.Length != (long)tokens * e) throw new ArgumentException($"Logits must hold {tokens} rows of {e}.", nameof(routerLogits));
        for (int i = 0; i < e; i++)
        {
            ArgumentNullException.ThrowIfNull(experts[i]);
            ExpertDescriptor shape = layer.Routed.ExpertAt(i);
            // The reference feeds every expert the same input rows, so every expert must read the layer's width.
            if (shape.HiddenSize != h)
                throw new ArgumentException($"Expert {i} reads width {shape.HiddenSize}; the layer's input width is {h}.", nameof(layer));
            if (experts[i].Hidden != shape.HiddenSize || experts[i].Intermediate != shape.IntermediateSize)
                throw new ArgumentException($"Expert {i} weights do not match its descriptor shape.", nameof(experts));
            experts[i].Validated();
        }
        if (!selectionBias.IsEmpty && selectionBias.Length != e) throw new ArgumentException($"Selection bias must hold {e} values.",
                nameof(selectionBias));
        if (layer.Router.HasSelectionBias && selectionBias.IsEmpty)
            throw new ArgumentException("The router has a selection bias; supply it.", nameof(selectionBias));
        if (!layer.Router.HasSelectionBias && !selectionBias.IsEmpty)
            throw new ArgumentException(
                "The router has no selection bias; a bias here would change expert selection away from the layer's definition.",
                nameof(selectionBias));
        if (layer.Router.HasTokenKindBias)
        {
            if (altSelectionBias.Length != e) throw new ArgumentException($"Alternate selection bias must hold {e} values.",
                    nameof(altSelectionBias));
            if (tokenKinds.Length != tokens) throw new ArgumentException($"Token kinds must hold {tokens} entries.", nameof(tokenKinds));
        }
        else if (!altSelectionBias.IsEmpty || !tokenKinds.IsEmpty)
            throw new ArgumentException("The router has no token-kind bias; alternate bias and token kinds would change selection.",
                    nameof(tokenKinds));
        if (tokens == 0) return;

        int k = prefill ? layer.Router.TopKPrefill : layer.Router.TopKDecode;
        // Only the selected phase's buffers are allocated, so only its size can overflow 32-bit offsets.
        if ((long)tokens * k * h > int.MaxValue || (long)tokens * k > int.MaxValue)
            throw new ArgumentException("The reference path indexes with 32-bit offsets; split the batch.", nameof(tokens));
        MoeRouteArgs args = layer.Router.ToRouteArgs(prefill);
        int pairs = tokens * k;

        using Tensor logits = Host(new TensorShape(tokens, e), DType.F32, routerLogits);
        using Tensor topkIdx = Host(new TensorShape(tokens, k), DType.I32);
        using Tensor topkWeight = Host(new TensorShape(tokens, k), DType.F32);
        using Tensor? bias = selectionBias.IsEmpty ? null : Host(new TensorShape(e), DType.F32, selectionBias);
        using Tensor? altBias = altSelectionBias.IsEmpty ? null : Host(new TensorShape(e), DType.F32, altSelectionBias);
        using Tensor? kinds = tokenKinds.IsEmpty ? null : HostInts(new TensorShape(tokens), tokenKinds);
        MoeReference.Route(topkIdx, topkWeight, logits, args, bias: bias, altBias: altBias, tokenKinds: kinds);

        using Tensor counts = Host(new TensorShape(e), DType.I32);
        using Tensor offsets = Host(new TensorShape(e + 1), DType.I32);
        using Tensor permutedToken = Host(new TensorShape(pairs), DType.I32);
        using Tensor pairSlot = Host(new TensorShape(pairs), DType.I32);
        MoeReference.BuildDispatch(counts, offsets, permutedToken, pairSlot, topkIdx, e);

        float[] expertOut = new float[(long)pairs * h];
        int[] slots = ReadInts(offsets, e + 1);
        int[] tokenOfSlot = ReadInts(permutedToken, pairs);
        float[] gathered = new float[(long)pairs * h];
        for (int ex = 0; ex < e; ex++)
        {
            int begin = slots[ex], rows = slots[ex + 1] - begin;
            if (rows == 0) continue;
            for (int r = 0; r < rows; r++)
                x.Slice(tokenOfSlot[begin + r] * h, h).CopyTo(gathered.AsSpan((begin + r) * h, h));
            ExpertProgramReference.Apply(layer.Program, experts[ex], gathered.AsSpan(begin * h, rows * h), rows,
                expertOut.AsSpan(begin * h, rows * h));
        }

        using Tensor expertTensor = Host(new TensorShape(pairs, h), DType.F32, expertOut);
        using Tensor output = Host(new TensorShape(tokens, h), DType.F32);
        MoeReference.Combine(output, expertTensor, pairSlot, topkWeight, k, accumulate: false);
        output.AsReadOnlySpan<float>().CopyTo(y);
    }

    private static Tensor Host(TensorShape shape, DType dtype) => new(shape, dtype);

    private static Tensor Host(TensorShape shape, DType dtype, ReadOnlySpan<float> values)
    {
        Tensor t = new(shape, dtype);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static Tensor HostInts(TensorShape shape, ReadOnlySpan<int> values)
    {
        Tensor t = new(shape, DType.I32);
        values.CopyTo(t.AsSpan<int>());
        return t;
    }

    private static int[] ReadInts(Tensor t, int count)
    {
        int[] r = new int[count];
        t.AsReadOnlySpan<int>().Slice(0, count).CopyTo(r);
        return r;
    }
}
