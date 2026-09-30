using System.Buffers;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>CPU reference for the MoE routing, dispatch and combine primitives; CUDA is tested against it.</summary>
/// <remarks>Route reproduces the host router of the LLM package exactly: strict-greater scans (lowest index wins a tie),
/// the same float operation order, and the HF <c>masked_fill</c> quirk for dropped groups.</remarks>
public static class MoeReference
{
    /// <summary>Checks tensor shapes, dtypes and args for <see cref="Route"/>; shared by every backend.</summary>
    public static void ValidateRoute(Tensor topkIdx, Tensor topkWeight, Tensor logits, in MoeRouteArgs args,
        Tensor? bias, Tensor? altBias, Tensor? tokenKinds)
    {
        int e = args.NumExperts, k = args.TopK;
        if (e < 1 || e > MoeRouteArgs.MaxExperts)
        {
            throw new ArgumentOutOfRangeException(nameof(args),
                $"MoeRoute NumExperts must be in [1,{MoeRouteArgs.MaxExperts}]; got {e}.");
        }
        if (k < 1 || k > e)
            throw new ArgumentOutOfRangeException(nameof(args), $"MoeRoute TopK must be in [1,{e}]; got {k}.");
        if (args.IsGrouped)
        {
            if (e % args.GroupCount != 0 || e / args.GroupCount < 2)
                throw new ArgumentException(
                    $"MoeRoute GroupCount {args.GroupCount} must divide {e} experts into groups of at least 2.", nameof(args));
            if (args.GroupsKept < 1 || args.GroupsKept > args.GroupCount)
                throw new ArgumentOutOfRangeException(nameof(args), $"MoeRoute GroupsKept must be in [1,{args.GroupCount}]; got {args.GroupsKept}.");
        }
        if (!(args.LogitDivisor > 0f)) throw new ArgumentOutOfRangeException(nameof(args), "MoeRoute LogitDivisor must be positive.");
        if (logits.DType != DType.F32 || topkWeight.DType != DType.F32)
            throw new NotSupportedException("MoeRoute supports F32 logits and weights only.");
        if (topkIdx.DType != DType.I32) throw new NotSupportedException("MoeRoute requires I32 expert indices.");
        if (logits.ElementCount == 0 || logits.ElementCount % e != 0)
            throw new ArgumentException(
                $"MoeRoute logits ({logits.ElementCount} elements) must be a non-empty multiple of {e} experts.", nameof(logits));
        long tokens = logits.ElementCount / e;
        if (topkIdx.ElementCount != tokens * k || topkWeight.ElementCount != tokens * k)
            throw new ArgumentException(
                $"MoeRoute outputs must hold {tokens}x{k} entries; got {topkIdx.ElementCount} indices, {topkWeight.ElementCount} weights.");
        CheckBias(bias, e, nameof(bias));
        CheckBias(altBias, e, nameof(altBias));
        if ((altBias is null) != (tokenKinds is null))
            throw new ArgumentException("MoeRoute altBias and tokenKinds must be supplied together.");
        if (tokenKinds is not null && (tokenKinds.DType != DType.I32 || tokenKinds.ElementCount != tokens))
            throw new ArgumentException($"MoeRoute tokenKinds must be I32 with {tokens} entries.", nameof(tokenKinds));
    }

    /// <summary>Score, biased top-k selection and gathered, renormalised, scaled weights per token.</summary>
    /// <remarks>Selection uses score + bias; the weights are the raw scores at the selected ids. Order inside a token's
    /// top-k is selection score descending, then index ascending. A non-zero token kind picks <paramref name="altBias"/>.</remarks>
    public static unsafe void Route(Tensor topkIdx, Tensor topkWeight, Tensor logits, in MoeRouteArgs args,
        Tensor? bias, Tensor? altBias, Tensor? tokenKinds)
    {
        ValidateRoute(topkIdx, topkWeight, logits, args, bias, altBias, tokenKinds);
        int e = args.NumExperts, k = args.TopK;
        long tokens = logits.ElementCount / e;
        float* lp = (float*)logits.DataPointer;
        int* ip = (int*)topkIdx.DataPointer;
        float* wp = (float*)topkWeight.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;
        float* ap = altBias is null ? null : (float*)altBias.DataPointer;
        int* kp = tokenKinds is null ? null : (int*)tokenKinds.DataPointer;

        float[] raw = ArrayPool<float>.Shared.Rent(e);
        float[] sel = ArrayPool<float>.Shared.Rent(e);
        bool[] taken = ArrayPool<bool>.Shared.Rent(e);
        int[] pick = ArrayPool<int>.Shared.Rent(k);
        try
        {
            for (long t = 0; t < tokens; t++)
            {
                Score(raw, lp + t * e, e, args);
                float* b = kp != null && kp[t] != 0 ? ap : bp;
                for (int i = 0; i < e; i++) sel[i] = b == null ? raw[i] : raw[i] + b[i];
                if (args.IsGrouped) MaskDroppedGroups(sel, e, args);

                Array.Clear(taken, 0, e);
                float wsum = 0f;
                for (int kk = 0; kk < k; kk++)
                {
                    int best = -1;
                    float bestVal = float.NegativeInfinity;
                    for (int i = 0; i < e; i++)
                        if (!taken[i] && sel[i] > bestVal) { bestVal = sel[i]; best = i; }
                    if (best < 0) best = Array.IndexOf(taken, false, 0, e);
                    taken[best] = true;
                    pick[kk] = best;
                    wsum += raw[best];
                }
                for (int kk = 0; kk < k; kk++)
                {
                    float w = raw[pick[kk]];
                    if (args.Renormalize) w /= wsum + args.RenormEpsilon;
                    w *= args.Scale;
                    ip[t * k + kk] = pick[kk];
                    wp[t * k + kk] = w;
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(raw);
            ArrayPool<float>.Shared.Return(sel);
            ArrayPool<bool>.Shared.Return(taken);
            ArrayPool<int>.Shared.Return(pick);
        }
    }

    /// <summary>Checks tensors for <see cref="BuildDispatch"/>; shared by every backend.</summary>
    public static void ValidateDispatch(Tensor counts, Tensor offsets, Tensor permutedToken, Tensor pairSlot,
        Tensor topkIdx, int numExperts)
    {
        if (numExperts < 1) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (topkIdx.Shape.Rank < 1) throw new ArgumentException("MoeBuildDispatch topkIdx must be at least rank 1.", nameof(topkIdx));
        foreach (Tensor t in new[] { counts, offsets, permutedToken, pairSlot, topkIdx })
            if (t.DType != DType.I32) throw new NotSupportedException("MoeBuildDispatch tensors must be I32.");
        long pairs = topkIdx.ElementCount;
        if (counts.ElementCount != numExperts || offsets.ElementCount != numExperts + 1)
            throw new ArgumentException($"MoeBuildDispatch counts/offsets must hold {numExperts}/{numExperts + 1} entries.");
        if (permutedToken.ElementCount != pairs || pairSlot.ElementCount != pairs)
            throw new ArgumentException($"MoeBuildDispatch permutedToken/pairSlot must hold {pairs} entries.");
    }

    /// <summary>Histogram, exclusive scan and stable expert-major permutation of the (token, slot) pairs.</summary>
    /// <remarks>Inside one expert pairs keep flat order, so equal inputs give equal permutations on every backend. An
    /// index outside [0, E) drops its pair (<c>pairSlot = -1</c>); positions past <c>offsets[E]</c> of
    /// <paramref name="permutedToken"/> are -1.</remarks>
    public static unsafe void BuildDispatch(Tensor counts, Tensor offsets, Tensor permutedToken, Tensor pairSlot,
        Tensor topkIdx, int numExperts)
    {
        ValidateDispatch(counts, offsets, permutedToken, pairSlot, topkIdx, numExperts);
        int k = (int)topkIdx.Shape[topkIdx.Shape.Rank - 1];
        long pairs = topkIdx.ElementCount;
        int* idx = (int*)topkIdx.DataPointer;
        int* cp = (int*)counts.DataPointer, op = (int*)offsets.DataPointer;
        int* pt = (int*)permutedToken.DataPointer, ps = (int*)pairSlot.DataPointer;

        for (int e = 0; e < numExperts; e++) cp[e] = 0;
        for (long p = 0; p < pairs; p++)
            if ((uint)idx[p] < (uint)numExperts) cp[idx[p]]++;
        int total = 0;
        for (int e = 0; e < numExperts; e++) { op[e] = total; total += cp[e]; }
        op[numExperts] = total;
        for (long p = total; p < pairs; p++) pt[p] = -1;

        int[] cursor = ArrayPool<int>.Shared.Rent(numExperts);
        try
        {
            for (int e = 0; e < numExperts; e++) cursor[e] = op[e];
            for (long p = 0; p < pairs; p++)
            {
                int e = idx[p];
                if ((uint)e >= (uint)numExperts) { ps[p] = -1; continue; }
                int slot = cursor[e]++;
                pt[slot] = (int)(p / k);
                ps[p] = slot;
            }
        }
        finally { ArrayPool<int>.Shared.Return(cursor); }
    }

    /// <summary>Checks tensors for <see cref="Combine"/>; shared by every backend.</summary>
    public static void ValidateCombine(Tensor output, Tensor expertOut, Tensor pairSlot, Tensor topkWeight, int k)
    {
        if (k < 1) throw new ArgumentOutOfRangeException(nameof(k));
        if (output.DType != DType.F32 || expertOut.DType != DType.F32 || topkWeight.DType != DType.F32)
            throw new NotSupportedException("MoeCombine supports F32 output, expert rows and weights only.");
        if (pairSlot.DType != DType.I32) throw new NotSupportedException("MoeCombine requires I32 pairSlot.");
        if (output.Shape.Rank < 1 || expertOut.Shape.Rank < 1) throw new ArgumentException("MoeCombine tensors must be at least rank 1.");
        long h = output.Shape[output.Shape.Rank - 1];
        if (h != expertOut.Shape[expertOut.Shape.Rank - 1])
            throw new ArgumentException("MoeCombine output and expertOut must share the row width.");
        long tokens = output.ElementCount / h;
        if (pairSlot.ElementCount != tokens * k || topkWeight.ElementCount != tokens * k)
            throw new ArgumentException($"MoeCombine pairSlot/topkWeight must hold {tokens}x{k} entries.");
    }

    /// <summary>output[t] = (accumulate ? output[t] : 0) + sum over j of topkWeight[t,j] * expertOut[pairSlot[t,j]], j ascending.</summary>
    /// <remarks>A negative slot contributes nothing. Rows are summed in a fixed order, so the result is deterministic.</remarks>
    public static unsafe void Combine(Tensor output, Tensor expertOut, Tensor pairSlot, Tensor topkWeight, int k, bool accumulate)
    {
        ValidateCombine(output, expertOut, pairSlot, topkWeight, k);
        long h = output.Shape[output.Shape.Rank - 1];
        long tokens = output.ElementCount / h;
        float* o = (float*)output.DataPointer, x = (float*)expertOut.DataPointer, w = (float*)topkWeight.DataPointer;
        int* slots = (int*)pairSlot.DataPointer;
        long expertRows = expertOut.ElementCount / h;
        for (long t = 0; t < tokens; t++)
            for (long c = 0; c < h; c++)
            {
                float acc = 0f;
                for (int j = 0; j < k; j++)
                {
                    int slot = slots[t * k + j];
                    if (slot < 0) continue;
                    if (slot >= expertRows)
                        throw new ArgumentOutOfRangeException(nameof(pairSlot), $"MoeCombine slot {slot} >= {expertRows} expert rows.");
                    acc += w[t * k + j] * x[slot * h + c];
                }
                o[t * h + c] = accumulate ? o[t * h + c] + acc : acc;
            }
    }

    private static unsafe void Score(float[] score, float* logits, int e, in MoeRouteArgs args)
    {
        float div = args.LogitDivisor;
        for (int i = 0; i < e; i++) score[i] = div == 1f ? logits[i] : logits[i] / div;
        switch (args.Scoring)
        {
            case MoeRouteScoring.Softmax:
                float max = float.NegativeInfinity;
                for (int i = 0; i < e; i++) max = MathF.Max(max, score[i]);
                float sum = 0f;
                for (int i = 0; i < e; i++) { float v = MathF.Exp(score[i] - max); score[i] = v; sum += v; }
                for (int i = 0; i < e; i++) score[i] /= sum;
                break;
            case MoeRouteScoring.Sigmoid:
                for (int i = 0; i < e; i++) score[i] = 1f / (1f + MathF.Exp(-score[i]));
                break;
            default:
                for (int i = 0; i < e; i++) score[i] = MathF.Sqrt(SoftplusReference.Scalar(score[i]));
                break;
        }
    }

    private static void MaskDroppedGroups(float[] sel, int e, in MoeRouteArgs args)
    {
        int groups = args.GroupCount, per = e / groups;
        float[] groupScore = ArrayPool<float>.Shared.Rent(groups);
        bool[] kept = ArrayPool<bool>.Shared.Rent(groups);
        try
        {
            for (int g = 0; g < groups; g++)
            {
                float top1 = float.NegativeInfinity, top2 = float.NegativeInfinity;
                for (int j = 0; j < per; j++)
                {
                    float v = sel[g * per + j];
                    if (v > top1) { top2 = top1; top1 = v; }
                    else if (v > top2) top2 = v;
                }
                groupScore[g] = top1 + top2;
                kept[g] = false;
            }
            for (int kk = 0; kk < args.GroupsKept; kk++)
            {
                int bestG = -1;
                float bestV = float.NegativeInfinity;
                for (int g = 0; g < groups; g++)
                    if (!kept[g] && groupScore[g] > bestV) { bestV = groupScore[g]; bestG = g; }
                if (bestG >= 0) kept[bestG] = true;
            }
            for (int i = 0; i < e; i++)
                if (!kept[i / per]) sel[i] = args.MaskedGroupValue;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(groupScore);
            ArrayPool<bool>.Shared.Return(kept);
        }
    }

    private static void CheckBias(Tensor? bias, int e, string name)
    {
        if (bias is null) return;
        if (bias.DType != DType.F32 || bias.ElementCount != e)
            throw new ArgumentException($"MoeRoute {name} must be F32 with {e} entries.", name);
    }
}
