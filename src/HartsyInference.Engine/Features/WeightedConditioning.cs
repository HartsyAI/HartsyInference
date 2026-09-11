using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Features;

/// <summary>Builds a <see cref="ConditioningSchedule"/> for ComfyUI-style prompt weighting and <c>&lt;break&gt;</c> chunking on CLIP pipelines whose denoise loop consumes a batched <c>[2, seqLen, hidden]</c> (negative, positive) tensor. Returns null when neither prompt uses weighting syntax, so ordinary prompts keep the byte-identical plain-encode path at zero cost.</summary>
public static class WeightedConditioning
{
    /// <summary>Cheap pre-check for weighting <c>( )</c> or <c>&lt;break&gt;</c> — the only two things the weighted
    /// path handles differently from a plain encode. A bare <c>[</c> is deliberately NOT a trigger: brackets no
    /// longer carry alternation/scheduling (SwarmUI's 2026-09-01 parser emits <c>&lt;alternate:&gt;</c>/
    /// <c>&lt;fromto[N]:&gt;</c> tags instead, which <c>BuildSingleClipScheduled</c>/<c>BuildDualClipScheduled</c> test
    /// for separately) and <see cref="Diffusion.Prompting.PromptWeighting"/> leaves them untouched, so counting
    /// them here only dragged bracket-bearing prose onto the schedule path — which on SD1.5 forfeits the fused
    /// Euler loop and makes a non-default sampler selection throw.</summary>
    public static bool HasWeightingSyntax(params string?[] prompts)
    {
        ArgumentNullException.ThrowIfNull(prompts);
        foreach (string? p in prompts)
        {
            if (string.IsNullOrEmpty(p))
            {
                continue;
            }
            if (p.Contains('(', StringComparison.Ordinal) || p.Contains("<break>", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Single-CLIP (SD 1.5) weighted conditioning: a one-variant schedule holding <c>[2, 77*chunks, hidden]</c> (negative, positive), or null when there's no weighting syntax. <paramref name="layersFromEnd"/> is CLIP-skip (1 = last layer).</summary>
    public static ConditioningSchedule? BuildSingleClip(
        IBackend backend, ClipTextEncoder encoder, ClipTokenizer tokenizer,
        string? positive, string? negative, int layersFromEnd)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (!HasWeightingSyntax(positive, negative))
        {
            return null;
        }
        Tensor batched = EncodeSingleClipPair(backend, encoder, tokenizer, positive ?? "", negative ?? "", layersFromEnd);
        return new ConditioningSchedule
        {
            Variants = [batched],
            IndexForStep = static (_, _) => 0,
        };
    }

    /// <summary>Single-CLIP (SD 1.5) weighted + scheduled conditioning: like <see cref="BuildSingleClip"/>, but
    /// when <paramref name="positive"/> or <paramref name="negative"/> carries an <c>&lt;alternate:&gt;</c>/
    /// <c>&lt;fromto[N]:&gt;</c> tag (see <see cref="PromptTagScheduling"/>), the returned schedule has one
    /// encoded variant per distinct (positive-variant, negative-variant) pair that actually occurs across
    /// <paramref name="totalSteps"/> — the common case (no scheduling tags) is byte-identical to
    /// <see cref="BuildSingleClip"/>, since it just delegates there.</summary>
    public static ConditioningSchedule? BuildSingleClipScheduled(
        IBackend backend, ClipTextEncoder encoder, ClipTokenizer tokenizer,
        string? positive, string? negative, int layersFromEnd, int totalSteps)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(tokenizer);
        string pos = positive ?? "";
        string neg = negative ?? "";
        bool posScheduled = PromptTagScheduling.HasScheduling(pos);
        bool negScheduled = PromptTagScheduling.HasScheduling(neg);
        if (!posScheduled && !negScheduled)
        {
            return BuildSingleClip(backend, encoder, tokenizer, positive, negative, layersFromEnd);
        }
        PromptSchedule posSchedule = posScheduled ? PromptTagScheduling.Resolve(pos, totalSteps) : SingleVariantSchedule(pos, totalSteps);
        PromptSchedule negSchedule = negScheduled ? PromptTagScheduling.Resolve(neg, totalSteps) : SingleVariantSchedule(neg, totalSteps);
        (IReadOnlyList<(int PosIdx, int NegIdx)> pairs, int[] stepToVariant) = PairSchedules(posSchedule, negSchedule, totalSteps);
        Tensor[] variants = new Tensor[pairs.Count];
        for (int k = 0; k < pairs.Count; k++)
        {
            variants[k] = EncodeSingleClipPair(
                backend, encoder, tokenizer, posSchedule.Variants[pairs[k].PosIdx], negSchedule.Variants[pairs[k].NegIdx], layersFromEnd);
        }
        return new ConditioningSchedule
        {
            Variants = variants,
            IndexForStep = (step, total) => stepToVariant[Math.Clamp(step, 0, stepToVariant.Length - 1)],
        };
    }

    /// <summary>Dual-CLIP (SDXL) weighted conditioning: <c>[2, 77*chunks, 2048]</c> (negative, positive) — penultimate CLIP-L (768) concatenated with penultimate CLIP-G (1280) on the last dim, matching the SDXL pipeline's plain textEmbeddings. Both encoders share SDXL's single BPE tokenizer, so the per-chunk seqLens align without padding. The pooled vector is left to the pipeline's own plain encode. <paramref name="layersFromEnd"/> is 2 by SDXL spec.</summary>
    public static ConditioningSchedule? BuildDualClip(
        IBackend backend, ClipTextEncoder clipL, ClipTextEncoder clipG, ClipTokenizer tokenizer,
        string? positive, string? negative, int layersFromEnd)
    {
        ArgumentNullException.ThrowIfNull(clipL);
        ArgumentNullException.ThrowIfNull(clipG);
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (!HasWeightingSyntax(positive, negative))
        {
            return null;
        }
        Tensor batched = EncodeDualClipPair(backend, clipL, clipG, tokenizer, positive ?? "", negative ?? "", layersFromEnd);
        return new ConditioningSchedule
        {
            Variants = [batched],
            IndexForStep = static (_, _) => 0,
        };
    }

    /// <summary>Dual-CLIP (SDXL) weighted + scheduled conditioning: like <see cref="BuildDualClip"/>, but when
    /// <paramref name="positive"/> or <paramref name="negative"/> carries an <c>&lt;alternate:&gt;</c>/
    /// <c>&lt;fromto[N]:&gt;</c> tag (see <see cref="PromptTagScheduling"/>), the returned schedule has one
    /// encoded variant per distinct (positive-variant, negative-variant) pair that actually occurs across
    /// <paramref name="totalSteps"/> — the common case (no scheduling tags) is byte-identical to
    /// <see cref="BuildDualClip"/>, since it just delegates there.</summary>
    public static ConditioningSchedule? BuildDualClipScheduled(
        IBackend backend, ClipTextEncoder clipL, ClipTextEncoder clipG, ClipTokenizer tokenizer,
        string? positive, string? negative, int layersFromEnd, int totalSteps)
    {
        ArgumentNullException.ThrowIfNull(clipL);
        ArgumentNullException.ThrowIfNull(clipG);
        ArgumentNullException.ThrowIfNull(tokenizer);
        string pos = positive ?? "";
        string neg = negative ?? "";
        bool posScheduled = PromptTagScheduling.HasScheduling(pos);
        bool negScheduled = PromptTagScheduling.HasScheduling(neg);
        if (!posScheduled && !negScheduled)
        {
            return BuildDualClip(backend, clipL, clipG, tokenizer, positive, negative, layersFromEnd);
        }
        PromptSchedule posSchedule = posScheduled ? PromptTagScheduling.Resolve(pos, totalSteps) : SingleVariantSchedule(pos, totalSteps);
        PromptSchedule negSchedule = negScheduled ? PromptTagScheduling.Resolve(neg, totalSteps) : SingleVariantSchedule(neg, totalSteps);
        (IReadOnlyList<(int PosIdx, int NegIdx)> pairs, int[] stepToVariant) = PairSchedules(posSchedule, negSchedule, totalSteps);
        Tensor[] variants = new Tensor[pairs.Count];
        for (int k = 0; k < pairs.Count; k++)
        {
            variants[k] = EncodeDualClipPair(
                backend, clipL, clipG, tokenizer, posSchedule.Variants[pairs[k].PosIdx], negSchedule.Variants[pairs[k].NegIdx], layersFromEnd);
        }
        return new ConditioningSchedule
        {
            Variants = variants,
            IndexForStep = (step, total) => stepToVariant[Math.Clamp(step, 0, stepToVariant.Length - 1)],
        };
    }

    /// <summary>Encodes one (positive, negative) pair into a single <c>[2, S, 2048]</c> dual-CLIP batched
    /// tensor — the shared per-variant body both <see cref="BuildDualClip"/> and
    /// <see cref="BuildDualClipScheduled"/> run, once per schedule variant.</summary>
    private static Tensor EncodeDualClipPair(
        IBackend backend, ClipTextEncoder clipL, ClipTextEncoder clipG, ClipTokenizer tokenizer,
        string positive, string negative, int layersFromEnd)
    {
        (IReadOnlyList<int[]> posIds, IReadOnlyList<float[]> posW) = WeightedPromptTokenizer.Tokenize(tokenizer, positive);
        (IReadOnlyList<int[]> negIds, IReadOnlyList<float[]> negW) = WeightedPromptTokenizer.Tokenize(tokenizer, negative);
        EqualizeChunkCount(tokenizer, ref posIds, ref posW, ref negIds, ref negW);

        Tensor posL = EncodePenultimateHidden(backend, clipL, posIds, posW, layersFromEnd);
        Tensor negL = EncodePenultimateHidden(backend, clipL, negIds, negW, layersFromEnd);
        Tensor posG = EncodePenultimateHidden(backend, clipG, posIds, posW, layersFromEnd);
        Tensor negG = EncodePenultimateHidden(backend, clipG, negIds, negW, layersFromEnd);

        Tensor posConcat = CfgHelper.ConcatLastDim(posL, posG);
        Tensor negConcat = CfgHelper.ConcatLastDim(negL, negG);
        posL.Dispose();
        negL.Dispose();
        posG.Dispose();
        negG.Dispose();

        try
        {
            return StackBatch2(negConcat, posConcat);
        }
        finally
        {
            posConcat.Dispose();
            negConcat.Dispose();
        }
    }

    /// <summary>Encodes one (positive, negative) pair into a single <c>[2, S, hidden]</c> single-CLIP batched
    /// tensor — the shared per-variant body both <see cref="BuildSingleClip"/> and
    /// <see cref="BuildSingleClipScheduled"/> run, once per schedule variant.</summary>
    private static Tensor EncodeSingleClipPair(
        IBackend backend, ClipTextEncoder encoder, ClipTokenizer tokenizer,
        string positive, string negative, int layersFromEnd)
    {
        (IReadOnlyList<int[]> posIds, IReadOnlyList<float[]> posW) = WeightedPromptTokenizer.Tokenize(tokenizer, positive);
        (IReadOnlyList<int[]> negIds, IReadOnlyList<float[]> negW) = WeightedPromptTokenizer.Tokenize(tokenizer, negative);
        EqualizeChunkCount(tokenizer, ref posIds, ref posW, ref negIds, ref negW);

        Tensor posCond = encoder.EncodeWeighted(backend, posIds, posW, layersFromEnd);
        Tensor negCond = encoder.EncodeWeighted(backend, negIds, negW, layersFromEnd);
        try
        {
            return StackBatch2(negCond, posCond);
        }
        finally
        {
            posCond.Dispose();
            negCond.Dispose();
        }
    }

    /// <summary>A degenerate <see cref="PromptSchedule"/> for the side of a pair (positive or negative) that
    /// carries no scheduling tag of its own: one variant, used at every step.</summary>
    private static PromptSchedule SingleVariantSchedule(string prompt, int totalSteps) => new PromptSchedule([prompt], new int[totalSteps]);

    /// <summary>Pairs a positive and negative <see cref="PromptSchedule"/> step-for-step into the distinct
    /// (positive-variant, negative-variant) combinations that actually occur across <paramref name="totalSteps"/>
    /// — so, for the overwhelmingly common case of only one side varying, the encoded-variant count matches
    /// that side's variant count exactly rather than the full cross product.</summary>
    private static (IReadOnlyList<(int PosIdx, int NegIdx)> Pairs, int[] StepToVariant) PairSchedules(
        PromptSchedule positive, PromptSchedule negative, int totalSteps)
    {
        List<(int PosIdx, int NegIdx)> pairs = new List<(int PosIdx, int NegIdx)>();
        Dictionary<(int, int), int> seen = new Dictionary<(int, int), int>();
        int[] stepToVariant = new int[totalSteps];
        for (int s = 0; s < totalSteps; s++)
        {
            (int, int) key = (positive.StepToVariant[s], negative.StepToVariant[s]);
            if (!seen.TryGetValue(key, out int idx))
            {
                idx = pairs.Count;
                pairs.Add(key);
                seen[key] = idx;
            }
            stepToVariant[s] = idx;
        }
        return (pairs, stepToVariant);
    }

    /// <summary>Weighted penultimate hidden states for one prompt; the pooled output is discarded because the SDXL pipeline sources pooled from its own plain encode.</summary>
    private static Tensor EncodePenultimateHidden(IBackend backend, ClipTextEncoder encoder,
        IReadOnlyList<int[]> ids, IReadOnlyList<float[]> weights, int layersFromEnd)
    {
        (Tensor hidden, Tensor? pooled) = encoder.EncodeWeightedPenultimate(backend, ids, weights, ReadOnlySpan<int>.Empty, layersFromEnd);
        pooled?.Dispose();
        return hidden;
    }

    /// <summary>Pads the shorter of (positive, negative) with empty SOT..EOT chunks so both have the same chunk count — required before stacking into one <c>[2, …]</c> tensor.</summary>
    private static void EqualizeChunkCount(
        ClipTokenizer tokenizer,
        ref IReadOnlyList<int[]> posIds, ref IReadOnlyList<float[]> posW,
        ref IReadOnlyList<int[]> negIds, ref IReadOnlyList<float[]> negW)
    {
        int target = Math.Max(posIds.Count, negIds.Count);
        if (posIds.Count == negIds.Count)
        {
            return;
        }
        // An empty prompt tokenizes to exactly one bare SOT..EOT pad chunk — the neutral filler.
        (IReadOnlyList<int[]> emptyIds, IReadOnlyList<float[]> emptyW) = WeightedPromptTokenizer.Tokenize(tokenizer, "");
        int[] padIds = emptyIds[0];
        float[] padW = emptyW[0];

        posIds = Pad(posIds, padIds, target);
        posW = Pad(posW, padW, target);
        negIds = Pad(negIds, padIds, target);
        negW = Pad(negW, padW, target);
    }

    private static IReadOnlyList<T> Pad<T>(IReadOnlyList<T> list, T pad, int target)
    {
        if (list.Count >= target)
        {
            return list;
        }
        List<T> result = new List<T>(list);
        while (result.Count < target)
        {
            result.Add(pad);
        }
        return result;
    }

    /// <summary>Stacks two <c>[1, S, H]</c> F32 tensors into <c>[2, S, H]</c> — row 0 = uncond, row 1 = cond, the batch the CFG loop slices.</summary>
    private static unsafe Tensor StackBatch2(Tensor first, Tensor second)
    {
        if (first.Shape.Rank != 3 || second.Shape.Rank != 3)
        {
            throw new ArgumentException("StackBatch2 expects rank-3 [1,S,H] tensors.", nameof(first));
        }
        if (!first.Shape.Equals(second.Shape))
        {
            throw new ArgumentException($"StackBatch2 shape mismatch: {first.Shape} vs {second.Shape}.", nameof(second));
        }
        if (first.DType != DType.F32 || second.DType != DType.F32)
        {
            throw new ArgumentException("StackBatch2 expects F32 tensors.", nameof(first));
        }
        long s = first.Shape[1];
        long h = first.Shape[2];
        Tensor result = new Tensor(new TensorShape(2, s, h), DType.F32);
        long rowBytes = s * h * sizeof(float);
        byte* dst = (byte*)result.DataPointer;
        Buffer.MemoryCopy((void*)first.DataPointer, dst, rowBytes, rowBytes);
        Buffer.MemoryCopy((void*)second.DataPointer, dst + rowBytes, rowBytes, rowBytes);
        return result;
    }
}
