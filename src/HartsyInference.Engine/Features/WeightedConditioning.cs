using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Tokenizers;

namespace HartsyInference.Engine.Features;

/// <summary>Builds a <see cref="ConditioningSchedule"/> for ComfyUI-style prompt weighting and <c>&lt;break&gt;</c>
/// chunking on CLIP pipelines whose denoise loop consumes a batched <c>[2, seqLen, hidden]</c> (negative, positive)
/// tensor. Returns null when neither prompt uses weighting syntax, so ordinary prompts keep the byte-identical
/// plain-encode path at zero cost.</summary>
public static class WeightedConditioning
{
    /// <summary>Cheap pre-check for weighting <c>( )</c>, alternation/scheduling <c>[ ]</c>, or <c>&lt;break&gt;</c>.</summary>
    public static bool HasWeightingSyntax(params string?[] prompts)
    {
        ArgumentNullException.ThrowIfNull(prompts);
        foreach (string? p in prompts)
        {
            if (string.IsNullOrEmpty(p))
            {
                continue;
            }
            if (p.Contains('(', StringComparison.Ordinal) || p.Contains('[', StringComparison.Ordinal)
                || p.Contains("<break>", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Single-CLIP (SD 1.5) weighted conditioning: a one-variant schedule holding <c>[2, 77*chunks, hidden]</c>
    /// (negative, positive), or null when there's no weighting syntax. <paramref name="layersFromEnd"/> is CLIP-skip (1 = last layer).</summary>
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
        (IReadOnlyList<int[]> posIds, IReadOnlyList<float[]> posW) = WeightedPromptTokenizer.Tokenize(tokenizer, positive ?? "");
        (IReadOnlyList<int[]> negIds, IReadOnlyList<float[]> negW) = WeightedPromptTokenizer.Tokenize(tokenizer, negative ?? "");
        EqualizeChunkCount(tokenizer, ref posIds, ref posW, ref negIds, ref negW);

        Tensor posCond = encoder.EncodeWeighted(backend, posIds, posW, layersFromEnd);
        Tensor negCond = encoder.EncodeWeighted(backend, negIds, negW, layersFromEnd);
        Tensor batched;
        try
        {
            batched = StackBatch2(negCond, posCond);
        }
        finally
        {
            posCond.Dispose();
            negCond.Dispose();
        }
        return new ConditioningSchedule
        {
            Variants = [batched],
            IndexForStep = static (_, _) => 0,
        };
    }

    /// <summary>Dual-CLIP (SDXL) weighted conditioning: <c>[2, 77*chunks, 2048]</c> (negative, positive) — penultimate
    /// CLIP-L (768) concatenated with penultimate CLIP-G (1280) on the last dim, matching the SDXL pipeline's plain
    /// textEmbeddings. Both encoders share SDXL's single BPE tokenizer, so the per-chunk seqLens align without padding.
    /// The pooled vector is left to the pipeline's own plain encode. <paramref name="layersFromEnd"/> is 2 by SDXL spec.</summary>
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
        (IReadOnlyList<int[]> posIds, IReadOnlyList<float[]> posW) = WeightedPromptTokenizer.Tokenize(tokenizer, positive ?? "");
        (IReadOnlyList<int[]> negIds, IReadOnlyList<float[]> negW) = WeightedPromptTokenizer.Tokenize(tokenizer, negative ?? "");
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

        Tensor batched;
        try
        {
            batched = StackBatch2(negConcat, posConcat);
        }
        finally
        {
            posConcat.Dispose();
            negConcat.Dispose();
        }
        return new ConditioningSchedule
        {
            Variants = [batched],
            IndexForStep = static (_, _) => 0,
        };
    }

    /// <summary>Weighted penultimate hidden states for one prompt; the pooled output is discarded because the SDXL
    /// pipeline sources pooled from its own plain encode.</summary>
    private static Tensor EncodePenultimateHidden(IBackend backend, ClipTextEncoder encoder,
        IReadOnlyList<int[]> ids, IReadOnlyList<float[]> weights, int layersFromEnd)
    {
        (Tensor hidden, Tensor? pooled) = encoder.EncodeWeightedPenultimate(backend, ids, weights, ReadOnlySpan<int>.Empty, layersFromEnd);
        pooled?.Dispose();
        return hidden;
    }

    /// <summary>Pads the shorter of (positive, negative) with empty SOT..EOT chunks so both have the same chunk count —
    /// required before stacking into one <c>[2, …]</c> tensor.</summary>
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
