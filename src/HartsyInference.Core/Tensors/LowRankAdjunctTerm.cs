using HartsyInference.Core.Exceptions;

namespace HartsyInference.Core.Tensors;

/// <summary>One additive term of a <see cref="LowRankAdjunct"/>: <c>ΔW = Scale · (Up @ Down)</c>, or <c>ΔW = Scale · Down</c> when <see cref="Up"/> is null and <see cref="Down"/> already holds the full-rank matrix.</summary>
/// <remarks><para>Both matrices are F32 and owned by whoever built the adjunct (the LoRA stack), never by the weight
/// they hang off — the weight's bytes are the checkpoint's, borrowed from an mmap.</para>
/// <para>A GEMM consumes the term as <c>y += Scale · (x·Downᵀ)·Upᵀ</c>, which is why the rank-full case keeps
/// <see cref="Up"/> null rather than carrying an identity: one GEMM instead of two.</para></remarks>
public sealed class LowRankAdjunctTerm
{
    /// <summary>The down (A) matrix, <c>[rank, inFeatures]</c> — or the whole <c>[outFeatures, inFeatures]</c> delta when <see cref="Up"/> is null.</summary>
    public required Tensor Down { get; init; }

    /// <summary>The up (B) matrix, <c>[outFeatures, rank]</c>; null means <see cref="Down"/> is already the full-rank delta.</summary>
    public Tensor? Up { get; init; }

    /// <summary>User strength times the decomposition's own <c>alpha / rank</c>, folded in once by the applier.</summary>
    public required float Scale { get; init; }

    /// <summary>Rows of the delta this term contributes — the target weight's output dimension.</summary>
    public long OutFeatures => Up is null ? Down.Shape[0] : Up.Shape[0];

    /// <summary>Columns of the delta this term contributes — the target weight's input dimension.</summary>
    public long InFeatures => Down.Shape[1];

    /// <summary>Returns this term with <see cref="Up"/> narrowed to a contiguous run of output rows; <see cref="Down"/> rides along whole because it is indexed by the INPUT dimension, which a row window leaves alone.</summary>
    /// <remarks>A rank-full term has no <see cref="Up"/> to narrow, so its <see cref="Down"/> — which IS the delta,
    /// and therefore output-row-indexed — is the thing that slices.</remarks>
    public LowRankAdjunctTerm SliceRows(long rowOffset, long rowCount)
    {
        if (Up is null)
        {
            return new LowRankAdjunctTerm { Down = Down.SliceRows(rowOffset, rowCount), Scale = Scale };
        }
        return new LowRankAdjunctTerm { Down = Down, Up = Up.SliceRows(rowOffset, rowCount), Scale = Scale };
    }

    /// <summary>Throws unless this term's shapes agree with a weight of <paramref name="outFeatures"/> × <paramref name="inFeatures"/>.</summary>
    internal void Validate(long outFeatures, long inFeatures, string weightKey)
    {
        if (Down.Shape.Rank != 2 || (Up is not null && Up.Shape.Rank != 2))
        {
            throw new HartsyInferenceException(
                $"LoRA adjunct for '{weightKey}' must be rank-2; got down {Down.Shape}"
                + (Up is null ? "." : $" and up {Up.Shape}."));
        }
        if (InFeatures != inFeatures || OutFeatures != outFeatures)
        {
            throw new HartsyInferenceException(
                $"LoRA adjunct for '{weightKey}' produces a [{OutFeatures}, {InFeatures}] delta but the weight is "
                + $"[{outFeatures}, {inFeatures}].");
        }
        if (Up is not null && Up.Shape[1] != Down.Shape[0])
        {
            throw new HartsyInferenceException(
                $"LoRA adjunct for '{weightKey}' has mismatched ranks: up {Up.Shape} against down {Down.Shape}.");
        }
    }
}
