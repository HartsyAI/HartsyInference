using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Lora;

/// <summary>LyCORIS LoKr: <c>ΔW = W1 ⊗ W2</c> (Kronecker product). Either factor may be stored whole (<c>lokr_w1</c> / <c>lokr_w2</c>) or as its own low-rank pair (<c>lokr_w1_a @ lokr_w1_b</c>); the right factor may additionally carry a CP/Tucker core (<c>lokr_t2</c>) for conv kernels.</summary>
/// <remarks><para>Follows ComfyUI's <c>weight_adapter/lokr.py</c> including its scale rule, which is easy to get
/// wrong: alpha is divided by the rank of the LAST factored side, and when NEITHER side is factored the scale is
/// 1.0 regardless of the alpha the file stores.</para>
/// <para>A conv target keeps its kernel axes on the right factor only; ComfyUI unsqueezes the left factor to
/// <c>[a, b, 1, 1]</c> before the product, which is exactly the 2-D Kronecker product of <c>W1</c> with the right
/// factor flattened from dim 1 — so this materializes the flattened form directly.</para></remarks>
public sealed class LoKrDelta : LoraDelta
{
    /// <summary>Left factor stored whole, or null when it is factored into <see cref="W1A"/>/<see cref="W1B"/>.</summary>
    public Tensor? W1 { get; init; }

    /// <summary>Left factor's low-rank left matrix; paired with <see cref="W1B"/>.</summary>
    public Tensor? W1A { get; init; }

    /// <summary>Left factor's low-rank right matrix; paired with <see cref="W1A"/>.</summary>
    public Tensor? W1B { get; init; }

    /// <summary>Right factor stored whole, or null when it is factored into <see cref="W2A"/>/<see cref="W2B"/>.</summary>
    public Tensor? W2 { get; init; }

    /// <summary>Right factor's low-rank left matrix; paired with <see cref="W2B"/>.</summary>
    public Tensor? W2A { get; init; }

    /// <summary>Right factor's low-rank right matrix; paired with <see cref="W2A"/>.</summary>
    public Tensor? W2B { get; init; }

    /// <summary>Right factor's CP/Tucker core, or null. Only meaningful with <see cref="W2A"/>/<see cref="W2B"/>.</summary>
    public Tensor? T2 { get; init; }

    /// <summary>Alpha as stored in the file, or the rank when the file carries none.</summary>
    public required float Alpha { get; init; }

    /// <inheritdoc/>
    public override LoraVariant Variant => LoraVariant.LoKr;

    /// <inheritdoc/>
    public override long OutFeatures => LeftRows * RightRows;

    /// <inheritdoc/>
    public override long InFeatures => LeftColumns * RightColumns;

    /// <summary>Alpha divided by the rank of the last factored side, or 1.0 when both factors are stored whole — ComfyUI's <c>dim is None</c> case, where the file's alpha is deliberately ignored.</summary>
    public override float Scale
    {
        get
        {
            int? dimension = FactoredRank;
            return dimension is null ? 1.0f : Alpha / dimension.Value;
        }
    }

    /// <summary>Rank ComfyUI divides alpha by: <c>w1_b</c>'s leading dimension, overridden by <c>w2_b</c>'s when the right factor is also factored. Null when neither side is.</summary>
    private int? FactoredRank
    {
        get
        {
            int? dimension = W1 is null && W1B is not null ? (int)W1B.Shape[0] : null;
            if (W2 is null && W2B is not null)
            {
                dimension = (int)W2B.Shape[0];
            }
            return dimension;
        }
    }

    private long LeftRows => W1 is not null ? W1.Shape[0] : W1A!.Shape[0];

    private long LeftColumns => W1 is not null ? W1.Shape[1] : W1B!.Shape[1];

    private long RightRows => W2 is not null ? W2.Shape[0] : W2A!.Shape[0];

    private long RightColumns
    {
        get
        {
            if (W2 is not null)
            {
                return W2.Shape.ElementCount / W2.Shape[0];
            }
            return T2 is not null ? W2B!.Shape[1] * T2.Shape[2] * T2.Shape[3] : W2B!.Shape[1];
        }
    }

    /// <inheritdoc/>
    public override Tensor ComputeF32(IBackend backend)
    {
        Tensor left = RebuildLeft(backend);
        try
        {
            Tensor right = RebuildRight(backend);
            try
            {
                return Kronecker(left, right);
            }
            finally
            {
                right.Dispose();
            }
        }
        finally
        {
            left.Dispose();
        }
    }

    /// <summary>Kronecker product of two rank-2 F32 matrices. Host math on the load path; the result is the full delta, so there is no cheaper form.</summary>
    private static Tensor Kronecker(Tensor left, Tensor right)
    {
        long leftRows = left.Shape[0], leftColumns = left.Shape[1];
        long rightRows = right.Shape[0], rightColumns = right.Shape[1];
        Tensor result = new(new TensorShape(leftRows * rightRows, leftColumns * rightColumns), DType.F32);
        try
        {
            ReadOnlySpan<float> leftData = left.AsReadOnlySpan<float>();
            ReadOnlySpan<float> rightData = right.AsReadOnlySpan<float>();
            Span<float> output = result.AsSpan<float>();
            long outputColumns = leftColumns * rightColumns;
            for (long lr = 0; lr < leftRows; lr++)
            {
                for (long lc = 0; lc < leftColumns; lc++)
                {
                    float scalar = leftData[(int)(lr * leftColumns + lc)];
                    for (long rr = 0; rr < rightRows; rr++)
                    {
                        long outputBase = (lr * rightRows + rr) * outputColumns + lc * rightColumns;
                        long rightBase = rr * rightColumns;
                        for (long rc = 0; rc < rightColumns; rc++)
                        {
                            output[(int)(outputBase + rc)] = scalar * rightData[(int)(rightBase + rc)];
                        }
                    }
                }
            }
            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
    }

    private Tensor RebuildLeft(IBackend backend)
    {
        if (W1 is not null)
        {
            if (W1.Shape.Rank != 2)
                throw new HartsyInferenceException(
                    $"LoKr left factor must be rank-2, got {W1.Shape}; only the right factor carries kernel axes.");
            return W1.CastTo(DType.F32);
        }
        if (W1A is null || W1B is null)
            throw new HartsyInferenceException("LoKr adapter has neither lokr_w1 nor a complete lokr_w1_a / lokr_w1_b pair.");
        using Tensor leftF32 = W1A.CastTo(DType.F32);
        using Tensor rightF32 = W1B.CastTo(DType.F32);
        return MatMulF32(backend, leftF32, rightF32);
    }

    private Tensor RebuildRight(IBackend backend)
    {
        if (W2 is not null)
        {
            Tensor whole = W2.CastTo(DType.F32);
            if (whole.Shape.Rank == 2)
            {
                return whole;
            }
            // A view would borrow the cast's buffer and leave nothing owning it, so the kernel axes are folded by copy.
            try
            {
                Tensor flat = new(new TensorShape(whole.Shape[0], whole.Shape.ElementCount / whole.Shape[0]), DType.F32);
                try
                {
                    whole.AsReadOnlySpan<float>().CopyTo(flat.AsSpan<float>());
                    return flat;
                }
                catch
                {
                    flat.Dispose();
                    throw;
                }
            }
            finally
            {
                whole.Dispose();
            }
        }
        if (W2A is null || W2B is null)
            throw new HartsyInferenceException("LoKr adapter has neither lokr_w2 nor a complete lokr_w2_a / lokr_w2_b pair.");
        if (T2 is not null)
        {
            return TuckerRebuildF32(T2, W2A, W2B);
        }
        using Tensor leftF32 = W2A.CastTo(DType.F32);
        using Tensor rightF32 = W2B.CastTo(DType.F32);
        return MatMulF32(backend, leftF32, rightF32);
    }
}
