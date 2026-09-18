using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Lora;

/// <summary>The ordinary low-rank delta: <c>ΔW = B @ A</c> with <c>A</c> the down matrix <c>[rank, in]</c> and <c>B</c> the up matrix <c>[out, rank]</c>. A conv adapter stores <c>A</c> as <c>[rank, in, kh, kw]</c> and <c>B</c> as <c>[out, rank, 1, 1]</c>; both fold into the same GEMM.</summary>
public sealed class StandardLoraDelta : LoraDelta
{
    /// <summary>Down (A) matrix, <c>[rank, in]</c> or <c>[rank, in, kh, kw]</c>.</summary>
    public required Tensor Down { get; init; }

    /// <summary>Up (B) matrix, <c>[out, rank]</c> or <c>[out, rank, 1, 1]</c>.</summary>
    public required Tensor Up { get; init; }

    /// <summary>Alpha as stored in the file, or the rank when the file carries none (scale 1.0).</summary>
    public required float Alpha { get; init; }

    /// <summary>Intrinsic rank — the down matrix's leading dimension.</summary>
    public int Rank => (int)Down.Shape[0];

    /// <inheritdoc/>
    public override LoraVariant Variant => DoraScale is null ? LoraVariant.StandardLora : LoraVariant.DoRA;

    /// <inheritdoc/>
    public override long OutFeatures => Up.Shape[0];

    /// <inheritdoc/>
    public override long InFeatures => Down.Shape.ElementCount / Down.Shape[0];

    /// <inheritdoc/>
    public override float Scale => Alpha / Rank;

    /// <inheritdoc/>
    public override Tensor ComputeF32(IBackend backend)
    {
        Tensor upF32 = Up.CastTo(DType.F32);
        Tensor downF32 = Down.CastTo(DType.F32);
        Tensor upFlat = FlattenFromDim1(upF32);
        Tensor downFlat = FlattenFromDim1(downF32);
        try
        {
            return MatMulF32(backend, upFlat, downFlat);
        }
        finally
        {
            if (!ReferenceEquals(downFlat, downF32)) downFlat.Dispose();
            if (!ReferenceEquals(upFlat, upF32)) upFlat.Dispose();
            downF32.Dispose();
            upF32.Dispose();
        }
    }
}
