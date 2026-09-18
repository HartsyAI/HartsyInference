using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Lora;

/// <summary>LyCORIS LoHa: <c>ΔW = (W1a @ W1b) ⊙ (W2a @ W2b)</c>, the element-wise product of two independent low-rank reconstructions. With the optional CP/Tucker cores <c>T1</c>/<c>T2</c> each factor is rebuilt by the <c>"i j k l, j r, i p -&gt; p r k l"</c> contraction instead of a plain product, which is how a conv LoHa stores its kernel.</summary>
/// <remarks>The factor ordering is ComfyUI's <c>weight_adapter/loha.py</c>: <c>m1 = mm(hada_w1_a, hada_w1_b)</c>
/// with <c>hada_w1_a</c> <c>[out, rank]</c> and <c>hada_w1_b</c> <c>[rank, in]</c>. In Tucker mode LyCORIS stores the
/// same two names transposed (<c>[rank, out]</c> / <c>[rank, in]</c>), which is why that path reads the einsum's
/// index letters rather than reusing the plain-product convention.</remarks>
public sealed class LoHaDelta : LoraDelta
{
    /// <summary>First Hadamard factor's left matrix.</summary>
    public required Tensor W1A { get; init; }

    /// <summary>First Hadamard factor's right matrix.</summary>
    public required Tensor W1B { get; init; }

    /// <summary>Second Hadamard factor's left matrix.</summary>
    public required Tensor W2A { get; init; }

    /// <summary>Second Hadamard factor's right matrix.</summary>
    public required Tensor W2B { get; init; }

    /// <summary>First factor's CP/Tucker core, or null for the plain product form. Paired with <see cref="T2"/>.</summary>
    public Tensor? T1 { get; init; }

    /// <summary>Second factor's CP/Tucker core, or null for the plain product form. Paired with <see cref="T1"/>.</summary>
    public Tensor? T2 { get; init; }

    /// <summary>Alpha as stored in the file, or the rank when the file carries none.</summary>
    public required float Alpha { get; init; }

    /// <summary>Intrinsic rank — <c>W1b</c>'s leading dimension, which ComfyUI divides alpha by.</summary>
    public int Rank => (int)W1B.Shape[0];

    /// <inheritdoc/>
    public override LoraVariant Variant => LoraVariant.LoHa;

    /// <inheritdoc/>
    public override long OutFeatures => IsTucker ? W1A.Shape[1] : W1A.Shape[0];

    /// <inheritdoc/>
    public override long InFeatures => IsTucker
        ? W1B.Shape[1] * T1!.Shape[2] * T1.Shape[3]
        : W1B.Shape.ElementCount / W1B.Shape[0];

    /// <inheritdoc/>
    public override float Scale => Alpha / Rank;

    /// <summary>Whether this adapter stores CP/Tucker cores rather than plain factor products.</summary>
    private bool IsTucker => T1 is not null && T2 is not null;

    /// <inheritdoc/>
    public override Tensor ComputeF32(IBackend backend)
    {
        if ((T1 is null) != (T2 is null))
            throw new HartsyInferenceException(
                "LoHa adapter carries only one of hada_t1 / hada_t2; the CP/Tucker form needs both.");

        Tensor first = RebuildFactor(backend, W1A, W1B, T1);
        try
        {
            Tensor second = RebuildFactor(backend, W2A, W2B, T2);
            try
            {
                if (first.Shape != second.Shape)
                    throw new HartsyInferenceException(
                        $"LoHa factors rebuild to different shapes ({first.Shape} and {second.Shape}); "
                        + "the two Hadamard halves must describe the same weight.");
                backend.Mul(first, first, second);
                return first;
            }
            finally
            {
                second.Dispose();
            }
        }
        catch
        {
            first.Dispose();
            throw;
        }
    }

    /// <summary>Rebuilds one Hadamard factor as an owned F32 <c>[out, in·kh·kw]</c> matrix.</summary>
    private static Tensor RebuildFactor(IBackend backend, Tensor left, Tensor right, Tensor? core)
    {
        if (core is not null)
        {
            return TuckerRebuildF32(core, left, right);
        }
        using Tensor leftF32 = left.CastTo(DType.F32);
        using Tensor rightF32 = right.CastTo(DType.F32);
        Tensor leftFlat = FlattenFromDim1(leftF32);
        Tensor rightFlat = FlattenFromDim1(rightF32);
        try
        {
            return MatMulF32(backend, leftFlat, rightFlat);
        }
        finally
        {
            if (!ReferenceEquals(rightFlat, rightF32)) rightFlat.Dispose();
            if (!ReferenceEquals(leftFlat, leftF32)) leftFlat.Dispose();
        }
    }
}
