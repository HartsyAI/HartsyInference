using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>Weights of one SANA-WM Gated-DeltaNet layer's main branch. Tensors may be F32 or BF16; the layer converts them once.</summary>
public sealed record class SanaWmGdnWeights
{
    /// <summary>Fused q/k/v projection <c>[3C, C]</c>, no bias; rows <c>[0, C)</c> are q, <c>[C, 2C)</c> k, <c>[2C, 3C)</c> v.</summary>
    public required Tensor Qkv { get; init; }

    /// <summary>Depthwise temporal conv on K <c>[C, 1, 4]</c>; the last tap multiplies the current frame.</summary>
    public required Tensor ConvK { get; init; }

    /// <summary>Per-token beta gate projection <c>[H, C]</c>.</summary>
    public required Tensor BetaWeight { get; init; }

    /// <summary>Per-token beta gate bias <c>[H]</c>.</summary>
    public required Tensor BetaBias { get; init; }

    /// <summary>Per-frame decay gate projection <c>[H, C]</c>, applied to the frame mean.</summary>
    public required Tensor GateWeight { get; init; }

    /// <summary>Per-frame decay gate bias <c>[H]</c>.</summary>
    public required Tensor GateBias { get; init; }

    /// <summary>Decay rate log <c>[H]</c>; the rate is <c>exp(A_log)</c>.</summary>
    public required Tensor ALog { get; init; }

    /// <summary>Decay softplus offset <c>[H]</c>.</summary>
    public required Tensor DtBias { get; init; }

    /// <summary>Full-channel RMSNorm weight for q <c>[C]</c>.</summary>
    public required Tensor QNorm { get; init; }

    /// <summary>Full-channel RMSNorm weight for k <c>[C]</c>.</summary>
    public required Tensor KNorm { get; init; }
}
