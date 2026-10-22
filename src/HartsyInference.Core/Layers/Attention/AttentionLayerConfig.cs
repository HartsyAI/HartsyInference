using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Layers.Attention;

/// <summary>How the projection stage issues the Q, K, V and gate projections.</summary>
public enum AttentionProjection
{
    /// <summary>One <see cref="IBackend.Linear"/> per weight, in the order Q, K, V, gate.</summary>
    PerWeight,

    /// <summary>One <see cref="IBackend.LinearMulti"/> per distinct input: gate, Q, K and V together for self-attention,
    /// otherwise gate and Q on the query input and K and V on the key/value input, so an int8 backend quantizes each input
    /// once.</summary>
    GroupedByInput,

    /// <summary>One Linear of a fused <c>[q|k|v]</c> weight, then three <see cref="IBackend.SliceLastDim"/> copies;
    /// self-attention only.</summary>
    FusedSlice,

    /// <summary>One Linear of a fused <c>[q|k|v]</c> weight, then <see cref="IBackend.QkvSplitNorm"/> splits it and norms Q
    /// and K per head in one op; self-attention with a per-head norm of one epsilon.</summary>
    FusedSplitNorm,
}

/// <summary>The normalization applied to Q and K before attention.</summary>
public enum AttentionQkNorm
{
    /// <summary>None.</summary>
    None,

    /// <summary>RMSNorm over each head's dims: Q, K and V are declared <c>[B, S, heads, headDim]</c>.</summary>
    PerHead,

    /// <summary>RMSNorm over the full inner width, run with the rotation as one backend op (LTX-2): Q, K and V are declared
    /// <c>[.., S, heads·headDim]</c>.</summary>
    FullWidth,

    /// <summary>RMSNorm over the full inner width as its own op, any batch (HiDream): Q, K and V are declared
    /// <c>[.., S, heads·headDim]</c>.</summary>
    FullWidthRms,
}

/// <summary>How the attention output is gated before the output projection.</summary>
public enum AttentionGate
{
    /// <summary>No gate.</summary>
    None,

    /// <summary><c>attn ⊙ sigmoid(W_g·x)</c> over the full width, then the output projection (Krea 2).</summary>
    Sigmoid,

    /// <summary><c>attn ⊙ 2·sigmoid(W_g·x)</c> with one logit per head, applied inside the output projection (LTX-2.3).</summary>
    PerHeadDoubledSigmoid,
}

/// <summary>The fixed geometry and policy of one attention layer, decided at load.</summary>
public sealed record AttentionLayerConfig
{
    /// <summary>Query heads.</summary>
    public required int Heads { get; init; }

    /// <summary>Key/value heads; divides <see cref="Heads"/>.</summary>
    public required int KvHeads { get; init; }

    /// <summary>Dims per head.</summary>
    public required int HeadDim { get; init; }

    /// <summary>How Q, K, V and the gate are projected.</summary>
    public AttentionProjection Projection { get; init; }

    /// <summary>The QK normalization.</summary>
    public AttentionQkNorm QkNorm { get; init; }

    /// <summary>Epsilon of a <see cref="AttentionQkNorm.FullWidth"/> norm; per-head norms carry theirs in the weights.</summary>
    public float FullWidthNormEps { get; init; }

    /// <summary>The output gate.</summary>
    public AttentionGate Gate { get; init; }

    /// <summary>Width of the output projection; 0 means the inner width.</summary>
    public int OutDim { get; init; }

    /// <summary>The dtype Q, K and V are computed in; null follows each projection's input.</summary>
    public DType? ComputeDType { get; init; }

    /// <summary>An F32 key/value stream is widened to an F16 query's dtype before attention (LTX-2's F32 text connectors
    /// feeding an F16 block).</summary>
    public bool WidenKvToQuery { get; init; }

    /// <summary>The token-major entry the layer calls when the backend serves it and the call is one batch item:
    /// <see cref="AttentionRoute.TokenMajor"/>, <see cref="AttentionRoute.TokenMajorGqa"/> (also for equal head counts), or
    /// <see cref="AttentionRoute.Dense"/> to always attend head-major.</summary>
    public AttentionRoute TokenMajorRoute { get; init; } = AttentionRoute.Dense;

    /// <summary>A switch read on every call that can turn the token-major entry off.</summary>
    public Knob<bool>? TokenMajorKnob { get; init; }

    /// <summary>A batch of per-head operands takes the token-major entry too, where the backend plans a row for the whole call;
    /// otherwise only one batch item does.</summary>
    public bool TokenMajorBatches { get; init; }

    /// <summary>A call the CUDA head-major entry serves with SageAttention (F16, head dim 64 or 128, no bias, both sequences at
    /// least <c>numerics.sageF16MinSkv</c>) attends head-major instead of taking the token-major entry, which has no Sage kernel.</summary>
    public bool HeadMajorForSage { get; init; }

    /// <summary>The legacy <c>allowF16</c> assertion the attention call carries.</summary>
    public bool AllowF16 { get; init; }

    /// <summary>Multiply formats the attention call refuses (<see cref="AttentionNumerics.Refused"/>).</summary>
    public AttentionFormats RefusedFormats { get; init; }

    /// <summary>The attention call may store Q, K, P and V in BF16 (<see cref="AttentionNumerics.Bf16Storage"/>); exclusive
    /// with <see cref="AllowF16"/>.</summary>
    public bool AllowBf16 { get; init; }

    /// <summary>The softmax scale; 0 means <c>1/√headDim</c>.</summary>
    public float Scale { get; init; }

    /// <summary>The scale the attention call uses.</summary>
    public float EffectiveScale => Scale != 0f ? Scale : 1.0f / MathF.Sqrt(HeadDim);

    /// <summary>The permissions and refusals the attention call carries.</summary>
    public AttentionNumerics Numerics =>
        (AllowBf16 ? AttentionNumerics.Bf16Storage : AttentionNumerics.FromLegacy(AllowF16)) with { Refused = RefusedFormats };

    /// <summary>Width of the attention output before the output projection.</summary>
    public int InnerDim => Heads * HeadDim;

    /// <summary>Throws when the geometry is inconsistent.</summary>
    public void Validate()
    {
        if (Heads <= 0 || KvHeads <= 0 || HeadDim <= 0)
            throw new ArgumentException($"Attention needs positive heads, kv heads and head dim; got {Heads}, {KvHeads}, {HeadDim}.");
        if (Heads % KvHeads != 0)
            throw new ArgumentException($"Attention heads ({Heads}) must be a multiple of key/value heads ({KvHeads}).");
        if (AllowF16 && AllowBf16)
            throw new ArgumentException("AllowF16 and AllowBf16 are exclusive: with both, the saturating F16 path takes the call.");
        if (TokenMajorRoute is not (AttentionRoute.Dense or AttentionRoute.TokenMajor or AttentionRoute.TokenMajorGqa))
            throw new ArgumentException($"The token-major entry must be Dense, TokenMajor or TokenMajorGqa, not {TokenMajorRoute}.");
        if (TokenMajorRoute == AttentionRoute.TokenMajor && KvHeads != Heads)
            throw new ArgumentException("The equal-head token-major entry cannot serve grouped key/value heads; use TokenMajorGqa.");
        if (QkNorm == AttentionQkNorm.FullWidth && KvHeads != Heads)
            throw new ArgumentException("The full-width QK norm runs with equal query and key/value heads.");
        if (Gate == AttentionGate.PerHeadDoubledSigmoid && Projection != AttentionProjection.GroupedByInput)
            throw new ArgumentException("The per-head gate is projected with the query, grouped by input.");
        if (Projection is AttentionProjection.FusedSlice or AttentionProjection.FusedSplitNorm
            && (KvHeads != Heads || Gate != AttentionGate.None))
            throw new ArgumentException("A fused [q|k|v] projection has equal head counts and no gate.");
        if (Projection == AttentionProjection.FusedSplitNorm && QkNorm != AttentionQkNorm.PerHead)
            throw new ArgumentException("The fused split-norm projection norms Q and K per head.");
    }
}

/// <summary>The weights of one attention layer, borrowed from the model, which loads, slices and folds them.</summary>
public sealed record AttentionLayerWeights
{
    /// <summary>Query projection; null with a fused projection, or in a key/value-only weight set.</summary>
    public Tensor? Query { get; init; }

    /// <summary>Key projection; null with a fused projection.</summary>
    public Tensor? Key { get; init; }

    /// <summary>Value projection; null with a fused projection.</summary>
    public Tensor? Value { get; init; }

    /// <summary>Fused <c>[q|k|v]</c> projection, for <see cref="AttentionProjection.FusedSlice"/> and
    /// <see cref="AttentionProjection.FusedSplitNorm"/>.</summary>
    public Tensor? Qkv { get; init; }

    /// <summary>Fused projection bias.</summary>
    public Tensor? QkvBias { get; init; }

    /// <summary>Output projection; null for a stream whose attention output the model uses unprojected.</summary>
    public Tensor? Output { get; init; }

    /// <summary>Query projection bias.</summary>
    public Tensor? QueryBias { get; init; }

    /// <summary>Key projection bias.</summary>
    public Tensor? KeyBias { get; init; }

    /// <summary>Value projection bias.</summary>
    public Tensor? ValueBias { get; init; }

    /// <summary>Output projection bias.</summary>
    public Tensor? OutputBias { get; init; }

    /// <summary>Gate projection, when the config gates.</summary>
    public Tensor? Gate { get; init; }

    /// <summary>Gate projection bias.</summary>
    public Tensor? GateBias { get; init; }

    /// <summary>Query norm weight (per head or full width).</summary>
    public Tensor? QueryNorm { get; init; }

    /// <summary>Key norm weight (per head or full width).</summary>
    public Tensor? KeyNorm { get; init; }

    /// <summary>Epsilon of a per-head query norm.</summary>
    public float QueryNormEps { get; init; }

    /// <summary>Epsilon of a per-head key norm.</summary>
    public float KeyNormEps { get; init; }
}

/// <summary>The Q and K norm weights of a layer whose projections the model runs itself; a key/value-only layer has no Q norm.</summary>
public readonly record struct AttentionNormWeights(Tensor? Query, Tensor? Key, float QueryEps, float KeyEps);
