using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer.Dsa;

/// <summary>Weights of one DSA attention layer. Projections are <c>[out, in]</c> in any dtype <c>IBackend.Linear</c> takes; the
/// vectors are F32.</summary>
public sealed record DsaAttentionWeights
{
    /// <summary><c>q_a_proj</c>: hidden state to the query latent, <c>[QLoraRank, Dim]</c>.</summary>
    public required Tensor QA { get; init; }

    /// <summary><c>kv_a_proj_with_mqa</c>: hidden state to the key/value latent and the shared rotated key, <c>[KvLoraRank + QkRopeDim, Dim]</c>.</summary>
    public required Tensor KvA { get; init; }

    /// <summary><c>q_a_layernorm</c> (RMS) gain, <c>[QLoraRank]</c>.</summary>
    public required Tensor QANorm { get; init; }

    /// <summary><c>kv_a_layernorm</c> (RMS) gain, <c>[KvLoraRank]</c>.</summary>
    public required Tensor KvANorm { get; init; }

    /// <summary><c>q_b_proj</c>: query latent to <c>Heads * QkHeadDim</c> queries (no-position part first in each head).</summary>
    public required Tensor QB { get; init; }

    /// <summary><c>kv_b_proj</c>: key/value latent to <c>Heads * (QkNopeDim + VDim)</c>, each head's key rows before its value rows.
    /// The layer folds it into per-head absorption matrices once at construction.</summary>
    public required Tensor KvB { get; init; }

    /// <summary><c>wo_b</c>: <c>Heads * VDim</c> to the hidden width.</summary>
    public required Tensor WoB { get; init; }

    /// <summary><c>o_gate</c>: query latent to <c>Heads * VDim</c> sigmoid gate logits; null means no gate.</summary>
    public Tensor? OGate { get; init; }

    /// <summary>Per-head sink logits joining the softmax denominator, F32 <c>[Heads]</c> (ERNIE <c>softmax_offset</c>); null means none.</summary>
    public Tensor? Sink { get; init; }

    /// <summary>Dense head mix <c>[Heads, Heads]</c> applied before the gate (ERNIE VHA, folded by <see cref="Sparse.HeadMix.FoldLowRank"/>); null means none.</summary>
    public Tensor? HeadMix { get; init; }

    /// <summary>The indexer's weights.</summary>
    public required DsaIndexerWeights Indexer { get; init; }
}
