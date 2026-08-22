using HartsyInference.Core.Backends;

namespace HartsyInference.LLM.Transformer;

/// <summary>A resolved tensor-parallel plan for one model: one backend per rank plus the collective that sums per-rank partials at the two per-layer reduce seams. Sibling of the layer-split <see cref="LlmPlacement"/> — the two are mutually exclusive (TP replicates the layer STACK on every rank and splits each layer's attention heads/FFN columns; the layer split assigns whole layers to stages). The communicator is borrowed, not owned: whoever created it disposes it.</summary>
public sealed record TpPlacement
{
    /// <summary>Rank-ordered backends; rank r's weight shard and KV slice live on entry r. Rank 0 additionally owns the final norm, the lm_head, and the returned hidden state.</summary>
    public IReadOnlyList<IBackend> RankBackends { get; }

    /// <summary>Collective for the two per-layer AllReduceSum seams (attention-out and down_proj).</summary>
    public ICollectiveComm Comm { get; }

    public TpPlacement(IReadOnlyList<IBackend> rankBackends, ICollectiveComm comm)
    {
        ArgumentNullException.ThrowIfNull(rankBackends);
        ArgumentNullException.ThrowIfNull(comm);
        if (rankBackends.Count < 2)
        {
            throw new ArgumentException(
                $"Tensor parallelism needs at least 2 rank backends (got {rankBackends.Count}) — degree 1 is just the single-device path.",
                nameof(rankBackends));
        }
        foreach (IBackend backend in rankBackends)
        {
            ArgumentNullException.ThrowIfNull(backend, nameof(rankBackends));
        }
        if (comm.RankCount != rankBackends.Count)
        {
            throw new ArgumentException(
                $"Communicator spans {comm.RankCount} ranks but {rankBackends.Count} rank backends were given.",
                nameof(comm));
        }
        RankBackends = rankBackends;
        Comm = comm;
    }

    /// <summary>Tensor-parallel degree (rank count).</summary>
    public int Degree => RankBackends.Count;
}
