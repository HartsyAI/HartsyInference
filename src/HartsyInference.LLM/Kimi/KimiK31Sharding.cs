using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Dsa;
using HartsyInference.LLM.Transformer.Kda;

namespace HartsyInference.LLM.Kimi;

/// <summary>Rank-local shapes and slices of a Kimi K3.1 decoder. Attention heads (KDA and MLA), the dense and shared FFN intermediates
/// and the vocabulary split across ranks; the routed experts split into contiguous ranges (expert parallelism). The DSA indexer, the latent
/// projections, the router and the norms are replicated on every rank. With one rank the source tensors are used as they are and nothing is
/// owned; with more, every slice and replica is a host F32 copy the caller owns.</summary>
internal static class KimiK31Sharding
{
    /// <summary>Refuses a rank count the decoder's dimensions do not split across.</summary>
    /// <exception cref="ArgumentException">A split dimension is not divisible by <paramref name="ranks"/>.</exception>
    public static void Require(KimiK31Config c, int ranks)
    {
        ArgumentNullException.ThrowIfNull(c);
        ArgumentOutOfRangeException.ThrowIfLessThan(ranks, 1);
        Divisible("Heads (attention heads of the KDA and MLA layers)", c.Heads, ranks);
        Divisible("Vocab (embedding and lm_head rows)", c.Vocab, ranks);
        Divisible("Experts (routed expert ranges)", c.Experts, ranks);
        if (c.DenseLayers > 0) Divisible("DenseIntermediate (dense FFN columns)", c.DenseIntermediate, ranks);
        if (c.SharedExperts > 0) Divisible("SharedExperts * MoeIntermediate (shared expert columns)", c.SharedExperts * c.MoeIntermediate, ranks);
    }

    /// <summary>KDA layer shape of one rank: its share of the heads.</summary>
    public static KdaSettings KdaSettingsFor(KimiK31Config c, int ranks) =>
        new(c.Hidden, c.Heads / ranks, c.KdaHeadDim, c.ConvKernel, c.KdaForgetRank, c.GateLowerBound);

    /// <summary>MLA layer shape of one rank: its share of the heads, the same latent and indexer as every rank.</summary>
    public static DsaAttentionSettings MlaSettingsFor(KimiK31Config c, int ranks) => c.Mla with { Heads = c.Heads / ranks };

    /// <summary>Experts each rank holds.</summary>
    public static int ExpertsPerRank(KimiK31Config c, int ranks) => c.Experts / ranks;

    /// <summary>One rank's KDA weights: q, k and v and their conv taps split by head within each of the three sections; the forget up
    /// projection, its bias, the beta and output gate projections split by head; the output projection split along its input.</summary>
    public static KdaWeights KdaShard(KdaWeights f, KimiK31Config c, int rank, int ranks, List<Tensor> owned)
    {
        if (ranks == 1) return f;
        long key = c.KeyDim, perRank = c.KeyDim / ranks, offset = rank * perRank;
        Tensor qkv = Track(owned, Stack([
            RowRange(f.QkvProj, offset, perRank), RowRange(f.QkvProj, key + offset, perRank), RowRange(f.QkvProj, 2 * key + offset, perRank)]));
        Tensor taps = Track(owned, Stack([
            RowRange(f.ConvTaps, offset, perRank), RowRange(f.ConvTaps, key + offset, perRank), RowRange(f.ConvTaps, 2 * key + offset, perRank)]));
        return new KdaWeights(
            QkvProj: qkv,
            ConvTaps: taps,
            ForgetDown: Replica(f.ForgetDown, owned, ranks),
            ForgetUp: Track(owned, TensorParallelTransformer.SliceOutRows(f.ForgetUp, rank, ranks, "kda.forget_up")),
            ForgetBias: Track(owned, TensorParallelTransformer.SliceOutRows(f.ForgetBias, rank, ranks, "kda.forget_bias")),
            BetaProj: Track(owned, TensorParallelTransformer.SliceOutRows(f.BetaProj, rank, ranks, "kda.beta_proj")),
            OutGateProj: Track(owned, TensorParallelTransformer.SliceOutRows(f.OutGateProj, rank, ranks, "kda.g_proj")),
            OutNorm: Replica(f.OutNorm, owned, ranks),
            OutProj: Track(owned, TensorParallelTransformer.SliceInCols(f.OutProj, rank, ranks, "kda.o_proj")));
    }

    /// <summary>One rank's MLA weights: query and key/value up projections and the output gate split by head; the output projection split
    /// along its input; the latent down projections, norms and the DSA indexer replicated.</summary>
    public static DsaAttentionWeights MlaShard(DsaAttentionWeights f, int rank, int ranks, List<Tensor> owned)
    {
        if (ranks == 1) return f;
        DsaIndexerWeights ix = f.Indexer;
        return new DsaAttentionWeights
        {
            QA = Replica(f.QA, owned, ranks),
            KvA = Replica(f.KvA, owned, ranks),
            QANorm = Replica(f.QANorm, owned, ranks),
            KvANorm = Replica(f.KvANorm, owned, ranks),
            QB = Track(owned, TensorParallelTransformer.SliceOutRows(f.QB, rank, ranks, "mla.q_b_proj")),
            KvB = Track(owned, TensorParallelTransformer.SliceOutRows(f.KvB, rank, ranks, "mla.kv_b_proj")),
            WoB = Track(owned, TensorParallelTransformer.SliceInCols(f.WoB, rank, ranks, "mla.o_proj")),
            OGate = f.OGate is null ? null : Track(owned, TensorParallelTransformer.SliceOutRows(f.OGate, rank, ranks, "mla.g_proj")),
            Indexer = new DsaIndexerWeights
            {
                WqB = Replica(ix.WqB, owned, ranks),
                Wk = Replica(ix.Wk, owned, ranks),
                KeyNormWeight = Replica(ix.KeyNormWeight, owned, ranks),
                KeyNormBias = ix.KeyNormBias is null ? null : Replica(ix.KeyNormBias, owned, ranks),
                WeightsProj = Replica(ix.WeightsProj, owned, ranks),
            },
        };
    }

    /// <summary>A gated FFN's gate, up and down projections split along the intermediate: gate and up by output rows, down by input columns.</summary>
    public static (Tensor Gate, Tensor Up, Tensor Down) FfnShard(Tensor gate, Tensor up, Tensor down, int rank, int ranks, List<Tensor> owned, string name)
    {
        if (ranks == 1) return (gate, up, down);
        return (
            Track(owned, TensorParallelTransformer.SliceOutRows(gate, rank, ranks, $"{name}.gate")),
            Track(owned, TensorParallelTransformer.SliceOutRows(up, rank, ranks, $"{name}.up")),
            Track(owned, TensorParallelTransformer.SliceInCols(down, rank, ranks, $"{name}.down")));
    }

    /// <summary>Rank <paramref name="rank"/>'s rows of a vocabulary-major table: embedding or lm_head.</summary>
    public static Tensor VocabShard(Tensor table, int rank, int ranks, List<Tensor> owned, string name) =>
        ranks == 1 ? table : Track(owned, TensorParallelTransformer.SliceOutRows(table, rank, ranks, name));

    /// <summary>The per-rank MoE block config: the latent SiTU block with its shared expert removed, since the sharded decoder runs the
    /// shared expert as a tensor-parallel FFN beside the routed bank.</summary>
    public static MoeConfig BlockSettings(KimiK31Config c) => KimiK31Model.MoeSettings(c) with { SharedExpertIntermediateSize = 0 };

    /// <summary>A copy of a host F32 tensor, owned by the caller, so each rank holds its own replica. One rank reads the source in place.</summary>
    public static Tensor Replica(Tensor w, List<Tensor> owned, int ranks)
    {
        if (ranks == 1) return w;
        Tensor copy = new(w.Shape, DType.F32);
        w.AsReadOnlySpan<float>().CopyTo(copy.AsSpan<float>());
        return Track(owned, copy);
    }

    /// <summary>Rows <c>[start, start + count)</c> of a host F32 tensor, as a new tensor (the caller owns it).</summary>
    public static Tensor RowRange(Tensor w, long start, long count)
    {
        long rowElements = w.ElementCount / w.Shape[0];
        TensorShape shape = w.Shape.Rank == 1 ? new TensorShape(count) : new TensorShape(count, w.Shape[1]);
        Tensor slice = new(shape, DType.F32);
        w.AsReadOnlySpan<float>().Slice(checked((int)(start * rowElements)), checked((int)(count * rowElements))).CopyTo(slice.AsSpan<float>());
        return slice;
    }

    /// <summary>Stacks the parts' rows into one new tensor; every part shares the column count.</summary>
    public static Tensor Stack(Tensor[] parts)
    {
        long rows = 0;
        foreach (Tensor p in parts) rows += p.Shape[0];
        TensorShape shape = new(rows, parts[0].Shape[1]);
        Tensor all = new(shape, DType.F32);
        Span<float> dst = all.AsSpan<float>();
        int at = 0;
        foreach (Tensor p in parts)
        {
            int count = checked((int)p.ElementCount);
            p.AsReadOnlySpan<float>().CopyTo(dst.Slice(at, count));
            at = checked(at + count);
            p.Dispose();
        }
        return all;
    }

    private static Tensor Track(List<Tensor> owned, Tensor t)
    {
        owned.Add(t);
        return t;
    }

    private static void Divisible(string what, long value, int ranks)
    {
        if (value % ranks != 0)
            throw new ArgumentException($"Kimi K3.1 sharding needs {what} = {value} to divide evenly across {ranks} ranks.", "ranks");
    }
}
