using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Dsa;
using HartsyInference.LLM.Transformer.Kda;
using HartsyInference.LLM.Transformer.Quant;

namespace HartsyInference.LLM.Kimi;

/// <summary>Rank-local shapes and slices of a Kimi K3.1 decoder. Attention heads (KDA and MLA), the dense and shared FFN intermediates
/// and the vocabulary split across ranks; the routed experts split into contiguous ranges (expert parallelism). The DSA indexer, the latent
/// projections, the router and the norms are replicated on every rank. The loader reads each rank's share from the checkpoint directly;
/// <see cref="Slice"/> cuts the same shares out of a whole decoder held in memory.</summary>
public static class KimiK31Sharding
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
    internal static KdaSettings KdaSettingsFor(KimiK31Config c, int ranks) =>
        new(c.Hidden, c.Heads / ranks, c.KdaHeadDim, c.ConvKernel, c.KdaForgetRank, c.GateLowerBound);

    /// <summary>MLA layer shape of one rank: its share of the heads, the same latent and indexer as every rank.</summary>
    internal static DsaAttentionSettings MlaSettingsFor(KimiK31Config c, int ranks) => c.Mla with { Heads = c.Heads / ranks };

    /// <summary>Experts each rank holds.</summary>
    internal static int ExpertsPerRank(KimiK31Config c, int ranks) => c.Experts / ranks;

    /// <summary>The per-rank MoE block config: the latent SiTU block with its shared expert removed, since the sharded decoder runs the
    /// shared expert as a tensor-parallel FFN beside the routed bank.</summary>
    internal static MoeConfig BlockSettings(KimiK31Config c) => KimiK31Model.MoeSettings(c) with { SharedExpertIntermediateSize = 0 };

    /// <summary>Every rank's weights cut from a whole decoder held in memory. The tensors the cut creates are in <see cref="RankWeightSet.Owned"/>;
    /// the whole decoder's tensors are referenced, not copied. One rank returns the whole decoder unchanged.</summary>
    internal static RankWeightSet Slice(KimiK31Config c, KimiK31Weights full, int ranks)
    {
        ArgumentNullException.ThrowIfNull(full);
        Require(c, ranks);
        if (ranks == 1) return new RankWeightSet([full], []);
        List<Tensor> owned = [];
        List<KimiK31Weights> parts = new(ranks);
        for (int r = 0; r < ranks; r++) parts.Add(RankOf(c, full, r, ranks, owned));
        return new RankWeightSet(parts, owned);
    }

    /// <summary>KDA weights of one rank: q, k and v and their conv taps split by head within each of the three sections; the forget up
    /// projection, its bias, the beta and output gate projections split by head; the output projection split along its input.</summary>
    internal static KdaWeights KdaShard(KdaWeights f, KimiK31Config c, int rank, int ranks, List<Tensor> owned)
    {
        long key = c.KeyDim, perRank = c.KeyDim / ranks, offset = rank * perRank;
        Tensor qkv = Track(owned, Stack([
            RowRange(f.QkvProj, offset, perRank), RowRange(f.QkvProj, key + offset, perRank), RowRange(f.QkvProj, 2 * key + offset, perRank)]));
        Tensor taps = Track(owned, Stack([
            RowRange(f.ConvTaps, offset, perRank), RowRange(f.ConvTaps, key + offset, perRank), RowRange(f.ConvTaps, 2 * key + offset, perRank)]));
        return new KdaWeights(
            QkvProj: qkv,
            ConvTaps: taps,
            ForgetDown: Replica(f.ForgetDown, owned),
            ForgetUp: Track(owned, TensorParallelTransformer.SliceOutRows(f.ForgetUp, rank, ranks, "kda.forget_up")),
            ForgetBias: Track(owned, TensorParallelTransformer.SliceOutRows(f.ForgetBias, rank, ranks, "kda.forget_bias")),
            BetaProj: Track(owned, TensorParallelTransformer.SliceOutRows(f.BetaProj, rank, ranks, "kda.beta_proj")),
            OutGateProj: Track(owned, TensorParallelTransformer.SliceOutRows(f.OutGateProj, rank, ranks, "kda.g_proj")),
            OutNorm: Replica(f.OutNorm, owned),
            OutProj: Track(owned, TensorParallelTransformer.SliceInCols(f.OutProj, rank, ranks, "kda.o_proj")));
    }

    /// <summary>MLA weights of one rank: query and key/value up projections and the output gate split by head; the output projection split
    /// along its input; the latent down projections, norms and the DSA indexer replicated.</summary>
    internal static DsaAttentionWeights MlaShard(DsaAttentionWeights f, int rank, int ranks, List<Tensor> owned)
    {
        DsaIndexerWeights ix = f.Indexer;
        return new DsaAttentionWeights
        {
            QA = Replica(f.QA, owned),
            KvA = Replica(f.KvA, owned),
            QANorm = Replica(f.QANorm, owned),
            KvANorm = Replica(f.KvANorm, owned),
            QB = Track(owned, TensorParallelTransformer.SliceOutRows(f.QB, rank, ranks, "mla.q_b_proj")),
            KvB = Track(owned, TensorParallelTransformer.SliceOutRows(f.KvB, rank, ranks, "mla.kv_b_proj")),
            WoB = Track(owned, TensorParallelTransformer.SliceInCols(f.WoB, rank, ranks, "mla.o_proj")),
            OGate = f.OGate is null ? null : Track(owned, TensorParallelTransformer.SliceOutRows(f.OGate, rank, ranks, "mla.g_proj")),
            Indexer = new DsaIndexerWeights
            {
                WqB = Replica(ix.WqB, owned),
                Wk = Replica(ix.Wk, owned),
                KeyNormWeight = Replica(ix.KeyNormWeight, owned),
                KeyNormBias = ix.KeyNormBias is null ? null : Replica(ix.KeyNormBias, owned),
                WeightsProj = Replica(ix.WeightsProj, owned),
            },
        };
    }

    /// <summary>A gated FFN's gate, up and down projections split along the intermediate: gate and up by output rows, down by input columns.</summary>
    internal static (Tensor Gate, Tensor Up, Tensor Down) FfnShard(Tensor gate, Tensor up, Tensor down, int rank, int ranks, List<Tensor> owned, string name)
    {
        return (
            Track(owned, TensorParallelTransformer.SliceOutRows(gate, rank, ranks, $"{name}.gate")),
            Track(owned, TensorParallelTransformer.SliceOutRows(up, rank, ranks, $"{name}.up")),
            Track(owned, TensorParallelTransformer.SliceInCols(down, rank, ranks, $"{name}.down")));
    }

    /// <summary>A copy of a host F32 tensor, owned by the caller, so each rank holds its own replica.</summary>
    internal static Tensor Replica(Tensor w, List<Tensor> owned)
    {
        Tensor copy = new(w.Shape, DType.F32);
        w.AsReadOnlySpan<float>().CopyTo(copy.AsSpan<float>());
        return Track(owned, copy);
    }

    /// <summary>Rows <c>[start, start + count)</c> of a host F32 tensor, as a new tensor (the caller owns it).</summary>
    internal static Tensor RowRange(Tensor w, long start, long count)
    {
        long rowElements = w.ElementCount / w.Shape[0];
        TensorShape shape = w.Shape.Rank == 1 ? new TensorShape(count) : new TensorShape(count, w.Shape[1]);
        Tensor slice = new(shape, DType.F32);
        w.AsReadOnlySpan<float>().Slice((int)(start * rowElements), (int)(count * rowElements)).CopyTo(slice.AsSpan<float>());
        return slice;
    }

    /// <summary>Stacks the parts' rows into one new tensor; every part shares the column count.</summary>
    internal static Tensor Stack(Tensor[] parts)
    {
        long rows = 0;
        foreach (Tensor p in parts) rows += p.Shape[0];
        TensorShape shape = new(rows, parts[0].Shape[1]);
        Tensor all = new(shape, DType.F32);
        Span<float> dst = all.AsSpan<float>();
        int at = 0;
        foreach (Tensor p in parts)
        {
            p.AsReadOnlySpan<float>().CopyTo(dst.Slice(at, (int)p.ElementCount));
            at += (int)p.ElementCount;
            p.Dispose();
        }
        return all;
    }

    // Rank r's share of a whole decoder: the vocabulary, every layer's attention and FFN shares, and the routed experts of its range.
    private static KimiK31Weights RankOf(KimiK31Config c, KimiK31Weights f, int rank, int ranks, List<Tensor> owned)
    {
        List<KimiK31LayerWeights> layers = new(c.Layers);
        for (int i = 0; i < c.Layers; i++) layers.Add(LayerOf(c, f.Layers[i], i, rank, ranks, owned));
        return new KimiK31Weights
        {
            Embedding = VocabShard(f.Embedding, rank, ranks, owned, "embed_tokens"),
            FinalNorm = f.FinalNorm,
            LmHead = VocabShard(f.LmHead, rank, ranks, owned, "lm_head"),
            OutputResQuery = f.OutputResQuery,
            OutputResNorm = f.OutputResNorm,
            Layers = layers,
        };
    }

    private static KimiK31LayerWeights LayerOf(KimiK31Config c, KimiK31LayerWeights lw, int i, int rank, int ranks, List<Tensor> owned)
    {
        Tensor? gate = null, up = null, down = null;
        if (lw.DenseGate is not null)
        {
            (gate, up, down) = FfnShard(lw.DenseGate, lw.DenseUp!, lw.DenseDown!, rank, ranks, owned, $"layer{i}.dense");
        }
        return new KimiK31LayerWeights
        {
            InputNorm = lw.InputNorm,
            PostNorm = lw.PostNorm,
            AttnResAttnQuery = lw.AttnResAttnQuery,
            AttnResAttnNorm = lw.AttnResAttnNorm,
            AttnResMlpQuery = lw.AttnResMlpQuery,
            AttnResMlpNorm = lw.AttnResMlpNorm,
            Kda = lw.Kda is null ? null : KdaShard(lw.Kda, c, rank, ranks, owned),
            Mla = lw.Mla is null ? null : MlaShard(lw.Mla, rank, ranks, owned),
            DenseGate = gate,
            DenseUp = up,
            DenseDown = down,
            Moe = lw.Moe is null ? null : MoeOf(c, lw, i, rank, ranks, owned),
            PackedExperts = lw.PackedExperts is null ? null : OwnedExperts(c, lw.PackedExperts, rank, ranks),
            MoePrefix = lw.MoePrefix,
        };
    }

    // Rank r's routed block: the router, latent projections and norms as the rank's own replicas; the shared expert's shares; the experts of
    // its range by reference.
    private static Dictionary<string, Tensor> MoeOf(KimiK31Config c, KimiK31LayerWeights lw, int i, int rank, int ranks, List<Tensor> owned)
    {
        IReadOnlyDictionary<string, Tensor> whole = lw.Moe!;
        string mlp = lw.MoePrefix + ".mlp";
        Dictionary<string, Tensor> block = new(StringComparer.Ordinal);
        foreach (string key in new[] { "gate.weight", "gate.e_score_correction_bias", "pre_expert_proj.weight", "pre_expert_norm.weight", "post_expert_proj.weight" })
        {
            if (whole.TryGetValue($"{mlp}.{key}", out Tensor? replicated)) block[$"{mlp}.{key}"] = Replica(replicated, owned);
        }
        if (c.SharedExperts > 0)
        {
            string shared = $"{mlp}.shared_expert.";
            (Tensor g, Tensor u, Tensor d) = FfnShard(
                whole[shared + "gate_proj.weight"], whole[shared + "up_proj.weight"], whole[shared + "down_proj.weight"], rank, ranks, owned, $"layer{i}.shared");
            block[shared + "gate_proj.weight"] = g;
            block[shared + "up_proj.weight"] = u;
            block[shared + "down_proj.weight"] = d;
        }
        int perRank = ExpertsPerRank(c, ranks), first = rank * perRank;
        for (int e = first; e < first + perRank; e++)
        {
            foreach (string proj in new[] { "gate_proj", "up_proj", "down_proj" })
            {
                string key = $"{mlp}.experts.{e}.{proj}.weight";
                if (whole.TryGetValue(key, out Tensor? expert)) block[key] = expert;
            }
        }
        return block;
    }

    private static IReadOnlyList<PackedMxfp4Expert> OwnedExperts(KimiK31Config c, IReadOnlyList<PackedMxfp4Expert> all, int rank, int ranks)
    {
        int perRank = ExpertsPerRank(c, ranks), first = rank * perRank;
        List<PackedMxfp4Expert> local = new(perRank);
        for (int e = first; e < first + perRank; e++) local.Add(all[e]);
        return local;
    }

    private static Tensor VocabShard(Tensor table, int rank, int ranks, List<Tensor> owned, string name) =>
        Track(owned, TensorParallelTransformer.SliceOutRows(table, rank, ranks, name));

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

    /// <summary>The rank weights of one decoder, and the tensors the cut created for them.</summary>
    internal sealed record RankWeightSet(IReadOnlyList<KimiK31Weights> Ranks, IReadOnlyList<Tensor> Owned);
}
