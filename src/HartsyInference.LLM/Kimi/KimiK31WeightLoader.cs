using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Dsa;
using HartsyInference.LLM.Transformer.Kda;
using HartsyInference.LLM.Transformer.Quant;

namespace HartsyInference.LLM.Kimi;

/// <summary>Maps a Kimi K3.1 checkpoint's tensor names and shapes (verified against the published index) onto <see cref="KimiK31Weights"/>.
/// KDA's q, k and v are fused and its forget gate is folded (see <see cref="KdaLayer.FoldForgetGate"/>), and the MLA output gate reads the
/// hidden state. Routed experts load as MXFP4-packed weights when the checkpoint stores them packed, and as F32 otherwise.</summary>
/// <remarks>A rank loads only its share: the rows or columns its heads, vocabulary and FFN intermediates occupy (read as byte ranges), and
/// the routed experts of its expert range. Replicated tensors are read whole. Latent projections, the router and the norms follow the
/// dense-weight policy and stay replicated. Dense Linear weights go through <c>denseWeight</c>, which picks the dtype the rank's backend
/// runs. The tensors the loader creates are listed in <see cref="KimiK31Weights.Owned"/>.</remarks>
public sealed class KimiK31WeightLoader
{
    private const string Root = "language_model.model.";

    private readonly KimiK31Config _c;
    private readonly IKimiK31WeightSource _src;
    private readonly KimiK31Shard _shard;
    private readonly Func<Tensor, string, Tensor> _dense;
    private readonly List<Tensor> _owned = [];

    private KimiK31WeightLoader(KimiK31Config c, IKimiK31WeightSource src, KimiK31Shard shard, Func<Tensor, string, Tensor> dense)
    {
        _c = c;
        _src = src;
        _shard = shard;
        _dense = dense;
    }

    /// <summary>Loads every tensor of the whole decoder as F32. The result owns the tensors the loader created (<see cref="KimiK31Weights.Owned"/>).</summary>
    public static KimiK31Weights Load(KimiK31Config c, IKimiK31WeightSource src) => Load(c, src, KimiK31Shard.Whole, (t, _) => TensorCasts.EnsureF32(t));

    /// <summary>Loads the part of the decoder that <paramref name="shard"/> holds.</summary>
    /// <param name="denseWeight">Returns a dense Linear weight in the dtype its backend runs. A returned tensor other than the argument is
    /// owned by the loader; the argument is disposed by the loader.</param>
    /// <exception cref="ArgumentException">A split load needs a source that reads ranges, or the config does not split across the ranks.</exception>
    /// <exception cref="InvalidDataException">A tensor's shape does not match the config.</exception>
    public static KimiK31Weights Load(KimiK31Config c, IKimiK31WeightSource src, KimiK31Shard shard, Func<Tensor, string, Tensor> denseWeight)
    {
        ArgumentNullException.ThrowIfNull(c);
        ArgumentNullException.ThrowIfNull(src);
        ArgumentNullException.ThrowIfNull(denseWeight);
        if (shard.Ranks > 1 && src is not IKimiK31RankWeightSource)
            throw new ArgumentException($"A {shard.Ranks}-rank load needs a source that reads ranges (IKimiK31RankWeightSource).", nameof(src));
        KimiK31Sharding.Require(c, shard.Ranks);
        return new KimiK31WeightLoader(c, src, shard, denseWeight).LoadAll();
    }

    private KimiK31Weights LoadAll()
    {
        try
        {
            List<KimiK31LayerWeights> layers = new(_c.Layers);
            for (int i = 0; i < _c.Layers; i++) layers.Add(LoadLayer(i));
            return new KimiK31Weights
            {
                Embedding = F32Part(Root + "embed_tokens.weight", 0, _c.Vocab, _c.Hidden),
                FinalNorm = F32Whole(Root + "norm.weight", _c.Hidden),
                LmHead = DensePart("language_model.lm_head.weight", 0, _c.Vocab, _c.Hidden),
                OutputResQuery = Column(Root + "output_attn_res_proj.weight", _c.Hidden),
                OutputResNorm = F32Whole(Root + "output_attn_res_norm.weight", _c.Hidden),
                Layers = layers,
                Owned = _owned,
            };
        }
        catch
        {
            // A failed load releases what it already created; the caller never sees the partial weights.
            foreach (Tensor t in _owned) t.Dispose();
            _owned.Clear();
            throw;
        }
    }

    private KimiK31LayerWeights LoadLayer(int i)
    {
        string p = Layer(i);
        bool mla = _c.IsMla(i), dense = _c.IsDense(i);
        return new KimiK31LayerWeights
        {
            InputNorm = F32Whole(p + "input_layernorm.weight", _c.Hidden),
            PostNorm = F32Whole(p + "post_attention_layernorm.weight", _c.Hidden),
            AttnResAttnQuery = Column(p + "self_attention_res_proj.weight", _c.Hidden),
            AttnResAttnNorm = F32Whole(p + "self_attention_res_norm.weight", _c.Hidden),
            AttnResMlpQuery = Column(p + "mlp_res_proj.weight", _c.Hidden),
            AttnResMlpNorm = F32Whole(p + "mlp_res_norm.weight", _c.Hidden),
            Kda = mla ? null : LoadKda(i),
            Mla = mla ? LoadMla(i) : null,
            DenseGate = dense ? DensePart(p + "mlp.gate_proj.weight", 0, _c.DenseIntermediate, _c.Hidden) : null,
            DenseUp = dense ? DensePart(p + "mlp.up_proj.weight", 0, _c.DenseIntermediate, _c.Hidden) : null,
            DenseDown = dense ? DensePart(p + "mlp.down_proj.weight", 1, _c.Hidden, _c.DenseIntermediate) : null,
            Moe = dense ? null : LoadMoe(i),
            PackedExperts = dense ? null : LoadPackedExperts(i),
            MoePrefix = Prefix(i),
        };
    }

    private KdaWeights LoadKda(int i)
    {
        string p = Layer(i) + "self_attn.";
        int key = _c.KeyDim, h = _c.Hidden, k = _c.ConvKernel, rank = _c.KdaForgetRank;
        int keyRank = key / _shard.Ranks, firstChannel = _shard.Rank * keyRank;

        // q, k and v stacked in the order the KDA conv splits them. The depthwise taps are F32 [3 * keyRank, k], the rank-2 shape KdaWeights takes.
        Tensor qkv = Own(ConcatRows(Part(p + "q_proj.weight", 0, key, h), Part(p + "k_proj.weight", 0, key, h), Part(p + "v_proj.weight", 0, key, h)),
            p + "qkv_proj.weight");
        float[] taps =
        [
            .. HostPart(p + "q_conv1d.weight", 0, key, 1, k), .. HostPart(p + "k_conv1d.weight", 0, key, 1, k), .. HostPart(p + "v_conv1d.weight", 0, key, 1, k),
        ];

        // A_log holds one value per head; every key channel of a head takes that head's value. The fold is per channel, so this
        // rank folds only its own channels.
        float[] aLogHead = HostPerHead(p + "A_log", _c.Heads, _c.KdaHeadDim);
        float[] aLog = new float[keyRank];
        for (int ch = 0; ch < keyRank; ch++) aLog[ch] = aLogHead[(firstChannel + ch) / _c.KdaHeadDim];
        (float[] up, float[] bias) = KdaLayer.FoldForgetGate(
            HostPart(p + "f_b_proj.weight", 0, key, rank), aLog, HostPart(p + "dt_bias", 0, key), keyRank, rank);

        return new KdaWeights(
            QkvProj: qkv,
            ConvTaps: Make(taps, 3L * keyRank, k),
            ForgetDown: DensePart(p + "f_a_proj.weight", -1, rank, h),
            ForgetUp: Make(up, keyRank, rank),
            ForgetBias: Make(bias, keyRank),
            BetaProj: DensePart(p + "b_proj.weight", 0, _c.Heads, h),
            OutGateProj: DensePart(p + "g_proj.weight", 0, key, h),
            OutNorm: F32Whole(p + "o_norm.weight", _c.KdaHeadDim),
            OutProj: DensePart(p + "o_proj.weight", 1, h, key));
    }

    private DsaAttentionWeights LoadMla(int i)
    {
        string p = Layer(i) + "self_attn.";
        int h = _c.Hidden, qk = _c.QkNopeDim + _c.QkRopeDim, heads = _c.Heads;
        return new DsaAttentionWeights
        {
            QA = DensePart(p + "q_a_proj.weight", -1, _c.QLoraRank, h),
            KvA = DensePart(p + "kv_a_proj_with_mqa.weight", -1, _c.KvLoraRank + _c.QkRopeDim, h),
            QANorm = F32Whole(p + "q_a_layernorm.weight", _c.QLoraRank),
            KvANorm = F32Whole(p + "kv_a_layernorm.weight", _c.KvLoraRank),
            QB = DensePart(p + "q_b_proj.weight", 0, heads * qk, _c.QLoraRank),
            KvB = DensePart(p + "kv_b_proj.weight", 0, heads * (_c.QkNopeDim + _c.VDim), _c.KvLoraRank),
            WoB = DensePart(p + "o_proj.weight", 1, h, heads * _c.VDim),
            OGate = DensePart(p + "g_proj.weight", 0, heads * _c.VDim, h),
            Indexer = new DsaIndexerWeights
            {
                WqB = DensePart(p + "indexer.wq_b.weight", -1, _c.IndexHeads * _c.IndexHeadDim, _c.QLoraRank),
                Wk = DensePart(p + "indexer.wk.weight", -1, _c.IndexHeadDim, h),
                KeyNormWeight = F32Whole(p + "indexer.k_norm.weight", _c.IndexHeadDim),
                KeyNormBias = F32Whole(p + "indexer.k_norm.bias", _c.IndexHeadDim),
                WeightsProj = DensePart(p + "indexer.weights_proj.weight", -1, _c.IndexHeads, h),
            },
        };
    }

    // Engine-side names as MoeFeedForward.LoadWeights reads them, under the layer's prefix. The router stays F32: the MoE gate reads it on the host.
    private Dictionary<string, Tensor> LoadMoe(int i)
    {
        string block = Layer(i) + "block_sparse_moe.", mlp = Prefix(i) + ".mlp";
        int h = _c.Hidden, ls = _c.LatentSize, mi = _c.MoeIntermediate, sh = _c.SharedExperts * _c.MoeIntermediate;
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal)
        {
            [$"{mlp}.gate.weight"] = F32Whole(block + "gate.weight", _c.Experts, h),
            [$"{mlp}.gate.e_score_correction_bias"] = F32Whole(block + "gate.e_score_correction_bias", _c.Experts),
            [$"{mlp}.pre_expert_proj.weight"] = DensePart(block + "routed_expert_down_proj.weight", -1, ls, h),
            [$"{mlp}.pre_expert_norm.weight"] = F32Whole(block + "routed_expert_norm.weight", ls),
            [$"{mlp}.post_expert_proj.weight"] = DensePart(block + "routed_expert_up_proj.weight", -1, h, ls),
        };
        if (_c.SharedExperts > 0)
        {
            w[$"{mlp}.shared_expert.gate_proj.weight"] = DensePart(block + "shared_experts.gate_proj.weight", 0, sh, h);
            w[$"{mlp}.shared_expert.up_proj.weight"] = DensePart(block + "shared_experts.up_proj.weight", 0, sh, h);
            w[$"{mlp}.shared_expert.down_proj.weight"] = DensePart(block + "shared_experts.down_proj.weight", 1, h, sh);
        }
        // A packed checkpoint's experts load through LoadPackedExperts; F32 experts are read here, this rank's range only.
        if (!_src.Has($"{block}experts.0.w1.weight_packed"))
        {
            int perRank = KimiK31Sharding.ExpertsPerRank(_c, _shard.Ranks), first = _shard.Rank * perRank;
            for (int e = first; e < first + perRank; e++)
            {
                string x = $"{block}experts.{e}";
                w[$"{mlp}.experts.{e}.gate_proj.weight"] = DensePart($"{x}.w1.weight", -1, mi, ls);
                w[$"{mlp}.experts.{e}.up_proj.weight"] = DensePart($"{x}.w3.weight", -1, mi, ls);
                w[$"{mlp}.experts.{e}.down_proj.weight"] = DensePart($"{x}.w2.weight", -1, ls, mi);
            }
        }
        return w;
    }

    // The MXFP4-packed routed experts of layer i in this rank's range: w1 is the gate, w3 the up and w2 the down projection, each a U8 weight
    // with its E8M0 scales. Null when the checkpoint stores F32 experts. Packed bytes stay as the shard maps them; nothing is widened.
    private IReadOnlyList<PackedMxfp4Expert>? LoadPackedExperts(int i)
    {
        string block = Layer(i) + "block_sparse_moe.";
        if (!_src.Has($"{block}experts.0.w1.weight_packed")) return null;
        int ls = _c.LatentSize, mi = _c.MoeIntermediate, perRank = KimiK31Sharding.ExpertsPerRank(_c, _shard.Ranks), first = _shard.Rank * perRank;
        List<PackedMxfp4Expert> experts = new(perRank);
        for (int e = first; e < first + perRank; e++)
        {
            string x = $"{block}experts.{e}";
            experts.Add(new PackedMxfp4Expert(
                Projection(x, "w1", mi, ls),
                Projection(x, "w3", mi, ls),
                Projection(x, "w2", ls, mi)));
        }
        return experts;
    }

    private PackedMxfp4Projection Projection(string expert, string proj, int rows, int cols)
    {
        // The packed model keeps these views for the decoder's lifetime; the loader's owned list releases each one as it is read, so a
        // failure on the second read still releases the first.
        Tensor packed = Track(_src.Read($"{expert}.{proj}.weight_packed"));
        Tensor scale = Track(_src.Read($"{expert}.{proj}.weight_scale"));
        Expect(packed, $"{expert}.{proj}.weight_packed", DType.U8, rows, cols / 2);
        Expect(scale, $"{expert}.{proj}.weight_scale", DType.U8, rows, cols / 32);
        return new PackedMxfp4Projection(packed, new QuantRecipe
        {
            Encoding = QuantEncoding.Mxfp4E8M0, Geometry = new BlockGeometry(1, 32), ScaleDType = DType.U8,
            LogicalRows = rows, LogicalCols = cols, Scale = scale,
        });
    }

    // This rank's part of a checkpoint tensor whose whole shape is dims: its rows (dim 0) or columns (dim 1), or the whole tensor when dim
    // is -1 or the decoder runs on one rank. The result is in the stored dtype and is not tracked; the caller disposes it.
    private Tensor Part(string name, int dim, params long[] dims)
    {
        bool whole = dim < 0 || _shard.Ranks == 1;
        Tensor t;
        long[] want = (long[])dims.Clone();
        if (whole)
        {
            t = _src.Read(name);
        }
        else
        {
            long count = dims[dim] / _shard.Ranks;
            want[dim] = count;
            t = ((IKimiK31RankWeightSource)_src).ReadSlice(name, dim, _shard.Rank * count, count);
        }
        if (!HasShape(t.Shape, want))
        {
            t.Dispose();
            throw Mismatch(name, t.Shape, want);
        }
        return t;
    }

    private static bool HasShape(TensorShape shape, long[] want)
    {
        if (shape.Rank != want.Length) return false;
        for (int d = 0; d < want.Length; d++)
            if (shape[d] != want[d]) return false;
        return true;
    }

    // The rank's part as the dtype its backend runs (dense Linear weights).
    private Tensor DensePart(string name, int dim, params long[] dims) => Own(Part(name, dim, dims), name);

    private Tensor F32Part(string name, int dim, params long[] dims) => Track(ToF32(Part(name, dim, dims)));

    private Tensor F32Whole(string name, params long[] dims) => F32Part(name, -1, dims);

    // Applies the dense policy to an untracked part, disposing the part when the policy returned a copy.
    private Tensor Own(Tensor part, string name)
    {
        Tensor w = _dense(part, name);
        if (!ReferenceEquals(w, part)) part.Dispose();
        return Track(w);
    }

    private static Tensor ToF32(Tensor t)
    {
        Tensor f = TensorCasts.EnsureF32(t);
        if (!ReferenceEquals(f, t)) t.Dispose();
        return f;
    }

    private float[] HostPart(string name, int dim, params long[] dims)
    {
        Tensor f = ToF32(Part(name, dim, dims));
        try
        {
            return f.AsReadOnlySpan<float>().ToArray();
        }
        finally
        {
            f.Dispose();
        }
    }

    // The first n values of a vector. A published vector may carry trailing padding past its heads; only the leading n are read.
    private float[] HostLeading(string name, int n)
    {
        Tensor f = ToF32(_src.Read(name));
        try
        {
            if (f.Shape.Rank != 1 || f.Shape[0] < n)
                throw new InvalidDataException($"Tensor '{name}' must be a vector of at least {n} values, got {f.Shape}.");
            return f.AsReadOnlySpan<float>().Slice(0, n).ToArray();
        }
        finally
        {
            f.Dispose();
        }
    }

    // One value per head. Exactly `heads` values, or the stored `headDim` form whose entries past `heads` must all be zero.
    private float[] HostPerHead(string name, int heads, int headDim)
    {
        Tensor f = ToF32(_src.Read(name));
        try
        {
            long len = f.Shape.Rank == 1 ? f.Shape[0] : -1;
            bool padded = len == headDim && headDim > heads;
            if (len != heads && !padded)
                throw new InvalidDataException(
                    $"Tensor '{name}' has shape {f.Shape}; A_log is read as one value per head, [{heads}], or stored padded to [{headDim}] "
                    + "with zeros past the heads. Any other shape is refused.");
            float[] values = f.AsReadOnlySpan<float>().ToArray();
            for (int i = heads; i < values.Length; i++)
                if (values[i] != 0f)
                    throw new InvalidDataException(
                        $"Tensor '{name}' has shape {f.Shape}, but padding entry {i} is {values[i]}, not zero. The padded form is accepted only "
                        + "with a zero tail (docs/Research/KIMI_K31_ARCHITECTURE.md, A_log).");
            return values[..heads];
        }
        finally
        {
            f.Dispose();
        }
    }

    // A [1, n] pseudo-query becomes the [n, 1] column the attention residual takes.
    private Tensor Column(string name, long n) => Make(HostPart(name, -1, 1L, n), n, 1L);

    private Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return Track(t);
    }

    // The parts' rows as one new tensor, copied byte for byte. The parts share a dtype and their trailing dimensions. The caller owns the
    // result; the parts are disposed.
    private static unsafe Tensor ConcatRows(params Tensor[] parts)
    {
        long rows = 0;
        foreach (Tensor p in parts)
        {
            if (p.DType != parts[0].DType || p.Shape.Rank != parts[0].Shape.Rank)
                throw new InvalidDataException("Row parts must share a dtype and rank.");
            rows += p.Shape[0];
        }
        long[] dims = new long[parts[0].Shape.Rank];
        for (int d = 0; d < dims.Length; d++) dims[d] = parts[0].Shape[d];
        dims[0] = rows;
        Tensor all = new(new TensorShape(dims), parts[0].DType);
        long at = 0;
        foreach (Tensor p in parts)
        {
            long bytes = p.DType.ComputeByteCount(p.ElementCount);
            new ReadOnlySpan<byte>(p.DataPointer, checked((int)bytes)).CopyTo(new Span<byte>((byte*)all.DataPointer + at, checked((int)bytes)));
            at += bytes;
            p.Dispose();
        }
        return all;
    }

    private Tensor Track(Tensor t)
    {
        _owned.Add(t);
        return t;
    }

    private static void Expect(Tensor t, string name, DType dtype, long rows, long cols)
    {
        if (t.DType != dtype || t.Shape.Rank != 2 || t.Shape[0] != rows || t.Shape[1] != cols)
            throw new InvalidDataException($"Tensor '{name}' must be {dtype} [{rows}, {cols}], got {t.DType} {t.Shape}.");
    }

    private static string Layer(int i) => $"{Root}layers.{i}.";

    private static string Prefix(int i) => $"L{i}";

    private static InvalidDataException Mismatch(string name, TensorShape got, long[] want) =>
        new($"Tensor '{name}' has shape {got}, the config needs [{string.Join(", ", want)}] on this rank.");
}
