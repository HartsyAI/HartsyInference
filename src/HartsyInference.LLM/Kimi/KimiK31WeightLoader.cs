using HartsyInference.Core.Tensors.Quant;
using HartsyInference.LLM.Transformer.Quant;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Dsa;
using HartsyInference.LLM.Transformer.Kda;

namespace HartsyInference.LLM.Kimi;

/// <summary>Where <see cref="KimiK31WeightLoader"/> reads a Kimi K3.1 checkpoint's tensors: safetensors on disk, or any store with the
/// checkpoint's names.</summary>
public interface IKimiK31WeightSource
{
    /// <summary>Whether the source holds a tensor called <paramref name="name"/>.</summary>
    bool Has(string name);

    /// <summary>The tensor as the caller disposes it, in any dtype; the loader converts it to F32.</summary>
    /// <exception cref="KeyNotFoundException">The source has no tensor <paramref name="name"/>.</exception>
    Tensor Read(string name);
}

/// <summary>Maps a Kimi K3.1 checkpoint's tensor names and shapes (verified against the published index) onto
/// <see cref="KimiK31Weights"/>. KDA's q, k and v are fused and its forget gate is folded (see <see cref="KdaLayer.FoldForgetGate"/>), and
/// the MLA output gate reads the hidden state. Routed experts load as MXFP4-packed weights when the checkpoint stores them packed, and as F32
/// otherwise. The tensors the loader creates are listed in <see cref="KimiK31Weights.Owned"/>.</summary>
public sealed class KimiK31WeightLoader
{
    private const string Root = "language_model.model.";

    /// <summary>Loads every tensor of the decoder. The result owns the tensors the loader created (<see cref="KimiK31Weights.Owned"/>); dispose them with the model.</summary>
    public static KimiK31Weights Load(KimiK31Config c, IKimiK31WeightSource src) => new KimiK31WeightLoader().LoadAll(c, src);

    private readonly List<Tensor> _owned = new();

    /// <summary>Loads every tensor of the decoder.</summary>
    /// <exception cref="InvalidDataException">A tensor's shape does not match the config.</exception>
    private KimiK31Weights LoadAll(KimiK31Config c, IKimiK31WeightSource src)
    {
        ArgumentNullException.ThrowIfNull(c);
        ArgumentNullException.ThrowIfNull(src);
        try
        {
        List<KimiK31LayerWeights> layers = new(c.Layers);
        for (int i = 0; i < c.Layers; i++) layers.Add(LoadLayer(c, i, src));
        return new KimiK31Weights
        {
            Embedding = F32(src, Root + "embed_tokens.weight", c.Vocab, c.Hidden),
            FinalNorm = F32(src, Root + "norm.weight", c.Hidden),
            LmHead = F32(src, "language_model.lm_head.weight", c.Vocab, c.Hidden),
            OutputResQuery = Column(src, Root + "output_attn_res_proj.weight", c.Hidden),
            OutputResNorm = F32(src, Root + "output_attn_res_norm.weight", c.Hidden),
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

    private KimiK31LayerWeights LoadLayer(KimiK31Config c, int i, IKimiK31WeightSource src)
    {
        string p = Layer(i);
        bool mla = c.IsMla(i), dense = c.IsDense(i);
        return new KimiK31LayerWeights
        {
            InputNorm = F32(src, p + "input_layernorm.weight", c.Hidden),
            PostNorm = F32(src, p + "post_attention_layernorm.weight", c.Hidden),
            AttnResAttnQuery = Column(src, p + "self_attention_res_proj.weight", c.Hidden),
            AttnResAttnNorm = F32(src, p + "self_attention_res_norm.weight", c.Hidden),
            AttnResMlpQuery = Column(src, p + "mlp_res_proj.weight", c.Hidden),
            AttnResMlpNorm = F32(src, p + "mlp_res_norm.weight", c.Hidden),
            Kda = mla ? null : LoadKda(c, i, src),
            Mla = mla ? LoadMla(c, i, src) : null,
            DenseGate = dense ? F32(src, p + "mlp.gate_proj.weight", c.DenseIntermediate, c.Hidden) : null,
            DenseUp = dense ? F32(src, p + "mlp.up_proj.weight", c.DenseIntermediate, c.Hidden) : null,
            DenseDown = dense ? F32(src, p + "mlp.down_proj.weight", c.Hidden, c.DenseIntermediate) : null,
            Moe = dense ? null : LoadMoe(c, i, src),
            PackedExperts = dense ? null : LoadPackedExperts(c, i, src),
            MoePrefix = Prefix(i),
        };
    }

    private KdaWeights LoadKda(KimiK31Config c, int i, IKimiK31WeightSource src)
    {
        string p = Layer(i) + "self_attn.";
        int key = c.KeyDim, h = c.Hidden, k = c.ConvKernel, rank = c.KdaForgetRank;

        // q, k and v stacked in the order the KDA conv splits them; their depthwise taps stacked the same way.
        float[] qkv = Stack(
            Host(src, p + "q_proj.weight", key, h),
            Host(src, p + "k_proj.weight", key, h),
            Host(src, p + "v_proj.weight", key, h));
        float[] taps = Stack(
            Host(src, p + "q_conv1d.weight", key, 1L, k),
            Host(src, p + "k_conv1d.weight", key, 1L, k),
            Host(src, p + "v_conv1d.weight", key, 1L, k));

        // One value per head. The stored [KdaHeadDim] form is accepted only with zero padding past the heads (docs/Research/KIMI_K31_ARCHITECTURE.md).
        float[] aLogHead = HostPerHead(src, p + "A_log", c.Heads, c.KdaHeadDim);
        float[] aLog = new float[key];
        for (int ch = 0; ch < key; ch++) aLog[ch] = aLogHead[ch / c.KdaHeadDim];
        (float[] up, float[] bias) = KdaLayer.FoldForgetGate(
            Host(src, p + "f_b_proj.weight", key, rank), aLog, Host(src, p + "dt_bias", key), key, rank);

        return new KdaWeights(
            QkvProj: Make(qkv, 3L * key, h),
            ConvTaps: Make(taps, 3L * key, k),
            ForgetDown: F32(src, p + "f_a_proj.weight", rank, h),
            ForgetUp: Make(up, key, rank),
            ForgetBias: Make(bias, key),
            BetaProj: F32(src, p + "b_proj.weight", c.Heads, h),
            OutGateProj: F32(src, p + "g_proj.weight", key, h),
            OutNorm: F32(src, p + "o_norm.weight", c.KdaHeadDim),
            OutProj: F32(src, p + "o_proj.weight", h, key));
    }

    private DsaAttentionWeights LoadMla(KimiK31Config c, int i, IKimiK31WeightSource src)
    {
        string p = Layer(i) + "self_attn.";
        int h = c.Hidden, qk = c.QkNopeDim + c.QkRopeDim, heads = c.Heads;
        return new DsaAttentionWeights
        {
            QA = F32(src, p + "q_a_proj.weight", c.QLoraRank, h),
            KvA = F32(src, p + "kv_a_proj_with_mqa.weight", c.KvLoraRank + c.QkRopeDim, h),
            QANorm = F32(src, p + "q_a_layernorm.weight", c.QLoraRank),
            KvANorm = F32(src, p + "kv_a_layernorm.weight", c.KvLoraRank),
            QB = F32(src, p + "q_b_proj.weight", heads * qk, c.QLoraRank),
            KvB = F32(src, p + "kv_b_proj.weight", heads * (c.QkNopeDim + c.VDim), c.KvLoraRank),
            WoB = F32(src, p + "o_proj.weight", h, heads * c.VDim),
            OGate = F32(src, p + "g_proj.weight", heads * c.VDim, h),
            Indexer = new DsaIndexerWeights
            {
                WqB = F32(src, p + "indexer.wq_b.weight", c.IndexHeads * c.IndexHeadDim, c.QLoraRank),
                Wk = F32(src, p + "indexer.wk.weight", c.IndexHeadDim, h),
                KeyNormWeight = F32(src, p + "indexer.k_norm.weight", c.IndexHeadDim),
                KeyNormBias = F32(src, p + "indexer.k_norm.bias", c.IndexHeadDim),
                WeightsProj = F32(src, p + "indexer.weights_proj.weight", c.IndexHeads, h),
            },
        };
    }

    // Engine-side names as MoeFeedForward.LoadWeights reads them, under the layer's prefix.
    private Dictionary<string, Tensor> LoadMoe(KimiK31Config c, int i, IKimiK31WeightSource src)
    {
        string block = Layer(i) + "block_sparse_moe.", mlp = Prefix(i) + ".mlp";
        int h = c.Hidden, ls = c.LatentSize, mi = c.MoeIntermediate, sh = c.SharedExperts * c.MoeIntermediate;
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal)
        {
            [$"{mlp}.gate.weight"] = F32(src, block + "gate.weight", c.Experts, h),
            [$"{mlp}.gate.e_score_correction_bias"] = F32(src, block + "gate.e_score_correction_bias", c.Experts),
            [$"{mlp}.pre_expert_proj.weight"] = F32(src, block + "routed_expert_down_proj.weight", ls, h),
            [$"{mlp}.pre_expert_norm.weight"] = F32(src, block + "routed_expert_norm.weight", ls),
            [$"{mlp}.post_expert_proj.weight"] = F32(src, block + "routed_expert_up_proj.weight", h, ls),
            [$"{mlp}.shared_expert.gate_proj.weight"] = F32(src, block + "shared_experts.gate_proj.weight", sh, h),
            [$"{mlp}.shared_expert.up_proj.weight"] = F32(src, block + "shared_experts.up_proj.weight", sh, h),
            [$"{mlp}.shared_expert.down_proj.weight"] = F32(src, block + "shared_experts.down_proj.weight", h, sh),
        };
        // A packed checkpoint's experts load through LoadPackedExperts; F32 experts are read here.
        if (!src.Has($"{block}experts.0.w1.weight_packed"))
        {
            for (int e = 0; e < c.Experts; e++)
            {
                string x = $"{block}experts.{e}";
                w[$"{mlp}.experts.{e}.gate_proj.weight"] = F32(src, $"{x}.w1.weight", mi, ls);
                w[$"{mlp}.experts.{e}.up_proj.weight"] = F32(src, $"{x}.w3.weight", mi, ls);
                w[$"{mlp}.experts.{e}.down_proj.weight"] = F32(src, $"{x}.w2.weight", ls, mi);
            }
        }
        return w;
    }

    // The MXFP4-packed routed experts of layer i: w1 is the gate, w3 the up and w2 the down projection, each a U8 weight with its E8M0
    // scales. Null when the checkpoint stores F32 experts.
    private IReadOnlyList<PackedMxfp4Expert>? LoadPackedExperts(KimiK31Config c, int i, IKimiK31WeightSource src)
    {
        string block = Layer(i) + "block_sparse_moe.";
        if (!src.Has($"{block}experts.0.w1.weight_packed")) return null;
        int ls = c.LatentSize, mi = c.MoeIntermediate;
        List<PackedMxfp4Expert> experts = new(c.Experts);
        for (int e = 0; e < c.Experts; e++)
        {
            string x = $"{block}experts.{e}";
            experts.Add(new PackedMxfp4Expert(
                Projection(src, x, "w1", mi, ls),
                Projection(src, x, "w3", mi, ls),
                Projection(src, x, "w2", ls, mi)));
        }
        return experts;
    }

    private PackedMxfp4Projection Projection(IKimiK31WeightSource src, string expert, string proj, int rows, int cols)
    {
        // The packed model keeps these views for the decoder's lifetime; the loader's owned list releases each one as it is read, so a
        // failure on the second read still releases the first.
        Tensor packed = src.Read($"{expert}.{proj}.weight_packed");
        _owned.Add(packed);
        Tensor scale = src.Read($"{expert}.{proj}.weight_scale");
        _owned.Add(scale);
        Expect(packed, $"{expert}.{proj}.weight_packed", DType.U8, rows, cols / 2);
        Expect(scale, $"{expert}.{proj}.weight_scale", DType.U8, rows, cols / 32);
        return new PackedMxfp4Projection(packed, new QuantRecipe
        {
            Encoding = QuantEncoding.Mxfp4E8M0, Geometry = new BlockGeometry(1, 32), ScaleDType = DType.U8,
            LogicalRows = rows, LogicalCols = cols, Scale = scale,
        });
    }

    private static void Expect(Tensor t, string name, DType dtype, long rows, long cols)
    {
        if (t.DType != dtype || t.Shape.Rank != 2 || t.Shape[0] != rows || t.Shape[1] != cols)
            throw new InvalidDataException($"Tensor '{name}' must be {dtype} [{rows}, {cols}], got {t.DType} {t.Shape}.");
    }

    private static string Layer(int i) => $"{Root}layers.{i}.";

    private static string Prefix(int i) => $"L{i}";

    // Reads a tensor as host F32 values and checks its exact shape.
    private static float[] Host(IKimiK31WeightSource src, string name, params long[] dims)
    {
        Tensor t = src.Read(name);
        Tensor f = t.DType == DType.F32 ? t : ProjectionOps.EnsureF32(t);
        try
        {
            if (f.Shape.Rank != dims.Length) throw Mismatch(name, f.Shape, dims);
            for (int d = 0; d < dims.Length; d++)
                if (f.Shape[d] != dims[d]) throw Mismatch(name, f.Shape, dims);
            return f.AsReadOnlySpan<float>().ToArray();
        }
        finally
        {
            if (!ReferenceEquals(f, t)) f.Dispose();
            t.Dispose();
        }
    }

    // Reads a per-head vector as host F32: exactly `heads` values, or the stored `headDim` form whose entries past `heads` must all be zero.
    private static float[] HostPerHead(IKimiK31WeightSource src, string name, int heads, int headDim)
    {
        Tensor t = src.Read(name);
        Tensor f = t.DType == DType.F32 ? t : ProjectionOps.EnsureF32(t);
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
            {
                if (values[i] != 0f)
                    throw new InvalidDataException(
                        $"Tensor '{name}' has shape {f.Shape}, but padding entry {i} is {values[i]}, not zero. The padded form is accepted only "
                        + "with a zero tail (docs/Research/KIMI_K31_ARCHITECTURE.md, A_log).");
            }
            return values[..heads];
        }
        finally
        {
            if (!ReferenceEquals(f, t)) f.Dispose();
            t.Dispose();
        }
    }

    private Tensor F32(IKimiK31WeightSource src, string name, params long[] dims) => Make(Host(src, name, dims), dims);

    // A [1, n] pseudo-query becomes the [n, 1] column the attention residual takes.
    private Tensor Column(IKimiK31WeightSource src, string name, long n) => Make(Host(src, name, 1L, n), n, 1L);

    private Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        _owned.Add(t);
        return t;
    }

    private static float[] Stack(params float[][] parts)
    {
        int total = parts.Sum(p => p.Length);
        float[] all = new float[total];
        int at = 0;
        foreach (float[] part in parts)
        {
            part.CopyTo(all, at);
            at += part.Length;
        }
        return all;
    }

    private static InvalidDataException Mismatch(string name, TensorShape got, long[] want) =>
        new($"Tensor '{name}' has shape {got}, the config needs [{string.Join(", ", want)}].");
}
