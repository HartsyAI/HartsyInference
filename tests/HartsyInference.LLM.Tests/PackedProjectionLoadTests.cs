using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Covers what <see cref="GenericTransformer"/> does at load with a checkpoint whose Linear weights are
/// ComfyUI <c>int8_tensorwise</c> — YuE2's published repack, and every other one that keeps its dequant scale on
/// <see cref="Tensor.QuantInfo"/> rather than inside the dtype.</summary>
/// <remarks>Two load-time shortcuts assume the bytes are self-describing. Q/K/V and gate/up are byte-concatenated
/// into one fused dispatch, which renumbers exactly the rows the row scale is indexed by; and the embedding table is
/// forced to F32 with a cast, which for I8 reads the raw quantization levels. Neither is a wrong magnitude — both are
/// a different weight, and nothing about either fails at load.</remarks>
public sealed unsafe class PackedProjectionLoadTests
{
    private const int Hidden = 32;
    private const int Heads = 4;
    private const int KvHeads = 2;
    private const int HeadDim = 8;
    private const int Intermediate = 32;
    private const int Vocab = 16;

    private static TransformerConfig Config() => new()
    {
        HiddenSize = Hidden, NumLayers = 1, NumHeads = Heads, NumKvHeads = KvHeads, HeadDim = HeadDim,
        IntermediateSize = Intermediate, VocabSize = Vocab, MaxPositionEmbeddings = 64,
        AttentionBias = false, QkNorm = true, TieWordEmbeddings = false,
    };

    private static Tensor F32(params long[] dims)
    {
        Tensor tensor = new Tensor(new TensorShape(dims), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = 0.01f * ((i * 13) % 41 - 20);
        return tensor;
    }

    /// <summary>An int8_tensorwise weight with a per-output-row scale and no rotation.</summary>
    private static Tensor Packed(long rows, long columns, List<Tensor> owned)
    {
        Tensor weight = new Tensor(new TensorShape(rows, columns), DType.I8);
        Span<sbyte> bytes = weight.AsSpan<sbyte>();
        for (int i = 0; i < bytes.Length; i++) bytes[i] = (sbyte)((i * 37) % 251 - 125);
        Tensor rowScale = new Tensor(new TensorShape(rows, 1), DType.F32);
        Span<float> scales = rowScale.AsSpan<float>();
        for (int i = 0; i < rows; i++) scales[i] = 0.002f * (i + 1);
        weight.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = rowScale };
        owned.Add(rowScale);
        owned.Add(weight);
        return weight;
    }

    private static Dictionary<string, Tensor> Weights(bool packedProjections, List<Tensor> owned)
    {
        Tensor Make(long rows, long columns)
        {
            if (packedProjections) return Packed(rows, columns, owned);
            Tensor dense = F32(rows, columns);
            owned.Add(dense);
            return dense;
        }
        Tensor Dense(params long[] dims)
        {
            Tensor dense = F32(dims);
            owned.Add(dense);
            return dense;
        }
        const string layer = "model.layers.0";
        return new Dictionary<string, Tensor>(StringComparer.Ordinal)
        {
            ["model.embed_tokens.weight"] = Dense(Vocab, Hidden),
            ["model.norm.weight"] = Dense(Hidden),
            ["lm_head.weight"] = Dense(Vocab, Hidden),
            [$"{layer}.input_layernorm.weight"] = Dense(Hidden),
            [$"{layer}.post_attention_layernorm.weight"] = Dense(Hidden),
            [$"{layer}.self_attn.q_proj.weight"] = Make(Heads * HeadDim, Hidden),
            [$"{layer}.self_attn.k_proj.weight"] = Make(KvHeads * HeadDim, Hidden),
            [$"{layer}.self_attn.v_proj.weight"] = Make(KvHeads * HeadDim, Hidden),
            [$"{layer}.self_attn.o_proj.weight"] = Make(Hidden, Heads * HeadDim),
            [$"{layer}.self_attn.q_norm.weight"] = Dense(HeadDim),
            [$"{layer}.self_attn.k_norm.weight"] = Dense(HeadDim),
            [$"{layer}.mlp.gate_proj.weight"] = Make(Intermediate, Hidden),
            [$"{layer}.mlp.up_proj.weight"] = Make(Intermediate, Hidden),
            [$"{layer}.mlp.down_proj.weight"] = Make(Hidden, Intermediate),
        };
    }

    [Fact]
    public void LoadWeights_NeverHandsTheBackendAPackedWeightWithoutItsScale()
    {
        List<Tensor> owned = new();
        try
        {
            Dictionary<string, Tensor> weights = Weights(packedProjections: true, owned);
            using GenericTransformer transformer = new GenericTransformer(Config());
            transformer.LoadWeights(weights, "model", "lm_head.weight");

            foreach (Tensor weight in transformer.EnumerateWeights())
            {
                Assert.False(weight.DType == DType.I8 && weight.QuantInfo is null,
                    "a fused copy of an int8_tensorwise weight carries no row scale, so it is raw quantization levels");
            }
            // The separate projections must still be the ones the layer runs, or the packed weights were fused away.
            List<Tensor> resident = [.. transformer.EnumerateWeights(includeRedundantSplits: false)];
            foreach (string key in (string[])["model.layers.0.self_attn.q_proj.weight",
                                              "model.layers.0.self_attn.k_proj.weight",
                                              "model.layers.0.self_attn.v_proj.weight",
                                              "model.layers.0.mlp.gate_proj.weight",
                                              "model.layers.0.mlp.up_proj.weight"])
            {
                Assert.Contains(resident, candidate => ReferenceEquals(candidate, weights[key]));
            }
        }
        finally
        {
            foreach (Tensor tensor in owned) tensor.Dispose();
        }
    }

    /// <summary>The dense control: fusion is still taken, so this test cannot pass by the guard simply never firing.</summary>
    [Fact]
    public void LoadWeights_StillFusesDenseProjections()
    {
        List<Tensor> owned = new();
        try
        {
            Dictionary<string, Tensor> weights = Weights(packedProjections: false, owned);
            using GenericTransformer transformer = new GenericTransformer(Config());
            transformer.LoadWeights(weights, "model", "lm_head.weight");

            List<Tensor> resident = [.. transformer.EnumerateWeights(includeRedundantSplits: false)];
            Assert.DoesNotContain(resident,
                candidate => ReferenceEquals(candidate, weights["model.layers.0.self_attn.q_proj.weight"]));
            Assert.DoesNotContain(resident,
                candidate => ReferenceEquals(candidate, weights["model.layers.0.mlp.gate_proj.weight"]));
        }
        finally
        {
            foreach (Tensor tensor in owned) tensor.Dispose();
        }
    }

    [Fact]
    public void EnsureF32_DecodesAPackedEmbeddingTableInsteadOfCastingItsBytes()
    {
        List<Tensor> owned = new();
        try
        {
            Tensor packed = Packed(Vocab, Hidden, owned);
            using Tensor decoded = GenericTransformer.EnsureF32(packed);
            Assert.Equal(DType.F32, decoded.DType);

            ReadOnlySpan<sbyte> levels = packed.AsReadOnlySpan<sbyte>();
            ReadOnlySpan<float> scales = packed.QuantInfo!.RowScale!.AsReadOnlySpan<float>();
            ReadOnlySpan<float> actual = decoded.AsReadOnlySpan<float>();
            bool differsFromTheRawLevels = false;
            for (int row = 0; row < Vocab; row++)
            {
                for (int column = 0; column < Hidden; column++)
                {
                    int i = row * Hidden + column;
                    float expected = levels[i] * scales[row];
                    // The decode lands in BF16 before widening, which keeps 8 mantissa bits — a relative bound.
                    Assert.True(Math.Abs(expected - actual[i]) <= 0.01f * Math.Abs(expected) + 1e-6f,
                        $"row {row} column {column}: expected ≈{expected}, got {actual[i]}");
                    differsFromTheRawLevels |= Math.Abs(actual[i] - levels[i]) > 1e-3f;
                }
            }
            Assert.True(differsFromTheRawLevels, "a raw cast of the quantization levels would pass a scale-free check");
        }
        finally
        {
            foreach (Tensor tensor in owned) tensor.Dispose();
        }
    }
}
