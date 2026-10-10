using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests.Moe;

/// <summary>
/// <see cref="GenericTransformer.EnumerateWeightGroups"/> feeds the grouped preload. A weight it drops would stay off the device
/// and be re-uploaded on every use; one it repeats would be preloaded twice. Neither fails loudly, so this pins the contract: the
/// same tensors as <see cref="GenericTransformer.EnumerateWeights"/>, each once, with every MoE projection's experts as one group
/// in expert order.
/// </summary>
public sealed class MoeWeightGroupTests
{
    private static Tensor F2(int a, int b) => new(new TensorShape(a, b), DType.F32);

    private static Tensor Ones(int n) => new(new TensorShape(n), DType.F32);

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void Groups_CoverEveryWeightOnce_AndGroupExpertsByProjection(int firstDenseLayers)
    {
        MoeConfig moe = new() { NumExperts = 4, NumExpertsPerTok = 2, MoeIntermediateSize = 8, FirstDenseLayers = firstDenseLayers };
        TransformerConfig cfg = new()
        {
            HiddenSize = 16, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = 4,
            IntermediateSize = 24, VocabSize = 32, MaxPositionEmbeddings = 64, Moe = moe,
        };
        using GenericTransformer model = new(cfg);
        model.LoadWeights(Weights(cfg), "model");

        List<IReadOnlyList<Tensor>> groups = [.. model.EnumerateWeightGroups()];
        List<Tensor> flat = [.. groups.SelectMany(static g => g)];
        List<Tensor> expected = [.. model.EnumerateWeights()];
        Assert.Equal(expected.Count, flat.Count);
        Assert.Equal(flat.Count, flat.Distinct(ReferenceEqualityComparer.Instance).Count());
        Assert.True(expected.ToHashSet(ReferenceEqualityComparer.Instance).SetEquals(flat));

        // One group per projection per MoE layer, each holding that projection's experts in expert order.
        List<IReadOnlyList<Tensor>> expertGroups = [.. groups.Where(static g => g.Count > 1)];
        int moeLayers = cfg.NumLayers - firstDenseLayers;
        Assert.Equal(moeLayers * 3, expertGroups.Count);
        Assert.All(expertGroups, g => Assert.Equal(moe.NumExperts, g.Count));
        Assert.All(groups.Where(static g => g.Count == 1), g => Assert.Single(g));
    }

    private static Dictionary<string, Tensor> Weights(TransformerConfig c)
    {
        int h = c.HiddenSize;
        MoeConfig m = c.Moe!;
        Dictionary<string, Tensor> w = new() { ["model.embed_tokens.weight"] = F2(c.VocabSize, h), ["model.norm.weight"] = Ones(h) };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            w[$"{p}.input_layernorm.weight"] = Ones(h);
            w[$"{p}.post_attention_layernorm.weight"] = Ones(h);
            w[$"{p}.self_attn.q_proj.weight"] = F2(c.QDim, h);
            w[$"{p}.self_attn.k_proj.weight"] = F2(c.KvDim, h);
            w[$"{p}.self_attn.v_proj.weight"] = F2(c.KvDim, h);
            w[$"{p}.self_attn.o_proj.weight"] = F2(h, c.QDim);
            if (!c.IsMoeLayer(i))
            {
                w[$"{p}.mlp.gate_proj.weight"] = F2(c.IntermediateSize, h);
                w[$"{p}.mlp.up_proj.weight"] = F2(c.IntermediateSize, h);
                w[$"{p}.mlp.down_proj.weight"] = F2(h, c.IntermediateSize);
                continue;
            }
            w[$"{p}.mlp.gate.weight"] = F2(m.NumExperts, h);
            for (int e = 0; e < m.NumExperts; e++)
            {
                w[$"{p}.mlp.experts.{e}.gate_proj.weight"] = F2(m.MoeIntermediateSize, h);
                w[$"{p}.mlp.experts.{e}.up_proj.weight"] = F2(m.MoeIntermediateSize, h);
                w[$"{p}.mlp.experts.{e}.down_proj.weight"] = F2(h, m.MoeIntermediateSize);
            }
        }
        return w;
    }
}
