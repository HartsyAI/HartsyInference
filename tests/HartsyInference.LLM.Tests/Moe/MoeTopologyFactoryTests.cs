using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Tests.DeepSeekV41;
using HartsyInference.Core.Backends;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests.Moe;

/// <summary>Maps real configs onto the generic topology: a Qwen-style transformer and the pinned official DeepSeek-V4.1 config.</summary>
public sealed class MoeTopologyFactoryTests
{
    private static TransformerConfig QwenStyle(int layers = 4, int firstDense = 1, int shared = 512) => new()
    {
        HiddenSize = 256,
        NumLayers = layers,
        NumHeads = 4,
        NumKvHeads = 2,
        HeadDim = 64,
        IntermediateSize = 512,
        VocabSize = 1000,
        Moe = new MoeConfig
        {
            NumExperts = 16,
            NumExpertsPerTok = 4,
            MoeIntermediateSize = 128,
            SharedExpertIntermediateSize = shared,
            FirstDenseLayers = firstDense,
            NormTopKProb = true,
        },
    };

    [Fact]
    public void TransformerConfig_FirstDenseLayersStayDenseAndRestRouteThroughMoeConfig()
    {
        SparseModelTopology topology = MoeTopologyFactory.FromTransformer(QwenStyle(), DType.F32, sharedExpertGated: true);

        Assert.False(topology.Layers[0].IsSparse);
        for (int i = 1; i < 4; i++)
        {
            MoeLayerDescriptor moe = topology.Layers[i].Moe!;
            Assert.Equal(16, moe.ExpertCount);
            Assert.Equal(4, moe.Router.TopKDecode);
            Assert.Equal(MoeRouteScoring.Softmax, moe.Router.Scoring);
            Assert.True(moe.Router.Renormalize);
            Assert.Equal(0f, moe.Router.RenormEpsilon);
            Assert.NotNull(moe.Shared);
            Assert.True(moe.SharedIsGated);
            Assert.False(moe.Program.IsClamped);
        }
        Assert.True(topology.Capabilities.HasMixedDenseAndSparseLayers);
        Assert.Equal(DType.F32, topology.Layers[1].Moe!.Routed.Shape.WeightDType);
    }

    [Fact]
    public void TransformerConfig_WithoutSharedExpertOrGate_HasNoSharedGroup()
    {
        SparseModelTopology topology = MoeTopologyFactory.FromTransformer(QwenStyle(firstDense: 0, shared: 0), sharedExpertGated: true);

        Assert.All(topology.Layers, static layer => Assert.Null(layer.Moe!.Shared));
        Assert.All(topology.Layers, static layer => Assert.False(layer.Moe!.SharedIsGated));
    }

    [Fact]
    public void Qwen35HybridRule_FullAttentionOnTheIntervalElseRecurrent_OverrideWins()
    {
        Func<int, SequenceStateKind> states = MoeTopologyFactory.Qwen35States(fullAttentionInterval: 4);
        Assert.Equal(SequenceStateKind.RecurrentState, states(0));
        Assert.Equal(SequenceStateKind.RecurrentState, states(2));
        Assert.Equal(SequenceStateKind.StandardKv, states(3));
        Assert.Equal(SequenceStateKind.StandardKv, states(7));

        bool[] explicitList = [false, false, true, false];
        Func<int, SequenceStateKind> overridden = MoeTopologyFactory.Qwen35States(4, explicitList);
        Assert.Equal(SequenceStateKind.StandardKv, overridden(0));
        Assert.Equal(SequenceStateKind.RecurrentState, overridden(2));
    }

    [Fact]
    public void OfficialDeepSeekV41Config_MapsBackboneDraftAndCapabilities()
    {
        DeepSeekV41Config config = DeepSeekV41Config.Parse(DeepSeekV41Fixtures.Read("official_config.json"));

        SparseModelTopology topology = DeepSeekV41Topology.Build(config);

        Assert.Equal(config.TotalLayerCount, topology.Layers.Count);
        Assert.Equal("deepseek-v4.1", topology.Family);

        // Backbone: 384 routed experts, shared expert, clamped SwiGLU, sqrt-softplus gate with renormalization.
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            MoeLayerDescriptor moe = topology.Layers[i].Moe!;
            Assert.False(topology.Layers[i].IsDraft);
            Assert.Equal(config.NRoutedExperts, moe.ExpertCount);
            Assert.Equal(config.NumExpertsPerTok, moe.Router.TopKDecode);
            Assert.Equal(MoeRouteScoring.SqrtSoftplus, moe.Router.Scoring);
            Assert.True(moe.Router.HasSelectionBias);
            Assert.Equal(config.Vision is not null, moe.Router.HasTokenKindBias);
            Assert.NotNull(moe.Shared);
            Assert.Equal(config.MoeIntermediateSize, moe.Shared!.Shape.IntermediateSize);
            Assert.Equal((float)config.SwigluLimit, moe.Program.GateMax);
            Assert.Equal(-(float)config.SwigluLimit, moe.Program.UpMin);
        }

        // Draft (MTP/DSpark) layers carry their own expert count and no verified shared expert.
        for (int i = config.NumHiddenLayers; i < config.TotalLayerCount; i++)
        {
            SparseLayerDescriptor layer = topology.Layers[i];
            Assert.True(layer.IsDraft);
            Assert.Equal(config.DsparkNRoutedExperts, layer.Moe!.ExpertCount);
            Assert.Equal(config.DsparkNumExpertsPerTok, layer.Moe.Router.TopKDecode);
            Assert.Null(layer.Moe.Shared);
            Assert.Equal(config.SwigluLimit > 0, layer.Moe.Program.IsClamped);
        }

        SparseCapabilities caps = topology.Capabilities;
        Assert.True(caps.HasSparseExperts);
        Assert.False(caps.HasDenseLayers);
        Assert.True(caps.HasSharedExperts);
        Assert.True(caps.VariableExpertsPerLayer);
        Assert.True(caps.HasDraftLayers);
        Assert.True(caps.HasClampedPrograms);
        Assert.True(caps.HasTokenKindRouting == (config.Vision is not null));
        Assert.False(caps.RequiresHostRouting);
        Assert.Contains(DType.F4E2M1, caps.ExpertDTypes);
        Assert.Contains(DType.F8E4M3, caps.ExpertDTypes);
        Assert.Equal(DType.F8E4M3, topology.Layers[0].Moe!.Shared!.Shape.WeightDType);
        Assert.True(caps.StateKinds.Contains(SequenceStateKind.CompressedKv));
        Assert.True(caps.StateKinds.Contains(SequenceStateKind.SlidingWindowKv));
        Assert.True(topology.TotalExpertPayloadBytes > 0);
    }

    [Fact]
    public void FlatSigmoidLogitAdd_BiasesTheLogitWhileGroupedBiasesTheScore()
    {
        MoeConfig flat = new()
        {
            NumExperts = 8, NumExpertsPerTok = 2, MoeIntermediateSize = 16, Scoring = MoeScoring.SigmoidLogitAdd,
        };
        MoeConfig grouped = new()
        {
            NumExperts = 8, NumExpertsPerTok = 2, MoeIntermediateSize = 16, Scoring = MoeScoring.SigmoidLogitAdd,
            ExpertGroupCount = 2, ExpertGroupUsedCount = 1,
        };
        TransformerConfig Wrap(MoeConfig m) => new()
        {
            HiddenSize = 32, NumLayers = 1, NumHeads = 2, NumKvHeads = 2, HeadDim = 16, IntermediateSize = 64, VocabSize = 100, Moe = m,
        };

        Assert.Equal(SelectionBiasSpace.Logit, MoeTopologyFactory.FromTransformer(Wrap(flat)).Layers[0].Moe!.Router.BiasSpace);
        Assert.Equal(SelectionBiasSpace.Score, MoeTopologyFactory.FromTransformer(Wrap(grouped)).Layers[0].Moe!.Router.BiasSpace);
    }

    [Fact]
    public void OfficialDeepSeekV41Config_FingerprintIsStableAcrossBuilds()
    {
        DeepSeekV41Config config = DeepSeekV41Config.Parse(DeepSeekV41Fixtures.Read("official_config.json"));

        Assert.Equal(DeepSeekV41Topology.Build(config).Fingerprint, DeepSeekV41Topology.Build(config).Fingerprint);
    }

    [Fact]
    public void Qwen35States_RejectsBadArgumentsWithClearErrors()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => MoeTopologyFactory.Qwen35States(0));
        Func<int, SequenceStateKind> shortList = MoeTopologyFactory.Qwen35States(4, [false, true]);
        Assert.Equal(SequenceStateKind.RecurrentState, shortList(1));
        Assert.Throws<ArgumentOutOfRangeException>(() => shortList(2));
        Assert.Throws<ArgumentOutOfRangeException>(() => shortList(-1));
    }
}

