using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Graph decode that a backend turns on by itself must never turn a request that decodes eagerly into an error:
/// a capture that throws logs and falls back, while a graph the request asked for still reports its failure. The model
/// here claims graph support and fails every capture; the decode underneath is the real CPU transformer.</summary>
public sealed class GraphDecodeImplicitFallbackTests
{
    private static uint _rng = 0xFA11BACu;
    private static uint NextRaw() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return _rng; }
    private static unsafe Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = ((NextRaw() & 0xFFFF) / 65535f - 0.5f) * 0.3f; return t; }
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static unsafe Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); new Span<float>((float*)t.DataPointer, n).Fill(1f); return t; }

    private static TransformerConfig Cfg() => new()
    {
        HiddenSize = 16, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = 4,
        IntermediateSize = 32, VocabSize = 32, MaxPositionEmbeddings = 1024, QkNorm = false,
    };

    private static Dictionary<string, Tensor> Weights(TransformerConfig c)
    {
        int h = c.HiddenSize;
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
            w[$"{p}.mlp.gate_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.up_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.down_proj.weight"] = F2(h, c.IntermediateSize);
        }
        return w;
    }

    private sealed class StubTokenizer : ILlmTokenizer
    {
        public int[] Encode(string text, bool addSpecial) => throw new NotSupportedException();
        public int[] EncodeOrdinary(string text) => throw new NotSupportedException();
        public string Decode(IReadOnlyList<int> ids) => string.Join(",", ids);
        public int? SpecialId(string token) => null;
        public int? BosId => null;
        public int? EosId => null;
        public IReadOnlyList<int> StopIds => [];
        public string? BosToken => null;
        public string? EosToken => null;
    }

    /// <summary>The real model, plus a graph capability whose capture always throws.</summary>
    private sealed class FailingGraphModel(GenericTransformerModel inner) : IGenerationModel, IGraphDecodable
    {
        public int CaptureAttempts { get; private set; }

        public GenerationModelInfo Info => inner.Info;
        public GenerationCapabilities Capabilities => inner.Capabilities;
        public IBackend OutputBackend => inner.OutputBackend;
        public ISequenceState CreateSequenceState(SequenceStateOptions options) => inner.CreateSequenceState(options);
        public Tensor Prefill(in PrefillChunk chunk, ISequenceState state) => inner.Prefill(chunk, state);
        public Tensor Prefill(in PrefillChunk chunk, ISequenceState state, CancellationToken cancel) => inner.Prefill(chunk, state, cancel);
        public Tensor DecodeBatch(ReadOnlySpan<int> tokenIds, ISequenceState[] states) => inner.DecodeBatch(tokenIds, states);
        public Tensor ProjectLogits(Tensor hidden, int rows) => inner.ProjectLogits(hidden, rows);
        public IEnumerable<Tensor> EnumerateWeights(bool includeRedundantSplits) => inner.EnumerateWeights(includeRedundantSplits);
        public long EstimateSequenceBytes(int contextTokens) => inner.EstimateSequenceBytes(contextTokens);
        public CapacitySnapshot Capacity() => inner.Capacity();
        public void Dispose() => inner.Dispose();

        bool IGraphDecodable.SupportsGraphDecode(IBackend backend) => true;
        bool IGraphDecodable.GraphDecodeByDefault(IBackend backend) => true;
        void IGraphDecodable.CommitReplayedStep(ISequenceState state) { }

        GraphDecodeSession IGraphDecodable.CaptureDecodeGraph(ISequenceState state, int pos, int firstToken, float repetitionPenalty, DeviceSamplerConfig? sampler)
        {
            CaptureAttempts++;
            throw new InvalidOperationException("synthetic capture failure (test)");
        }
    }

    private static GenerationRequest Req(int maxTokens, bool? graph) => new()
    {
        RawTokenIds = [1, 2, 3, 4, 5],
        MaxTokens = maxTokens,
        Sampling = SamplingOptions.Default with { Greedy = true },
        GraphDecode = graph,
    };

    private static (FailingGraphModel Model, TextGenerationPipeline Pipeline, Dictionary<string, Tensor> W, CpuBackend Backend) Build()
    {
        TransformerConfig cfg = Cfg();
        Dictionary<string, Tensor> w = Weights(cfg);
        CpuBackend backend = new();
        GenericTransformer transformer = new(cfg);
        transformer.LoadWeights(w, "model");
        FailingGraphModel model = new(new GenericTransformerModel(transformer, backend));
        return (model, new TextGenerationPipeline(model, new StubTokenizer()), w, backend);
    }

    [Fact]
    public void ADefaultGraph_ThatCannotBeCaptured_DecodesEagerlyWithTheSameTokens()
    {
        (FailingGraphModel model, TextGenerationPipeline pipeline, Dictionary<string, Tensor> w, CpuBackend backend) = Build();
        using (model) using (backend)
        {
            int n = TextGenerationPipeline.ImplicitGraphMinTokens;
            GenerationResult got = pipeline.Generate(Req(n, graph: null));
            Assert.Equal(1, model.CaptureAttempts);
            Assert.Equal(n, got.TokenIds.Count);

            GenerationResult eager = pipeline.Generate(Req(n, graph: false));
            Assert.Equal(1, model.CaptureAttempts);   // an explicit "no" never tries
            Assert.Equal(eager.TokenIds, got.TokenIds);
        }
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void AnExplicitGraph_ThatCannotBeCaptured_ReportsTheFailure()
    {
        (FailingGraphModel model, TextGenerationPipeline pipeline, Dictionary<string, Tensor> w, CpuBackend backend) = Build();
        using (model) using (backend)
        {
            InvalidOperationException ex = Assert.Throws<InvalidOperationException>(() => pipeline.Generate(Req(16, graph: true)));
            Assert.Equal("synthetic capture failure (test)", ex.Message);
        }
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void AShortRequest_DoesNotTakeTheDefaultGraph()
    {
        (FailingGraphModel model, TextGenerationPipeline pipeline, Dictionary<string, Tensor> w, CpuBackend backend) = Build();
        using (model) using (backend)
        {
            pipeline.Generate(Req(TextGenerationPipeline.ImplicitGraphMinTokens - 1, graph: null));
            Assert.Equal(0, model.CaptureAttempts);
        }
        foreach (Tensor t in w.Values) t.Dispose();
    }
}
