using System.Reflection;
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
/// here claims graph support and fails every capture; the decode underneath is the real CPU transformer.
/// The implicit threshold is the backend's own (<c>IBackend.LlmGraphDecodeDefaultMinTokens</c>): 512 on the CPU and CUDA, 128 on Vulkan.
/// The sliding-window and soft-cap gate (Gemma-2/3 on Vulkan) is at the end: the backend answers only the graph capability queries,
/// and the real gate and refusal decide what the pipeline and the scheduler do with each request.</summary>
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
            int n = ((IBackend)backend).LlmGraphDecodeDefaultMinTokens;
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
            pipeline.Generate(Req(((IBackend)backend).LlmGraphDecodeDefaultMinTokens - 1, graph: null));
            Assert.Equal(0, model.CaptureAttempts);
        }
        foreach (Tensor t in w.Values) t.Dispose();
    }

    /// <summary>The implicit graph threshold per backend: 512 for CUDA and the CPU, which were not measured lower; 128 for Vulkan, measured on the RTX 3060.</summary>
    private const int DefaultMinTokens = 512;
    private const int VulkanMinTokens = 128;

    /// <summary>A backend that answers only the graph-decode capability queries the gate reads. <see cref="ByDefault"/> is the implicit-on policy (Vulkan takes it,
    /// CUDA keeps the opt-in); <see cref="WindowSoftcap"/> says whether the device attention takes a sliding window and a soft-cap (CUDA does, Vulkan does not);
    /// <see cref="MinTokens"/> is the implicit threshold it reports. Any other member throws, so a test that reaches a device op through it fails loudly instead of passing on a stub value.</summary>
    private class GraphGateBackend : DispatchProxy
    {
        public bool ByDefault { get; set; }
        public bool WindowSoftcap { get; set; }
        public int MinTokens { get; set; }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            string name = targetMethod!.Name;
            return name switch
            {
                "get_LlmGraphDecodeSupported" => true,
                "get_LlmGraphDecodeDefault" => ByDefault,
                "get_LlmGraphDecodeDefaultMinTokens" => MinTokens,
                "get_LlmGraphDecodeWindowSoftcap" => WindowSoftcap,
                _ => throw new NotSupportedException($"the graph gate test backend answers only graph capability queries, not {name}"),
            };
        }
    }

    /// <summary>A backend that reports graph decode on, with the given implicit-default policy, window/soft-cap support and implicit threshold.</summary>
    private static IBackend Gate(bool byDefault, bool windowSoftcap, int minTokens)
    {
        IBackend backend = DispatchProxy.Create<IBackend, GraphGateBackend>();
        GraphGateBackend probe = (GraphGateBackend)(object)backend;
        probe.ByDefault = byDefault;
        probe.WindowSoftcap = windowSoftcap;
        probe.MinTokens = minTokens;
        return backend;
    }

    /// <summary>The real model, with the backend the pipeline and the scheduler read graph decisions from set to <paramref name="gate"/>. Every tensor op still runs on
    /// the inner model's CPU backend, and the gate and the refusal are the real <see cref="GenericTransformer"/> code, asked about <paramref name="gate"/>. A capture always fails and is counted.</summary>
    private sealed class GateModel(GenericTransformerModel inner, IBackend gate) : IGenerationModel, IGraphDecodable
    {
        public int CaptureAttempts { get; private set; }

        public GenerationModelInfo Info => inner.Info;
        public GenerationCapabilities Capabilities => inner.Capabilities;
        public IBackend OutputBackend => gate;
        public ISequenceState CreateSequenceState(SequenceStateOptions options) => inner.CreateSequenceState(options);
        public Tensor Prefill(in PrefillChunk chunk, ISequenceState state) => inner.Prefill(chunk, state);
        public Tensor Prefill(in PrefillChunk chunk, ISequenceState state, CancellationToken cancel) => inner.Prefill(chunk, state, cancel);
        public Tensor DecodeBatch(ReadOnlySpan<int> tokenIds, ISequenceState[] states) => inner.DecodeBatch(tokenIds, states);
        public Tensor ProjectLogits(Tensor hidden, int rows) => inner.ProjectLogits(hidden, rows);
        public IEnumerable<Tensor> EnumerateWeights(bool includeRedundantSplits) => inner.EnumerateWeights(includeRedundantSplits);
        public long EstimateSequenceBytes(int contextTokens) => inner.EstimateSequenceBytes(contextTokens);
        public CapacitySnapshot Capacity() => inner.Capacity();
        public void Dispose() => inner.Dispose();

        bool IGraphDecodable.SupportsGraphDecode(IBackend backend) => ((IGraphDecodable)inner).SupportsGraphDecode(backend);
        bool IGraphDecodable.GraphDecodeByDefault(IBackend backend) => ((IGraphDecodable)inner).GraphDecodeByDefault(backend);
        string? IGraphDecodable.GraphDecodeRefusal(IBackend backend) => ((IGraphDecodable)inner).GraphDecodeRefusal(backend);
        bool IGraphDecodable.SupportsDeviceSampling(IBackend backend) => ((IGraphDecodable)inner).SupportsDeviceSampling(backend);
        void IGraphDecodable.CommitReplayedStep(ISequenceState state) => ((IGraphDecodable)inner).CommitReplayedStep(state);

        GraphDecodeSession IGraphDecodable.CaptureDecodeGraph(ISequenceState state, int pos, int firstToken, float repetitionPenalty, DeviceSamplerConfig? sampler)
        {
            CaptureAttempts++;
            throw new InvalidOperationException("synthetic capture failure (test)");
        }
    }

    /// <summary>Gemma-2's attention at test size: a sliding window in the local layers, a soft-cap on the scores, and the final-logit cap.</summary>
    private static TransformerConfig Gemma2Cfg() => Cfg() with
    {
        Architecture = "gemma2", SlidingWindowPattern = 2, SlidingWindow = 4, AttnLogitSoftcap = 50f, FinalLogitSoftcap = 30f,
    };

    /// <summary>Gemma-3's attention at test size: a sliding window in the local layers and no soft-cap.</summary>
    private static TransformerConfig Gemma3Cfg() => Cfg() with { Architecture = "gemma3", SlidingWindowPattern = 2, SlidingWindow = 4 };

    private const string VulkanGemma2Refusal =
        "gemma2: sliding-window attention and attention logit soft-cap are not supported by the Vulkan decode graph; decode eagerly or use CUDA";
    private const string VulkanGemma3Refusal =
        "gemma3: sliding-window attention is not supported by the Vulkan decode graph; decode eagerly or use CUDA";

    private static (GateModel Model, TextGenerationPipeline Pipeline, Dictionary<string, Tensor> W, CpuBackend Backend) BuildGated(TransformerConfig cfg, IBackend gate)
    {
        Dictionary<string, Tensor> w = Weights(cfg);
        CpuBackend backend = new();
        GenericTransformer transformer = new(cfg);
        transformer.LoadWeights(w, "model");
        GateModel model = new(new GenericTransformerModel(transformer, backend), gate);
        return (model, new TextGenerationPipeline(model, new StubTokenizer()), w, backend);
    }

    [Fact]
    public void TheCpuBackend_KeepsTheImplicitGraphThresholdAt512()
    {
        using CpuBackend cpu = new();
        Assert.False(((IBackend)cpu).LlmGraphDecodeDefault);
        Assert.Equal(DefaultMinTokens, ((IBackend)cpu).LlmGraphDecodeDefaultMinTokens);
    }

    [Fact]
    public void VulkanShape_ImplicitGraphStartsAt128Tokens_AndNotBelow()
    {
        (GateModel model, TextGenerationPipeline pipeline, Dictionary<string, Tensor> w, CpuBackend backend) = BuildGated(Cfg(), Gate(byDefault: true, windowSoftcap: false, minTokens: VulkanMinTokens));
        using (model) using (backend)
        {
            pipeline.Generate(Req(VulkanMinTokens - 1, graph: null));
            Assert.Equal(0, model.CaptureAttempts);

            GenerationResult got = pipeline.Generate(Req(VulkanMinTokens, graph: null));
            Assert.Equal(1, model.CaptureAttempts);   // the capture fails, and the request still decodes eagerly

            GenerationResult eager = pipeline.Generate(Req(VulkanMinTokens, graph: false));
            Assert.Equal(eager.TokenIds, got.TokenIds);
        }
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void ImplicitGraph_ReadsTheBackendsOwnThreshold_At512()
    {
        // A backend that defaults on at 512 must not start at 128: the gate reads the backend's value, not a constant.
        (GateModel model, TextGenerationPipeline pipeline, Dictionary<string, Tensor> w, CpuBackend backend) = BuildGated(Cfg(), Gate(byDefault: true, windowSoftcap: true, minTokens: DefaultMinTokens));
        using (model) using (backend)
        {
            pipeline.Generate(Req(DefaultMinTokens - 1, graph: null));
            Assert.Equal(0, model.CaptureAttempts);

            pipeline.Generate(Req(DefaultMinTokens, graph: null));
            Assert.Equal(1, model.CaptureAttempts);
        }
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void ImplicitGraph_OnABackendThatDoesNotDefaultIt_StaysOptIn_AtAnyLength()
    {
        // CUDA's shape: graph decode is opt-in there, so a 512-token request never tries the capture on its own.
        (GateModel model, TextGenerationPipeline pipeline, Dictionary<string, Tensor> w, CpuBackend backend) = BuildGated(Cfg(), Gate(byDefault: false, windowSoftcap: true, minTokens: DefaultMinTokens));
        using (model) using (backend)
        {
            pipeline.Generate(Req(DefaultMinTokens, graph: null));
            Assert.Equal(0, model.CaptureAttempts);
        }
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void GemmaConfigs_OnVulkan_ReportGraphDecodeUnsupported_AndNameTheRefusal()
    {
        IBackend vulkan = Gate(byDefault: true, windowSoftcap: false, minTokens: VulkanMinTokens);
        using GenericTransformer gemma2 = new(TransformerConfig.Gemma2_2B);
        using GenericTransformer gemma3 = new(TransformerConfig.Gemma3_1B);

        Assert.False(gemma2.SupportsGraphDecode(vulkan));
        Assert.False(gemma3.SupportsGraphDecode(vulkan));
        Assert.Equal(VulkanGemma2Refusal, gemma2.GraphDecodeRefusal(vulkan));
        Assert.Equal(VulkanGemma3Refusal, gemma3.GraphDecodeRefusal(vulkan));
    }

    [Fact]
    public void WindowAndSoftcap_OnlyCountWhereALayerUsesThem()
    {
        IBackend vulkan = Gate(byDefault: true, windowSoftcap: false, minTokens: VulkanMinTokens);
        using GenericTransformer plain = new(Cfg());
        Assert.True(plain.SupportsGraphDecode(vulkan));
        Assert.Null(plain.GraphDecodeRefusal(vulkan));

        // With pattern 0 every layer is global, so no layer passes a window to the kernel: Vulkan runs it, as it always did.
        using GenericTransformer allGlobal = new(Cfg() with { SlidingWindow = 4 });
        Assert.True(allGlobal.SupportsGraphDecode(vulkan));
        Assert.Null(allGlobal.GraphDecodeRefusal(vulkan));
    }

    [Fact]
    public void GemmaConfigs_OnCudaAndCpu_AreUnchanged()
    {
        using GenericTransformer gemma2 = new(TransformerConfig.Gemma2_2B);
        using GenericTransformer gemma3 = new(TransformerConfig.Gemma3_1B);

        IBackend cuda = Gate(byDefault: false, windowSoftcap: true, minTokens: DefaultMinTokens);
        Assert.True(gemma2.SupportsGraphDecode(cuda));
        Assert.True(gemma3.SupportsGraphDecode(cuda));
        Assert.Null(gemma2.GraphDecodeRefusal(cuda));
        Assert.Null(gemma3.GraphDecodeRefusal(cuda));

        // The CPU backend reports no graph decode at all, so the same configs stay ineligible and unrefused.
        using CpuBackend cpu = new();
        Assert.False(gemma2.SupportsGraphDecode(cpu));
        Assert.Null(gemma2.GraphDecodeRefusal(cpu));
    }

    [Theory]
    [InlineData("gemma2", VulkanGemma2Refusal)]
    [InlineData("gemma3", VulkanGemma3Refusal)]
    public void AnExplicitGraph_ForAGemmaShapeOnVulkan_IsRefusedByName_BeforeAnyWork(string arch, string expected)
    {
        TransformerConfig cfg = arch == "gemma2" ? Gemma2Cfg() : Gemma3Cfg();
        (GateModel model, TextGenerationPipeline pipeline, Dictionary<string, Tensor> w, CpuBackend backend) = BuildGated(cfg, Gate(byDefault: true, windowSoftcap: false, minTokens: VulkanMinTokens));
        using (model) using (backend)
        {
            NotSupportedException ex = Assert.Throws<NotSupportedException>(() => pipeline.Generate(Req(16, graph: true)));
            Assert.Equal(expected, ex.Message);
            Assert.Equal(0, model.CaptureAttempts);
        }
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void AnExplicitGraph_ForAGemmaShapeOnCuda_StillReachesTheCapture()
    {
        (GateModel model, TextGenerationPipeline pipeline, Dictionary<string, Tensor> w, CpuBackend backend) = BuildGated(Gemma2Cfg(), Gate(byDefault: false, windowSoftcap: true, minTokens: DefaultMinTokens));
        using (model) using (backend)
        {
            InvalidOperationException ex = Assert.Throws<InvalidOperationException>(() => pipeline.Generate(Req(16, graph: true)));
            Assert.Equal("synthetic capture failure (test)", ex.Message);
            Assert.Equal(1, model.CaptureAttempts);
        }
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void AnImplicitGraph_ForAGemmaShapeOnVulkan_DecodesEagerly_WithoutAttemptingACapture()
    {
        (GateModel model, TextGenerationPipeline pipeline, Dictionary<string, Tensor> w, CpuBackend backend) = BuildGated(Gemma2Cfg(), Gate(byDefault: true, windowSoftcap: false, minTokens: VulkanMinTokens));
        using (model) using (backend)
        {
            int n = VulkanMinTokens;
            GenerationResult got = pipeline.Generate(Req(n, graph: null));
            Assert.Equal(0, model.CaptureAttempts);
            Assert.Equal(n, got.TokenIds.Count);

            GenerationResult eager = pipeline.Generate(Req(n, graph: false));
            Assert.Equal(eager.TokenIds, got.TokenIds);
        }
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void AnImplicitGraph_ForAGemmaShapeOnABackendThatSupportsIt_StillTriesTheCapture()
    {
        // The control for the test above: the same model and request, on a backend whose device attention takes the window and soft-cap.
        (GateModel model, TextGenerationPipeline pipeline, Dictionary<string, Tensor> w, CpuBackend backend) = BuildGated(Gemma2Cfg(), Gate(byDefault: true, windowSoftcap: true, minTokens: DefaultMinTokens));
        using (model) using (backend)
        {
            int n = DefaultMinTokens;
            GenerationResult got = pipeline.Generate(Req(n, graph: null));
            Assert.Equal(1, model.CaptureAttempts);

            GenerationResult eager = pipeline.Generate(Req(n, graph: false));
            Assert.Equal(eager.TokenIds, got.TokenIds);
        }
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public async Task AnExplicitGraph_ForAGemmaShapeOnVulkan_FailsAtAdmission_NamingTheRefusal()
    {
        (GateModel model, _, Dictionary<string, Tensor> w, CpuBackend backend) = BuildGated(Gemma2Cfg(), Gate(byDefault: true, windowSoftcap: false, minTokens: VulkanMinTokens));
        using (model) using (backend)
        {
            using DynamicBatchScheduler scheduler = new(model, new DynamicBatchSchedulerTests.StubTokenizer(), pool: null);
            NotSupportedException ex = await Assert.ThrowsAsync<NotSupportedException>(
                () => scheduler.SubmitAsync(Req(16, graph: true), null, CancellationToken.None)).WaitAsync(TimeSpan.FromSeconds(60));
            Assert.Equal(VulkanGemma2Refusal, ex.Message);
            Assert.Equal(0, model.CaptureAttempts);
        }
        foreach (Tensor t in w.Values) t.Dispose();
    }
}
