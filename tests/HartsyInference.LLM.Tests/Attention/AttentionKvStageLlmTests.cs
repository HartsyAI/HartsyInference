using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests.Attention;

/// <summary>How the LLM caches serve the attention stage: the default <see cref="IAttentionKvSource"/> members forward to the cache's own methods, and a
/// forward reads each layer's K then V exactly once after the append (a paged cache gathers on every read), through whole, plain buffers.</summary>
public sealed unsafe class AttentionKvStageLlmTests
{
    private const int Vocab = 24, Hidden = 16, Inter = 24, Capacity = 16, Layers = 4;

    public enum Kind { Fixed, Growing, Paged, Streaming }

    private static TransformerConfig Config(int sharedFrom = 0) => new()
    {
        HiddenSize = Hidden, NumLayers = Layers, NumHeads = 4, NumKvHeads = 2, HeadDim = 8, IntermediateSize = Inter, VocabSize = Vocab,
        MaxPositionEmbeddings = 64, RopeTheta = 10_000f, TieWordEmbeddings = false, KvSharedFromLayer = sharedFrom,
    };

    private static Tensor Seeded(string name, params long[] dims)
    {
        Tensor tensor = new(new TensorShape(dims), DType.F32);
        float* data = (float*)tensor.DataPointer;
        uint state = 2166136261u;
        foreach (char c in name) state = (state ^ c) * 16777619u;
        for (long i = 0; i < tensor.ElementCount; i++)
        {
            state = state * 1664525u + 1013904223u;
            data[i] = (((state >> 8) & 0xFFFF) / 32768f - 1f) * 0.5f;
        }
        return tensor;
    }

    private static Dictionary<string, Tensor> Weights(TransformerConfig c)
    {
        Dictionary<string, Tensor> w = new();
        void Add(string name, params long[] dims) => w[name] = Seeded(name, dims);
        Add("model.embed_tokens.weight", c.VocabSize, c.HiddenSize);
        Add("lm_head.weight", c.VocabSize, c.HiddenSize);
        Add("model.norm.weight", c.HiddenSize);
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            Add($"{p}.input_layernorm.weight", c.HiddenSize);
            Add($"{p}.post_attention_layernorm.weight", c.HiddenSize);
            Add($"{p}.self_attn.q_proj.weight", c.NumHeads * c.HeadDim, c.HiddenSize);
            if (c.HasOwnKv(i))
            {
                Add($"{p}.self_attn.k_proj.weight", c.NumKvHeads * c.HeadDim, c.HiddenSize);
                Add($"{p}.self_attn.v_proj.weight", c.NumKvHeads * c.HeadDim, c.HiddenSize);
            }
            Add($"{p}.self_attn.o_proj.weight", c.HiddenSize, c.NumHeads * c.HeadDim);
            Add($"{p}.mlp.gate_proj.weight", Inter, c.HiddenSize);
            Add($"{p}.mlp.up_proj.weight", Inter, c.HiddenSize);
            Add($"{p}.mlp.down_proj.weight", c.HiddenSize, Inter);
        }
        // Norm scales near 1 keep the tiny model well-conditioned; only the call pattern matters here.
        foreach ((string name, Tensor t) in w.Where(e => e.Key.EndsWith("norm.weight", StringComparison.Ordinal) || e.Key.EndsWith("layernorm.weight", StringComparison.Ordinal)))
            for (long i = 0; i < t.ElementCount; i++) ((float*)t.DataPointer)[i] += 1f;
        return w;
    }

    private sealed class Owned : IDisposable
    {
        private readonly List<IDisposable> _items = [];

        public Dictionary<string, Tensor>? Weights { get; set; }

        public T Add<T>(T item) where T : IDisposable
        {
            _items.Add(item);
            return item;
        }

        public IKvCache Cache(Kind kind, TransformerConfig c, PagedKvPool? pool = null) => Add<IKvCache>(kind switch
        {
            Kind.Fixed => new FixedKvCache(c.NumLayers, 1, c.NumKvHeads, c.HeadDimsPerLayer(), Capacity),
            Kind.Growing => new KvCache(c.NumLayers, 1, c.NumKvHeads, c.HeadDim),
            Kind.Streaming => new StreamingKvCache(c.NumLayers, 1, c.NumKvHeads, Capacity, c.HeadDim),
            _ => new PagedKvCache(pool ?? throw new ArgumentNullException(nameof(pool))),
        });

        public void Dispose()
        {
            for (int i = _items.Count - 1; i >= 0; i--) _items[i].Dispose();
            if (Weights is not null)
                foreach (Tensor t in Weights.Values) t.Dispose();
        }
    }

    /// <summary>Wraps a cache and logs <c>A</c>ppend, <c>K</c>ey and <c>V</c>alue reads per layer; the default <see cref="IAttentionKvSource"/> members route through it.</summary>
    private sealed class CountingKvCache(IKvCache inner, List<string> log) : IKvCache
    {
        public int NumLayers => inner.NumLayers;
        public int CurrentLength => inner.CurrentLength;
        public int Length => inner.CurrentLength;
        public int Capacity => ((ISequenceState)inner).Capacity;
        public int MaxRollback => ((ISequenceState)inner).MaxRollback;

        public void AppendStep(IBackend backend, int layer, Tensor newK, Tensor newV)
        {
            log.Add($"A{layer}");
            inner.AppendStep(backend, layer, newK, newV);
        }

        public Tensor KeyPrefix(int layer)
        {
            log.Add($"K{layer}");
            return inner.KeyPrefix(layer);
        }

        public Tensor ValuePrefix(int layer)
        {
            log.Add($"V{layer}");
            return inner.ValuePrefix(layer);
        }

        public void AdvanceLength(int by) => inner.AdvanceLength(by);
        public void Truncate(int newLength) => ((ISequenceState)inner).Truncate(newLength);
        public void Reset() => inner.Reset();
        public void Dispose() { }
    }

    [Fact]
    public void TheDefaultSourceMembersForwardToTheCacheMethods()
    {
        List<string> log = [];
        using Owned owned = new();
        TransformerConfig cfg = Config();
        using CpuBackend cpu = new();
        CountingKvCache cache = new(owned.Cache(Kind.Growing, cfg), log);
        IAttentionKvSource source = cache;
        Tensor k = owned.Add(new Tensor(new TensorShape(1, 2, 3, 8), DType.F32)), v = owned.Add(new Tensor(new TensorShape(1, 2, 3, 8), DType.F32));

        source.AppendKv(cpu, 0, k, v);
        Tensor keys = source.KeyBuffer(0);
        Tensor values = source.ValueBuffer(0);

        Assert.Equal(["A0", "K0", "V0"], log);
        Assert.Same(cache.KeyPrefix(0), keys);
        Assert.Same(cache.ValuePrefix(0), values);
        Assert.Equal(0, cache.CurrentLength);
    }

    public static TheoryData<Kind, int> Forwards => new()
    {
        { Kind.Fixed, 0 }, { Kind.Growing, 0 }, { Kind.Paged, 0 }, { Kind.Streaming, 0 },
        { Kind.Fixed, 2 }, { Kind.Paged, 2 },
    };

    [Theory]
    [MemberData(nameof(Forwards))]
    public void AForwardReadsEachLayersKeyThenValueOnceAfterTheAppend(Kind kind, int sharedFrom)
    {
        TransformerConfig cfg = Config(sharedFrom);
        using Owned owned = new() { Weights = Weights(cfg) };
        using CpuBackend cpu = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(owned.Weights!, "model");
        PagedKvPool? pool = kind == Kind.Paged ? owned.Add(new PagedKvPool(Layers, cfg.NumKvHeads, cfg.HeadDimsPerLayer(), 4, 16)) : null;
        List<string> log = [];
        IKvCache cache = new CountingKvCache(owned.Cache(kind, cfg, pool), log);

        // A prefill of five tokens, then three single-token decodes at growing offsets.
        foreach (int[] ids in new int[][] { [3, 7, 1, 12, 5], [9], [4], [11] })
        {
            log.Clear();
            using Tensor embeds = new(new TensorShape(1, ids.Length, Hidden), DType.F32);
            model.EmbedLookup(embeds, ids);
            using Tensor hidden = model.ForwardEmbeds(cpu, embeds, ids.Length, cache.CurrentLength, cache, tokenIds: ids);

            List<string> expected = [];
            for (int l = 0; l < Layers; l++)
            {
                if (cfg.HasOwnKv(l)) expected.Add($"A{l}");
                int source = cfg.HasOwnKv(l) ? l : cfg.KvDonorLayer(l);
                expected.Add($"K{source}");
                expected.Add($"V{source}");
            }
            Assert.Equal(expected, log);
        }
    }

    [Theory]
    [InlineData(Kind.Fixed)]
    [InlineData(Kind.Paged)]
    public void ABatchedDecodeReadsEachSequencesKeyThenValueOncePerLayer(Kind kind)
    {
        TransformerConfig cfg = Config();
        using Owned owned = new() { Weights = Weights(cfg) };
        using CpuBackend cpu = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(owned.Weights!, "model");
        PagedKvPool? pool = kind == Kind.Paged ? owned.Add(new PagedKvPool(Layers, cfg.NumKvHeads, cfg.HeadDimsPerLayer(), 4, 16)) : null;
        List<string> logA = [], logB = [];
        IKvCache[] caches = [new CountingKvCache(owned.Cache(kind, cfg, pool), logA), new CountingKvCache(owned.Cache(kind, cfg, pool), logB)];
        int[][] prompts = [[3, 7, 1, 12, 5], [11, 2, 20]];
        for (int s = 0; s < 2; s++)
        {
            using Tensor embeds = new(new TensorShape(1, prompts[s].Length, Hidden), DType.F32);
            model.EmbedLookup(embeds, prompts[s]);
            using Tensor hidden = model.ForwardEmbeds(cpu, embeds, prompts[s].Length, 0, caches[s], tokenIds: prompts[s]);
        }
        logA.Clear();
        logB.Clear();

        using Tensor batch = new(new TensorShape(1, 2, Hidden), DType.F32);
        model.EmbedLookup(batch, [4, 6]);
        using Tensor result = model.ForwardBatchDecode(cpu, batch, [caches[0].CurrentLength, caches[1].CurrentLength], caches);

        string[] expected = [.. Enumerable.Range(0, Layers).SelectMany(l => new[] { $"A{l}", $"K{l}", $"V{l}" })];
        Assert.Equal(expected, logA);
        Assert.Equal(expected, logB);
    }

    [Theory]
    [InlineData(Kind.Fixed)]
    [InlineData(Kind.Growing)]
    [InlineData(Kind.Paged)]
    [InlineData(Kind.Streaming)]
    public void EveryCacheServesWholePlainHeadMajorBuffersTheFlashRouteAccepts(Kind kind)
    {
        TransformerConfig cfg = Config();
        using Owned owned = new() { Weights = Weights(cfg) };
        using CpuBackend cpu = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(owned.Weights!, "model");
        PagedKvPool? pool = kind == Kind.Paged ? owned.Add(new PagedKvPool(Layers, cfg.NumKvHeads, cfg.HeadDimsPerLayer(), 4, 16)) : null;
        IKvCache cache = owned.Cache(kind, cfg, pool);

        foreach (int[] ids in new int[][] { [3, 7, 1, 12, 5], [9] })
        {
            using Tensor embeds = new(new TensorShape(1, ids.Length, Hidden), DType.F32);
            model.EmbedLookup(embeds, ids);
            using Tensor hidden = model.ForwardEmbeds(cpu, embeds, ids.Length, cache.CurrentLength, cache, tokenIds: ids);
            IAttentionKvSource source = cache;
            for (int l = 0; l < Layers; l++)
            {
                foreach (Tensor buffer in (Tensor[])[source.KeyBuffer(l), source.ValueBuffer(l)])
                {
                    Assert.Equal(cfg.NumKvHeads, buffer.Shape[1]);
                    Assert.True(buffer.Shape[2] >= cache.CurrentLength, $"{kind} layer {l} buffer {buffer.Shape} is shorter than the committed length.");
                    Assert.True(AttentionOperand.HeadMajor(buffer).IsPlain);
                }
            }
        }
    }
}
