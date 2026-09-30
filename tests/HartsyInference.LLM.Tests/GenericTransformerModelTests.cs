using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Adapter, KV checkpoint/rollback and hoisted-helper gate for the generation contract.</summary>
public sealed unsafe class GenericTransformerModelTests
{
    private const float LogitTolerance = 1e-4f;

    private static uint _rng = 0x9E3779B9u;
    private static float Rand()
    {
        _rng ^= _rng << 13;
        _rng ^= _rng >> 17;
        _rng ^= _rng << 5;
        return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f;
    }
    private static Tensor Fill(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = Rand();
        return t;
    }
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor Ones(int n)
    {
        Tensor t = new(new TensorShape(n), DType.F32);
        new Span<float>((float*)t.DataPointer, n).Fill(1f);
        return t;
    }

    private static TransformerConfig Cfg() => new()
    {
        HiddenSize = 16, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = 4,
        IntermediateSize = 32, VocabSize = 24, MaxPositionEmbeddings = 256, AttentionBias = true, QkNorm = false,
    };

    private static Dictionary<string, Tensor> Weights(TransformerConfig c)
    {
        int h = c.HiddenSize, qDim = c.QDim, kvDim = c.KvDim;
        Dictionary<string, Tensor> w = new()
        {
            ["model.embed_tokens.weight"] = F2(c.VocabSize, h),
            ["model.norm.weight"] = Ones(h),
        };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            w[$"{p}.input_layernorm.weight"] = Ones(h);
            w[$"{p}.post_attention_layernorm.weight"] = Ones(h);
            w[$"{p}.self_attn.q_proj.weight"] = F2(qDim, h);
            w[$"{p}.self_attn.k_proj.weight"] = F2(kvDim, h);
            w[$"{p}.self_attn.v_proj.weight"] = F2(kvDim, h);
            w[$"{p}.self_attn.o_proj.weight"] = F2(h, qDim);
            w[$"{p}.self_attn.q_proj.bias"] = F1(qDim);
            w[$"{p}.self_attn.k_proj.bias"] = F1(kvDim);
            w[$"{p}.self_attn.v_proj.bias"] = F1(kvDim);
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
        public int? EosId => 23;
        public IReadOnlyList<int> StopIds => [23];
        public string? BosToken => null;
        public string? EosToken => null;
    }

    private static GenerationRequest Req(int[] promptIds, int maxTokens, bool? specDecode = false) => new()
    {
        RawTokenIds = promptIds,
        MaxTokens = maxTokens,
        Sampling = SamplingOptions.Default with { Greedy = true },
        SpeculativeDecode = specDecode,
    };

    private sealed class Fixture : IDisposable
    {
        public readonly TransformerConfig Cfg;
        public readonly CpuBackend Backend = new();
        public readonly GenericTransformer Model;
        public readonly GenericTransformerModel Adapter;
        private readonly Dictionary<string, Tensor> _weights;

        public Fixture(uint seed)
        {
            _rng = seed;
            Cfg = GenericTransformerModelTests.Cfg();
            _weights = Weights(Cfg);
            Model = new GenericTransformer(Cfg);
            Model.LoadWeights(_weights, "model");
            Adapter = new GenericTransformerModel(Model, Backend);
        }

        public void Dispose()
        {
            Adapter.Dispose();
            Model.Dispose();
            Backend.Dispose();
            foreach (Tensor t in _weights.Values) t.Dispose();
        }
    }

    private static int[] Prompt(int length, uint seed, int vocab)
    {
        uint r = seed;
        int[] ids = new int[length];
        for (int i = 0; i < length; i++)
        {
            r ^= r << 13; r ^= r >> 17; r ^= r << 5;
            ids[i] = (int)(r % (uint)(vocab - 1));
        }
        return ids;
    }

    private static float[] Row(Tensor logits, int row, int vocab)
    {
        float[] values = new float[vocab];
        new ReadOnlySpan<float>((float*)logits.DataPointer + (long)row * vocab, vocab).CopyTo(values);
        return values;
    }

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= LogitTolerance,
                $"{what}[{i}]: {expected[i]} vs {actual[i]}");
    }

    private static float[] LastLogits(GenericTransformerModel model, Tensor hidden, int rows, int vocab)
    {
        using Tensor logits = model.ProjectLogits(hidden, rows);
        return Row(logits, rows - 1, vocab);
    }

    /// <summary>Plain greedy loop over GenericTransformer, the oracle for the adapter-driven pipeline.</summary>
    private static List<int> DirectGreedy(Fixture f, int[] prompt, int maxTokens)
    {
        int vocab = f.Cfg.VocabSize;
        List<int> generated = [];
        int capacity = prompt.Length + maxTokens + 1;
        using FixedKvCache cache = new(f.Cfg.NumLayers, 1, f.Cfg.NumKvHeads, f.Cfg.HeadDim, capacity);
        using Tensor hidden = f.Model.Forward(f.Backend, prompt, 0, cache);
        using Tensor logits = f.Model.ProjectLogits(f.Backend, hidden, prompt.Length);
        int next = Argmax(Row(logits, prompt.Length - 1, vocab));
        for (int step = 0; step < maxTokens; step++)
        {
            if (next == 23) break;
            generated.Add(next);
            using Tensor h = f.Model.Forward(f.Backend, [next], cache.CurrentLength, cache);
            using Tensor l = f.Model.ProjectLogits(f.Backend, h, 1);
            next = Argmax(Row(l, 0, vocab));
        }
        return generated;
    }

    private static int Argmax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }

    [Theory]
    [InlineData(3, 40, 0xA5A5u)]
    [InlineData(1, 60, 0xC001u)]
    [InlineData(9, 50, 0xF00Du)]
    public void Pipeline_OverAdapter_MatchesDirectGreedyLoop(int promptLen, int maxTokens, uint seed)
    {
        using Fixture f = new(seed);
        int[] prompt = Prompt(promptLen, seed, f.Cfg.VocabSize);
        string expected = string.Join(",", DirectGreedy(f, prompt, maxTokens));
        StubTokenizer tokenizer = new();

        GenerationResult viaAdapter = new TextGenerationPipeline(f.Adapter, tokenizer).Generate(Req(prompt, maxTokens));
        TextGenerationPipeline legacy = new(f.Model, tokenizer, f.Backend);
        GenerationResult viaLegacyCtor = legacy.Generate(Req(prompt, maxTokens));
        GenerationResult speculative = new TextGenerationPipeline(f.Adapter, tokenizer)
            .Generate(Req(prompt, maxTokens, specDecode: true));

        Assert.Equal(expected, string.Join(",", viaAdapter.TokenIds));
        Assert.Equal(expected, string.Join(",", viaLegacyCtor.TokenIds));
        Assert.Equal(expected, string.Join(",", speculative.TokenIds));
    }

    private static GenerationResult[] Await(Task<GenerationResult>[] tasks)
    {
        Task.WaitAll(tasks);
        return tasks.Select(t => t.Result).ToArray();
    }

    [Fact]
    public void Scheduler_OverAdapter_MatchesSequentialPipeline()
    {
        using Fixture f = new(0xBA7Cu);
        StubTokenizer tokenizer = new();
        int[][] prompts = [[1, 2, 3], [4], [5, 6, 7, 8, 9]];
        string[] reference = new string[prompts.Length];
        for (int i = 0; i < prompts.Length; i++)
        {
            TextGenerationPipeline pipeline = new(f.Adapter, tokenizer);
            reference[i] = string.Join(",", pipeline.Generate(Req(prompts[i], 6)).TokenIds);
        }

        using PagedKvPool pool = new(f.Cfg.NumLayers, f.Cfg.NumKvHeads, f.Cfg.HeadDim, pageSize: 4, maxPages: 64);
        using DynamicBatchScheduler scheduler = new(f.Adapter, tokenizer, pool);
        Task<GenerationResult>[] tasks = new Task<GenerationResult>[prompts.Length];
        for (int i = 0; i < prompts.Length; i++)
            tasks[i] = scheduler.SubmitAsync(Req(prompts[i], 6), null, CancellationToken.None);
        GenerationResult[] results = Await(tasks);

        for (int i = 0; i < prompts.Length; i++)
            Assert.Equal(reference[i], string.Join(",", results[i].TokenIds));
    }

    [Fact]
    public void Prefill_LastRowOnly_EqualsLastRowOfFullPrefill()
    {
        using Fixture f = new(0x1111u);
        int vocab = f.Cfg.VocabSize;
        int[] prompt = Prompt(7, 0x1111u, vocab);
        using ISequenceState full = f.Adapter.CreateSequenceState(new SequenceStateOptions(32));
        using ISequenceState lastOnly = f.Adapter.CreateSequenceState(new SequenceStateOptions(32));

        using Tensor fullHidden = f.Adapter.Prefill(new PrefillChunk(prompt, 0), full);
        using Tensor lastHidden = f.Adapter.Prefill(new PrefillChunk(prompt, 0, LastRowOnly: true), lastOnly);

        Assert.Equal(prompt.Length, full.Length);
        Assert.Equal(prompt.Length, lastOnly.Length);
        Assert.Equal(1, lastHidden.Shape[1]);
        float[] expected = LastLogits(f.Adapter, fullHidden, prompt.Length, vocab);
        AssertClose(expected, LastLogits(f.Adapter, lastHidden, 1, vocab), "last row");
    }

    [Fact]
    public void Prefill_InChunks_MatchesOneShot()
    {
        using Fixture f = new(0x2222u);
        int vocab = f.Cfg.VocabSize;
        int[] prompt = Prompt(9, 0x2222u, vocab);
        using ISequenceState oneShot = f.Adapter.CreateSequenceState(new SequenceStateOptions(32));
        using ISequenceState chunked = f.Adapter.CreateSequenceState(new SequenceStateOptions(32));

        using Tensor whole = f.Adapter.Prefill(new PrefillChunk(prompt, 0), oneShot);
        using (Tensor head = f.Adapter.Prefill(new PrefillChunk(prompt.AsMemory(0, 4), 0), chunked)) { }
        using Tensor tail = f.Adapter.Prefill(new PrefillChunk(prompt.AsMemory(4), 4), chunked);

        Assert.Equal(oneShot.Length, chunked.Length);
        float[] expected = LastLogits(f.Adapter, whole, prompt.Length, vocab);
        AssertClose(expected, LastLogits(f.Adapter, tail, 5, vocab), "chunked");
    }

    [Fact]
    public void Prefill_WithEmbedsOverride_MatchesTokenLookup()
    {
        using Fixture f = new(0x3333u);
        int vocab = f.Cfg.VocabSize;
        int[] prompt = Prompt(5, 0x3333u, vocab);
        using Tensor embeds = new(new TensorShape(1, prompt.Length, f.Cfg.HiddenSize), DType.F32);
        f.Model.EmbedLookup(embeds, prompt);
        using ISequenceState byTokens = f.Adapter.CreateSequenceState(new SequenceStateOptions(32));
        using ISequenceState byEmbeds = f.Adapter.CreateSequenceState(new SequenceStateOptions(32));

        using Tensor a = f.Adapter.Prefill(new PrefillChunk(prompt, 0), byTokens);
        using Tensor b = f.Adapter.Prefill(new PrefillChunk(prompt, 0, Embeds: embeds), byEmbeds);

        float[] expected = LastLogits(f.Adapter, a, prompt.Length, vocab);
        AssertClose(expected, LastLogits(f.Adapter, b, prompt.Length, vocab), "embeds");
    }

    [Fact]
    public void DecodeBatch_RaggedBatch_MatchesSingleSequenceSteps()
    {
        using Fixture f = new(0x4444u);
        int vocab = f.Cfg.VocabSize;
        int[][] prompts = [Prompt(3, 1, vocab), Prompt(6, 2, vocab), Prompt(1, 3, vocab)];
        int[] nextTokens = [5, 7, 11];
        using PagedKvPool pool = new(f.Cfg.NumLayers, f.Cfg.NumKvHeads, f.Cfg.HeadDim, pageSize: 4, maxPages: 64);
        ISequenceState[] batched = new ISequenceState[prompts.Length];
        ISequenceState[] single = new ISequenceState[prompts.Length];
        try
        {
            for (int i = 0; i < prompts.Length; i++)
            {
                batched[i] = f.Adapter.CreateSequenceState(new SequenceStateOptions(32, pool));
                single[i] = f.Adapter.CreateSequenceState(new SequenceStateOptions(32));
                f.Adapter.Prefill(new PrefillChunk(prompts[i], 0, LastRowOnly: true), batched[i]).Dispose();
                f.Adapter.Prefill(new PrefillChunk(prompts[i], 0, LastRowOnly: true), single[i]).Dispose();
            }

            using Tensor batchHidden = f.Adapter.DecodeBatch(nextTokens, batched);
            using Tensor batchLogits = f.Adapter.ProjectLogits(batchHidden, prompts.Length);
            for (int i = 0; i < prompts.Length; i++)
            {
                PrefillChunk step = new(new[] { nextTokens[i] }, single[i].Length);
                using Tensor h = f.Adapter.Prefill(step, single[i]);
                AssertClose(LastLogits(f.Adapter, h, 1, vocab), Row(batchLogits, i, vocab), $"seq {i}");
                Assert.Equal(prompts[i].Length + 1, batched[i].Length);
            }
        }
        finally
        {
            foreach (ISequenceState s in batched) s?.Dispose();
            foreach (ISequenceState s in single) s?.Dispose();
        }
    }

    [Fact]
    public void CreateSequenceState_SelectsCacheKindAndReportsCapacity()
    {
        using Fixture f = new(0x5555u);
        using PagedKvPool pool = new(f.Cfg.NumLayers, f.Cfg.NumKvHeads, f.Cfg.HeadDim, pageSize: 4, maxPages: 8);

        using ISequenceState own = f.Adapter.CreateSequenceState(new SequenceStateOptions(20));
        using ISequenceState paged = f.Adapter.CreateSequenceState(new SequenceStateOptions(20, pool));
        using ISequenceState full = f.Adapter.CreateSequenceState(new SequenceStateOptions(20, FullPrecisionKv: true));

        Assert.IsType<FixedKvCache>(own);
        Assert.IsType<PagedKvCache>(paged);
        Assert.IsType<FixedKvCache>(full);
        Assert.Equal(20, own.Capacity);
        Assert.Equal(32, paged.Capacity);
        Assert.Equal(0, own.Length);
        Assert.Equal(f.Cfg.VocabSize, f.Adapter.Info.VocabSize);
        Assert.True(f.Adapter.Capabilities.SupportsSpeculation);
        Assert.True(f.Adapter.Capabilities.SupportsBatchDecode);
        Assert.False(f.Adapter.Capabilities.NeedsTokenIdsWithEmbeds);
        Assert.Equal(f.Cfg.NumLayers * 2L * f.Cfg.NumKvHeads * f.Cfg.HeadDim * 4 * 10,
            f.Adapter.EstimateSequenceBytes(10) * (KvCaches.F16Enabled ? 2 : 1));
        Assert.Same(f.Backend, f.Adapter.OutputBackend);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Checkpoint_Rollback_RestoresSequenceExactly(bool paged)
    {
        using Fixture f = new(0x6666u);
        int vocab = f.Cfg.VocabSize;
        int[] prompt = Prompt(5, 0x6666u, vocab);
        int[] more = [3, 9, 14];
        using PagedKvPool pool = new(f.Cfg.NumLayers, f.Cfg.NumKvHeads, f.Cfg.HeadDim, pageSize: 4, maxPages: 16);
        using ISequenceState state = f.Adapter.CreateSequenceState(new SequenceStateOptions(32, paged ? pool : null));

        f.Adapter.Prefill(new PrefillChunk(prompt, 0), state).Dispose();
        SequenceCheckpoint checkpoint = state.Checkpoint();
        Assert.Equal(prompt.Length, checkpoint.Length);
        Assert.Equal(prompt.Length, state.MaxRollback);

        float[] first;
        using (Tensor h = f.Adapter.Prefill(new PrefillChunk(more, state.Length), state))
            first = LastLogits(f.Adapter, h, more.Length, vocab);
        Assert.Equal(prompt.Length + more.Length, state.Length);

        state.Rollback(checkpoint);
        Assert.Equal(prompt.Length, state.Length);

        using Tensor replay = f.Adapter.Prefill(new PrefillChunk(more, state.Length), state);
        AssertClose(first, LastLogits(f.Adapter, replay, more.Length, vocab), "replay after rollback");
        Assert.Equal(prompt.Length + more.Length, state.Length);

        state.Reset();
        Assert.Equal(0, state.Length);
    }

    [Fact]
    public void KvCache_DefaultSequenceStateMappings_ArePassiveAndRefuseRollback()
    {
        TransformerConfig cfg = Cfg();
        using KvCache cache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim);
        ISequenceState state = cache;

        Assert.Equal(cache.CurrentLength, state.Length);
        Assert.Equal(int.MaxValue, state.Capacity);
        Assert.Equal(0, state.MaxRollback);
        Assert.Equal(new SequenceCheckpoint(0), state.Checkpoint());
        Assert.Throws<NotSupportedException>(() => state.Rollback(new SequenceCheckpoint(0)));
    }

    [Fact]
    public void RopeTables_BuildRope_MatchesClosedForm()
    {
        const int headDim = 8;
        const float theta = 10000f;
        using Tensor cos = new(new TensorShape(3, headDim), DType.F32);
        using Tensor sin = new(new TensorShape(3, headDim), DType.F32);

        RopeTables.BuildRope(cos, sin, 3, posStart: 5, headDim, rotaryDim: 0, theta, RopeScaling.None);

        float* pc = (float*)cos.DataPointer;
        float* ps = (float*)sin.DataPointer;
        for (int s = 0; s < 3; s++)
        {
            for (int i = 0; i < headDim / 2; i++)
            {
                double angle = (5 + s) * Math.Pow(theta, -2.0 * i / headDim);
                Assert.Equal((float)Math.Cos(angle), pc[s * headDim + i], 5);
                Assert.Equal(pc[s * headDim + i], pc[s * headDim + i + headDim / 2]);
                Assert.Equal((float)Math.Sin(angle), ps[s * headDim + i], 5);
                Assert.Equal(ps[s * headDim + i], ps[s * headDim + i + headDim / 2]);
            }
        }
    }

    [Fact]
    public void RopeTables_Batched_RowsMatchSequentialBuild()
    {
        const int headDim = 8;
        int[] positions = [9, 0, 4];
        using Tensor cosB = new(new TensorShape(3, headDim), DType.F32);
        using Tensor sinB = new(new TensorShape(3, headDim), DType.F32);
        using Tensor cos1 = new(new TensorShape(1, headDim), DType.F32);
        using Tensor sin1 = new(new TensorShape(1, headDim), DType.F32);

        RopeTables.BuildRopeBatched(cosB, sinB, positions, headDim, headDim, 10000f, RopeScaling.None);

        for (int b = 0; b < positions.Length; b++)
        {
            RopeTables.BuildRope(cos1, sin1, 1, positions[b], headDim, headDim, 10000f, RopeScaling.None);
            for (int i = 0; i < headDim; i++)
            {
                Assert.Equal(((float*)cos1.DataPointer)[i], ((float*)cosB.DataPointer)[b * headDim + i]);
                Assert.Equal(((float*)sin1.DataPointer)[i], ((float*)sinB.DataPointer)[b * headDim + i]);
            }
        }
    }

    [Fact]
    public void RopeTables_InverseAndTailVariants()
    {
        const int headDim = 8;
        const int rotary = 4;
        const int offset = 4;
        using Tensor cosF = new(new TensorShape(2, headDim), DType.F32);
        using Tensor sinF = new(new TensorShape(2, headDim), DType.F32);
        using Tensor cosI = new(new TensorShape(2, headDim), DType.F32);
        using Tensor sinI = new(new TensorShape(2, headDim), DType.F32);
        using Tensor cosT = new(new TensorShape(2, headDim), DType.F32);
        using Tensor sinT = new(new TensorShape(2, headDim), DType.F32);
        new Span<float>((float*)cosT.DataPointer, 2 * headDim).Fill(7f);
        new Span<float>((float*)sinT.DataPointer, 2 * headDim).Fill(7f);

        RopeTables.BuildRope(cosF, sinF, 2, 3, headDim, rotary, 10000f, RopeScaling.None);
        RopeTables.BuildRopeInverse(cosI, sinI, 2, 3, headDim, rotary, 10000f, RopeScaling.None);
        RopeTables.BuildRopeTail(cosT, sinT, 2, 3, headDim, rotary, offset, inverse: false, 10000f, RopeScaling.None);

        for (int s = 0; s < 2; s++)
        {
            for (int i = 0; i < rotary; i++)
            {
                int at = s * headDim + i;
                Assert.Equal(((float*)cosF.DataPointer)[at], ((float*)cosI.DataPointer)[at]);
                Assert.Equal(-((float*)sinF.DataPointer)[at], ((float*)sinI.DataPointer)[at]);
                Assert.Equal(((float*)cosF.DataPointer)[at], ((float*)cosT.DataPointer)[at + offset]);
                Assert.Equal(((float*)sinF.DataPointer)[at], ((float*)sinT.DataPointer)[at + offset]);
                Assert.Equal(7f, ((float*)cosT.DataPointer)[at]);
            }
        }
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            RopeTables.BuildRopeTail(cosT, sinT, 2, 0, headDim, rotary, offset + 1, false, 10000f, RopeScaling.None));
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(1.5f)]
    public void GatedFfn_SwiGlu_MatchesManualSiluTimesUp(float clamp)
    {
        _rng = 0x7777u;
        using CpuBackend backend = new();
        const int n = 40;
        Tensor gate = new(new TensorShape(1, 1, n), DType.F32);
        Tensor up = new(new TensorShape(1, 1, n), DType.F32);
        float[] g = new float[n];
        float[] u = new float[n];
        for (int i = 0; i < n; i++) { g[i] = (Rand() - 0.02f) * 40f; u[i] = Rand() * 40f; }
        g.CopyTo(new Span<float>((float*)gate.DataPointer, n));
        u.CopyTo(new Span<float>((float*)up.DataPointer, n));

        using Tensor result = GatedFfn.SwiGlu(backend, ActivationKind.Silu, gate, up, clamp);

        for (int i = 0; i < n; i++)
        {
            float gv = clamp > 0f ? MathF.Min(g[i], clamp) : g[i];
            float uv = clamp > 0f ? Math.Clamp(u[i], -clamp, clamp) : u[i];
            float expected = gv / (1f + MathF.Exp(-gv)) * uv;
            Assert.Equal(expected, ((float*)result.DataPointer)[i], 4);
        }
    }

    private class RecordingBackend : DispatchProxy
    {
        public bool Flag;
        public readonly List<bool> FlagAtLinear = [];
        public bool ThrowOnLinear;

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            switch (targetMethod!.Name)
            {
                case "get_HighPrecisionGemm": return Flag;
                case "set_HighPrecisionGemm": Flag = (bool)args![0]!; return null;
                case "Linear":
                    FlagAtLinear.Add(Flag);
                    if (ThrowOnLinear) throw new InvalidOperationException("boom");
                    return null;
                default: return null;
            }
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ProjectionOps_ProjectPrecise_ForcesHighPrecisionAndRestores(bool throws)
    {
        IBackend backend = DispatchProxy.Create<IBackend, RecordingBackend>();
        RecordingBackend probe = (RecordingBackend)(object)backend;
        probe.ThrowOnLinear = throws;
        using Tensor t = new(new TensorShape(1, 1, 2), DType.F32);

        if (throws)
        {
            Assert.Throws<InvalidOperationException>(
                () => ProjectionOps.ProjectPrecise(backend, t, t, t, null, lowVram: false));
        }
        else
        {
            ProjectionOps.ProjectPrecise(backend, t, t, t, null, lowVram: false);
        }

        Assert.Equal([true], probe.FlagAtLinear);
        Assert.False(probe.Flag);

        probe.ThrowOnLinear = false;
        probe.Flag = true;
        ProjectionOps.Project(backend, t, t, t, null, lowVram: false);
        Assert.Equal([true, true], probe.FlagAtLinear.Take(2));
    }
}
