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

    [Theory]
    [InlineData(false)]
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

    [Theory]
    [InlineData(0f)]
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
