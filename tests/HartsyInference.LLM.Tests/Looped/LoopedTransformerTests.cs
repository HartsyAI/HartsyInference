using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Looped;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests.Looped;

/// <summary>The backend-composed looped forward against the scalar reference (<see cref="LoopedReference"/>), its cache semantics across chunks in both
/// cache modes, batched decode against single decode, and the generation adapter's loop-count handling.</summary>
public sealed unsafe class LoopedTransformerTests
{
    private const double Tolerance = 5e-4;

    private sealed class Fixture : IDisposable
    {
        private uint _rng;

        public readonly LoopedTransformerConfig Cfg;
        public readonly CpuBackend Backend = new();
        public readonly Dictionary<string, float[]> Arrays = new(StringComparer.Ordinal);
        public readonly Dictionary<string, Tensor> Weights = new(StringComparer.Ordinal);
        public readonly LoopedTransformer Model;
        public readonly LoopedTransformerModel Adapter;

        public Fixture(LoopedKvCacheMode mode, uint seed = 0x9E3779B9u, int loops = 2)
        {
            _rng = seed;
            Cfg = new LoopedTransformerConfig
            {
                HiddenSize = 16, NumHeads = 2, NumKvHeads = 1, HeadDim = 8, IntermediateSize = 24, VocabSize = 20, MaxPositionEmbeddings = 64,
                InputLayers = 2, RecurrentLayers = 2, OutputLayers = 2, Loops = loops,
                LayerKinds = [LoopedLayerKind.SlidingAttention, LoopedLayerKind.Mlp, LoopedLayerKind.FullAttention, LoopedLayerKind.Mlp,
                    LoopedLayerKind.SlidingAttention, LoopedLayerKind.Mlp],
                RopeThetas = [10_000f, 0f, 100_000f, 0f, 10_000f, 0f],
                SlidingWindow = 3, KvCacheMode = mode, InitStd = 0.02f, AlphaInit = 0.05f, Architecture = "dory", LowVramQuant = false,
            };
            int h = Cfg.HiddenSize, nq = Cfg.QDim, nk = Cfg.KvDim, ff = Cfg.IntermediateSize, v = Cfg.VocabSize, d = Cfg.HeadDim;
            Matrix("model.embed_tokens.weight", v, h);
            Matrix("lm_head.weight", v, h);
            Vector("lm_head.logit_scale.weight", v, Cfg.InitStd, 0.5f);
            for (int p = 0; p < Cfg.PhysicalLayers; p++)
            {
                string prefix = $"model.layers.{p}";
                Vector($"{prefix}.residual_alpha.weight", h, Cfg.InitStd / Cfg.AlphaInit * 0.5f, 0.8f);
                if (Cfg.IsAttention(p))
                {
                    Matrix($"{prefix}.self_attn.q_proj.weight", nq, h);
                    Matrix($"{prefix}.self_attn.k_proj.weight", nk, h);
                    Matrix($"{prefix}.self_attn.v_proj.weight", nk, h);
                    Matrix($"{prefix}.self_attn.o_proj.weight", h, nq);
                    Matrix($"{prefix}.self_attn.gate_proj.weight", nq, h);
                    Vector($"{prefix}.self_attn.qk_scale.weight", nq, Cfg.InitStd / MathF.Sqrt(d), 0.3f);
                    Vector($"{prefix}.self_attn.gate_scale.weight", nq, Cfg.InitStd / MathF.Sqrt(Cfg.EffectiveGateInit), 0.3f);
                }
                else
                {
                    Matrix($"{prefix}.mlp.gate_proj.weight", ff, h);
                    Matrix($"{prefix}.mlp.up_proj.weight", ff, h);
                    Matrix($"{prefix}.mlp.down_proj.weight", h, ff);
                    Vector($"{prefix}.mlp.gate_scale.weight", ff, 1f / MathF.Sqrt(h), 0.3f);
                    Vector($"{prefix}.mlp.up_scale.weight", ff, 1f / MathF.Sqrt(h), 0.3f);
                }
            }
            Model = new LoopedTransformer(Cfg);
            Model.LoadWeights(Weights, "model");
            Adapter = new LoopedTransformerModel(Model, Backend);
        }

        public LoopedReference Reference() => new(Cfg, Arrays);

        private float Rand()
        {
            _rng ^= _rng << 13;
            _rng ^= _rng >> 17;
            _rng ^= _rng << 5;
            return (_rng & 0xFFFF) / 65535f - 0.5f;
        }

        /// <summary>A <c>[rows, cols]</c> weight with entries of order <c>1/sqrt(cols)</c>, so projections of unit-norm inputs stay O(1).</summary>
        private void Matrix(string key, int rows, int cols)
        {
            float[] a = new float[rows * cols];
            float scale = 2f / MathF.Sqrt(cols);
            for (int i = 0; i < a.Length; i++) a[i] = Rand() * scale;
            Store(key, a, new TensorShape(rows, cols));
        }

        /// <summary>A stored scale vector around <paramref name="center"/> with relative spread <paramref name="spread"/>.</summary>
        private void Vector(string key, int n, float center, float spread)
        {
            float[] a = new float[n];
            for (int i = 0; i < n; i++) a[i] = center * (1f + spread * 2f * Rand());
            Store(key, a, new TensorShape(n));
        }

        private void Store(string key, float[] values, TensorShape shape)
        {
            Tensor t = new(shape, DType.F32);
            values.AsSpan().CopyTo(t.AsSpan<float>());
            Arrays[key] = values;
            Weights[key] = t;
        }

        public void Dispose()
        {
            Adapter.Dispose();
            Model.Dispose();
            Backend.Dispose();
            foreach (Tensor t in Weights.Values) t.Dispose();
        }
    }

    private sealed class StubTokenizer : ILlmTokenizer
    {
        public int[] Encode(string text, bool addSpecial) => throw new NotSupportedException();
        public int[] EncodeOrdinary(string text) => throw new NotSupportedException();
        public string Decode(IReadOnlyList<int> ids) => string.Join(",", ids);
        public int? SpecialId(string token) => null;
        public int? BosId => null;
        public int? EosId => 19;
        public IReadOnlyList<int> StopIds => [19];
        public string? BosToken => null;
        public string? EosToken => null;
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

    private static void AssertClose(double[] expected, float[] actual, string what, double tolerance = Tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance, $"{what}[{i}]: expected {expected[i]}, got {actual[i]}");
    }

    private static void AssertClose(float[] expected, float[] actual, string what, double tolerance = Tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance, $"{what}[{i}]: expected {expected[i]}, got {actual[i]}");
    }

    private static LoopedSequenceState NewState(Fixture f, int capacity, int loops) =>
        new(new FixedKvCache(LoopedSchedule.MaxSlotCount(f.Cfg), 1, f.Cfg.NumKvHeads, f.Cfg.HeadDim, capacity), loops);

    [Theory]
    [InlineData(LoopedKvCacheMode.PerLoop)]
    [InlineData(LoopedKvCacheMode.LastLoop)]
    public void Prefill_MatchesTheScalarReference(LoopedKvCacheMode mode)
    {
        using Fixture f = new(mode);
        int[] prompt = Prompt(7, 0x1234u, f.Cfg.VocabSize);
        double[][] expected = f.Reference().ForwardChunk(prompt, 0, loops: 2);
        using LoopedSequenceState state = NewState(f, 16, 2);
        using Tensor hidden = f.Model.Forward(f.Backend, prompt, 0, state, 2);
        using Tensor logits = f.Model.ProjectLogits(f.Backend, hidden, prompt.Length);

        Assert.Equal(prompt.Length, state.CurrentLength);
        for (int i = 0; i < prompt.Length; i++) AssertClose(expected[i], Row(logits, i, f.Cfg.VocabSize), $"row {i}");
    }

    [Theory]
    [InlineData(LoopedKvCacheMode.PerLoop)]
    [InlineData(LoopedKvCacheMode.LastLoop)]
    public void PrefillThenDecode_MatchesTheChunkedReference(LoopedKvCacheMode mode)
    {
        using Fixture f = new(mode, seed: 0x7777u);
        int vocab = f.Cfg.VocabSize;
        int[] prompt = Prompt(7, 0x4444u, vocab);
        LoopedReference reference = f.Reference();
        reference.ForwardChunk(prompt[..4], 0, 2);
        double[] expected5 = reference.ForwardChunk([prompt[4]], 4, 2)[0];
        double[] expected6 = reference.ForwardChunk([prompt[5]], 5, 2)[0];
        double[] expected7 = reference.ForwardChunk([prompt[6]], 6, 2)[0];

        using LoopedSequenceState state = NewState(f, 16, 2);
        f.Model.Forward(f.Backend, prompt[..4], 0, state, 2).Dispose();
        float[] got5, got6, got7;
        using (Tensor h = f.Model.Forward(f.Backend, [prompt[4]], 4, state, 2))
        using (Tensor l = f.Model.ProjectLogits(f.Backend, h, 1)) got5 = Row(l, 0, vocab);
        using (Tensor h = f.Model.Forward(f.Backend, [prompt[5]], 5, state, 2))
        using (Tensor l = f.Model.ProjectLogits(f.Backend, h, 1)) got6 = Row(l, 0, vocab);
        using (Tensor h = f.Model.Forward(f.Backend, [prompt[6]], 6, state, 2))
        using (Tensor l = f.Model.ProjectLogits(f.Backend, h, 1)) got7 = Row(l, 0, vocab);

        Assert.Equal(7, state.CurrentLength);
        AssertClose(expected5, got5, "decode 5");
        AssertClose(expected6, got6, "decode 6");
        AssertClose(expected7, got7, "decode 7");
    }

    [Fact]
    public void PerLoop_IsChunkInvariant_AndLastLoop_IsNot()
    {
        using Fixture perLoop = new(LoopedKvCacheMode.PerLoop, seed: 0x3131u);
        using Fixture lastLoop = new(LoopedKvCacheMode.LastLoop, seed: 0x3131u);
        int vocab = perLoop.Cfg.VocabSize;
        int[] prompt = Prompt(6, 0x5151u, vocab);

        static float[] OneShotLast(Fixture f, int[] prompt)
        {
            using LoopedSequenceState state = NewState(f, 16, 2);
            using Tensor h = f.Model.Forward(f.Backend, prompt, 0, state, 2);
            using Tensor l = f.Model.ProjectLogits(f.Backend, h, prompt.Length);
            return Row(l, prompt.Length - 1, f.Cfg.VocabSize);
        }

        static float[] ChunkedLast(Fixture f, int[] prompt)
        {
            using LoopedSequenceState state = NewState(f, 16, 2);
            f.Model.Forward(f.Backend, prompt[..3], 0, state, 2).Dispose();
            using Tensor h = f.Model.Forward(f.Backend, prompt[3..], 3, state, 2);
            using Tensor l = f.Model.ProjectLogits(f.Backend, h, prompt.Length - 3);
            return Row(l, prompt.Length - 4, f.Cfg.VocabSize);
        }

        // Same weights: a single-chunk prefill cannot tell the modes apart (every loop rewrites the whole chunk).
        AssertClose(OneShotLast(perLoop, prompt), OneShotLast(lastLoop, prompt), "modes on one chunk");
        // Per-loop keys are loop-specific whatever the chunking; last-loop keys of earlier chunks come from the final pass.
        AssertClose(OneShotLast(perLoop, prompt), ChunkedLast(perLoop, prompt), "per-loop chunking");
        float[] oneShot = OneShotLast(lastLoop, prompt);
        float[] chunked = ChunkedLast(lastLoop, prompt);
        Assert.True(oneShot.Zip(chunked).Any(pair => Math.Abs(pair.First - pair.Second) > 1e-3), "last-loop output should depend on chunking");
    }

    [Fact]
    public void LoopCount_ChangesTheOutput()
    {
        using Fixture f = new(LoopedKvCacheMode.PerLoop);
        int[] prompt = Prompt(5, 0x2222u, f.Cfg.VocabSize);
        float[] one, two;
        using (LoopedSequenceState state = NewState(f, 16, 1))
        using (Tensor h = f.Model.Forward(f.Backend, prompt, 0, state, 1))
        using (Tensor l = f.Model.ProjectLogits(f.Backend, h, prompt.Length)) one = Row(l, prompt.Length - 1, f.Cfg.VocabSize);
        using (LoopedSequenceState state = NewState(f, 16, 2))
        using (Tensor h = f.Model.Forward(f.Backend, prompt, 0, state, 2))
        using (Tensor l = f.Model.ProjectLogits(f.Backend, h, prompt.Length)) two = Row(l, prompt.Length - 1, f.Cfg.VocabSize);

        Assert.True(one.Zip(two).Any(pair => Math.Abs(pair.First - pair.Second) > 1e-3), "a second pass should move the logits");
        AssertClose(f.Reference().ForwardChunk(prompt, 0, 1)[^1], one, "one loop");
    }

    [Fact]
    public void BatchDecode_MatchesSingleDecode()
    {
        using Fixture f = new(LoopedKvCacheMode.PerLoop, seed: 0xABCDu);
        int vocab = f.Cfg.VocabSize;
        int[] promptA = Prompt(5, 0x11u, vocab);
        int[] promptB = Prompt(3, 0x22u, vocab);
        using LoopedSequenceState soloA = NewState(f, 16, 2);
        using LoopedSequenceState soloB = NewState(f, 16, 2);
        using LoopedSequenceState batchA = NewState(f, 16, 2);
        using LoopedSequenceState batchB = NewState(f, 16, 2);
        foreach ((LoopedSequenceState state, int[] prompt) in new[] { (soloA, promptA), (soloB, promptB), (batchA, promptA), (batchB, promptB) })
            f.Model.Forward(f.Backend, prompt, 0, state, 2).Dispose();

        float[] expectedA, expectedB;
        using (Tensor h = f.Model.Forward(f.Backend, [7], 5, soloA, 2))
        using (Tensor l = f.Model.ProjectLogits(f.Backend, h, 1)) expectedA = Row(l, 0, vocab);
        using (Tensor h = f.Model.Forward(f.Backend, [9], 3, soloB, 2))
        using (Tensor l = f.Model.ProjectLogits(f.Backend, h, 1)) expectedB = Row(l, 0, vocab);

        using Tensor embeds = new(new TensorShape(1, 2, f.Cfg.HiddenSize), DType.F32);
        f.Model.EmbedLookup(embeds, [7, 9]);
        using Tensor hidden = f.Model.ForwardBatchDecode(f.Backend, embeds, [5, 3], [batchA, batchB], 2);
        using Tensor logits = f.Model.ProjectLogits(f.Backend, hidden, 2);

        AssertClose(expectedA, Row(logits, 0, vocab), "batched A");
        AssertClose(expectedB, Row(logits, 1, vocab), "batched B");
        Assert.Equal(6, batchA.CurrentLength);
        Assert.Equal(4, batchB.CurrentLength);
    }

    [Fact]
    public void Adapter_ResolvesLoopsAndGuardsRetainedStates()
    {
        using Fixture f = new(LoopedKvCacheMode.PerLoop);
        using ISequenceState byDefault = f.Adapter.CreateSequenceState(new SequenceStateOptions(8));
        using ISequenceState explicitThree = f.Adapter.CreateSequenceState(new SequenceStateOptions(8, null, 3));

        Assert.Equal(2, Assert.IsType<LoopedSequenceState>(byDefault).Loops);
        Assert.Equal(3, Assert.IsType<LoopedSequenceState>(explicitThree).Loops);
        Assert.Equal(LoopedSchedule.MaxSlotCount(f.Cfg), ((LoopedSequenceState)byDefault).NumLayers);
        Assert.Throws<ArgumentOutOfRangeException>(() => f.Adapter.CreateSequenceState(new SequenceStateOptions(8, null, 5)));
        Assert.True(f.Adapter.AcceptsRetainedState(byDefault, new SequenceStateOptions(8)));
        Assert.True(f.Adapter.AcceptsRetainedState(byDefault, new SequenceStateOptions(8, null, 2)));
        Assert.False(f.Adapter.AcceptsRetainedState(byDefault, new SequenceStateOptions(8, null, 3)));
        Assert.False(f.Adapter.AcceptsRetainedState(explicitThree, new SequenceStateOptions(8)));
        Assert.True(f.Adapter.EstimateSequenceBytes(10) > 0);
    }

    [Fact]
    public void Adapter_DecodeBatch_GroupsMixedLoopCounts()
    {
        using Fixture f = new(LoopedKvCacheMode.PerLoop, seed: 0x5A5Au);
        int vocab = f.Cfg.VocabSize;
        int[] promptA = Prompt(4, 0x31u, vocab);
        int[] promptB = Prompt(4, 0x32u, vocab);
        using ISequenceState soloA = f.Adapter.CreateSequenceState(new SequenceStateOptions(16, null, 1));
        using ISequenceState soloB = f.Adapter.CreateSequenceState(new SequenceStateOptions(16, null, 3));
        using ISequenceState mixedA = f.Adapter.CreateSequenceState(new SequenceStateOptions(16, null, 1));
        using ISequenceState mixedB = f.Adapter.CreateSequenceState(new SequenceStateOptions(16, null, 3));
        foreach ((ISequenceState state, int[] prompt) in new[] { (soloA, promptA), (soloB, promptB), (mixedA, promptA), (mixedB, promptB) })
            f.Adapter.Prefill(new PrefillChunk(prompt, 0), state).Dispose();

        float[] expectedA, expectedB;
        using (Tensor h = f.Adapter.Prefill(new PrefillChunk(new[] { 5 }, 4), soloA))
        using (Tensor l = f.Adapter.ProjectLogits(h, 1)) expectedA = Row(l, 0, vocab);
        using (Tensor h = f.Adapter.Prefill(new PrefillChunk(new[] { 6 }, 4), soloB))
        using (Tensor l = f.Adapter.ProjectLogits(h, 1)) expectedB = Row(l, 0, vocab);

        using Tensor hidden = f.Adapter.DecodeBatch([5, 6], [mixedA, mixedB]);
        using Tensor logits = f.Adapter.ProjectLogits(hidden, 2);

        AssertClose(expectedA, Row(logits, 0, vocab), "mixed A (1 loop)");
        AssertClose(expectedB, Row(logits, 1, vocab), "mixed B (3 loops)");
        Assert.Equal(5, mixedA.Length);
        Assert.Equal(5, mixedB.Length);
    }

    [Fact]
    public void Pipeline_GreedyAndSpeculative_AgreeAndHonorLoops()
    {
        using Fixture f = new(LoopedKvCacheMode.PerLoop, seed: 0xC0DEu);
        int[] prompt = Prompt(4, 0x77u, f.Cfg.VocabSize);
        StubTokenizer tokenizer = new();
        static GenerationRequest Req(int[] prompt, int loops, bool speculative) => new()
        {
            RawTokenIds = prompt, MaxTokens = 6, Loops = loops, SpeculativeDecode = speculative,
            Sampling = SamplingOptions.Default with { Greedy = true },
        };

        GenerationResult greedy2 = new TextGenerationPipeline(f.Adapter, tokenizer).Generate(Req(prompt, 2, false));
        GenerationResult speculative2 = new TextGenerationPipeline(f.Adapter, tokenizer).Generate(Req(prompt, 2, true));
        GenerationResult greedy1 = new TextGenerationPipeline(f.Adapter, tokenizer).Generate(Req(prompt, 1, false));

        Assert.NotEmpty(greedy2.TokenIds);
        Assert.Equal(greedy2.TokenIds, speculative2.TokenIds);
        // A direct greedy loop at two passes is the oracle for the adapter-driven pipeline.
        List<int> direct = [];
        using (LoopedSequenceState state = NewState(f, 16, 2))
        {
            using Tensor h0 = f.Model.Forward(f.Backend, prompt, 0, state, 2);
            using Tensor l0 = f.Model.ProjectLogits(f.Backend, h0, prompt.Length);
            int next = Argmax(Row(l0, prompt.Length - 1, f.Cfg.VocabSize));
            for (int step = 0; step < 6 && next != 19; step++)
            {
                direct.Add(next);
                using Tensor h = f.Model.Forward(f.Backend, [next], state.CurrentLength, state, 2);
                using Tensor l = f.Model.ProjectLogits(f.Backend, h, 1);
                next = Argmax(Row(l, 0, f.Cfg.VocabSize));
            }
        }
        Assert.Equal(direct, greedy2.TokenIds);
        Assert.True(greedy1.TokenIds.Count > 0);
    }

    private static int Argmax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }
}
