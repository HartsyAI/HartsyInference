using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>The prompt prefill stops between transformer layers once the request's token is cancelled. A cancel that
/// lands while layer k writes its K/V ends the stream with a Cancelled stop before layer k+1 touches the cache, on the
/// plain and the layer-split path; nothing is committed or retained; and the next request on the same pipeline matches
/// a fresh one. A live token that is never cancelled changes no bit of the prefill. CPU backend, synthetic weights.</summary>
public sealed class PrefillCancellationTests
{
    private const int Layers = 4;
    private static readonly SamplingOptions Greedy = SamplingOptions.Default with { Greedy = true };

    private static uint _rng = 0x9E3779B9u;
    private static uint NextRaw() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return _rng; }
    private static unsafe Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = ((NextRaw() & 0xFFFF) / 65535f - 0.5f) * 0.2f; return t; }
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static unsafe Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); new Span<float>((float*)t.DataPointer, n).Fill(1f); return t; }

    private static TransformerConfig Cfg() => new()
    {
        HiddenSize = 16, NumLayers = Layers, NumHeads = 4, NumKvHeads = 2, HeadDim = 4,
        IntermediateSize = 32, VocabSize = 37, MaxPositionEmbeddings = 512, AttentionBias = true, QkNorm = false,
    };

    private static Dictionary<string, Tensor> Weights(TransformerConfig c)
    {
        int h = c.HiddenSize, qDim = c.QDim, kvDim = c.KvDim;
        Dictionary<string, Tensor> w = new() { ["model.embed_tokens.weight"] = F2(c.VocabSize, h), ["model.norm.weight"] = Ones(h) };
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

    [Fact]
    public async Task ACancelDuringThePromptPrefill_EndsTheStreamCancelledBeforeTheNextLayer_AndTheNextRequestIsCorrect()
    {
        using Fixture f = new(0xCA11u);
        GenerationRequest request = new() { RawTokenIds = f.Prompt(24), MaxTokens = 6, Sampling = Greedy };
        using CancellationTokenSource caller = new();
        f.Model.CancelAt(layer: 1, caller);

        List<TextChunk> chunks = [];
        await foreach (TextChunk chunk in TextStreamPump.Run((_, token) => Produce(f.Pipeline, request, reuse: null, token), caller.Token))
        {
            chunks.Add(chunk);
        }

        TextChunk only = Assert.Single(chunks);
        Assert.Equal(TextChunkKind.StopReason, only.Kind);
        Assert.Equal(StopReason.Cancelled, only.Stop);
        Assert.Equal([0, 1], f.Model.PrefillLayers);
        RecordingKvCache cache = Assert.Single(f.Model.Caches);
        Assert.Equal(0, cache.CommittedAtDispose);

        f.Model.Reset();
        GenerationResult next = f.Pipeline.Generate(request);
        Assert.Equal(f.Reference(request), string.Join(",", next.TokenIds));
        Assert.Equal(Enumerable.Range(0, Layers), f.Model.PrefillLayers);
    }

    [Fact]
    public void ACancelDuringALaterTurnsPrefill_DropsTheRetainedSequence_AndTheNextTurnRunsUncached()
    {
        using Fixture f = new(0xBEEFu);
        using RetainedSequence retained = new();
        int[] first = f.Prompt(12);
        GenerationResult turn1 = f.Pipeline.Generate(
            new GenerationRequest { RawTokenIds = first, MaxTokens = 4, Sampling = Greedy, PrefixCacheCapacityHint = 128 }, retained);
        Assert.NotNull(retained.Cache);

        int[] second = [.. first, .. turn1.TokenIds, .. f.Prompt(6)];
        GenerationRequest request = new() { RawTokenIds = second, MaxTokens = 4, Sampling = Greedy, PrefixCacheCapacityHint = 128 };
        using CancellationTokenSource caller = new();
        f.Model.Reset();
        f.Model.CancelAt(layer: 2, caller);
        Assert.Throws<OperationCanceledException>(() => f.Pipeline.Generate(request, retained, ct: caller.Token));

        // The reused prefix was already in the cache, so only the new suffix was being prefilled, and it stopped
        // before layer 3. The whole entry is dropped rather than kept with a prompt the cache never received.
        Assert.Equal([0, 1, 2], f.Model.PrefillLayers);
        Assert.Null(retained.Cache);
        Assert.Empty(retained.TokenIds);

        f.Model.Reset();
        GenerationResult turn2 = f.Pipeline.Generate(request, retained);
        Assert.Equal(f.Reference(request), string.Join(",", turn2.TokenIds));
        Assert.Equal(0, turn2.ReusedPromptTokens);
    }

    [Fact]
    public void ACancelDuringALayerSplitPrefill_StopsInsideTheSecondStage_AndItsRowsAreNeverRead()
    {
        using Fixture f = new(0x5747u);
        LlmPlacement placement = new([new LlmStage(f.Backend, 0, 2), new LlmStage(f.Backend, 2, Layers)]);
        GenericTransformerModel staged = new(f.Transformer, f.Backend, placement);
        int[] prefix = f.Prompt(6);
        int[] stopped = f.Prompt(10);
        int[] shorter = f.Prompt(3);
        using CancellationTokenSource cancel = new();
        using RecordingKvCache cache = new((IKvCache)staged.CreateSequenceState(new SequenceStateOptions(64)));
        staged.Prefill(new PrefillChunk(prefix, 0), cache).Dispose();
        cache.ClearRecord();
        cache.CancelAt(layer: 2, cancel);

        Assert.Throws<OperationCanceledException>(() =>
            staged.Prefill(new PrefillChunk(stopped, prefix.Length, LastRowOnly: true), cache, cancel.Token));

        Assert.Equal([0, 1, 2], cache.PrefillLayers);
        Assert.Equal(prefix.Length, cache.Length);

        // Layers 0-2 wrote ten rows past the cursor. A shorter suffix overwrites only three of them, and the rest must
        // never be read: the result matches the same suffix on a cache that never saw the stopped call, bit for bit.
        cache.ClearRecord();
        using Tensor afterStop = staged.Prefill(new PrefillChunk(shorter, prefix.Length, LastRowOnly: true), cache);
        using FixedKvCache clean = (FixedKvCache)staged.CreateSequenceState(new SequenceStateOptions(64));
        staged.Prefill(new PrefillChunk(prefix, 0), clean).Dispose();
        using Tensor expected = staged.Prefill(new PrefillChunk(shorter, prefix.Length, LastRowOnly: true), clean);
        Assert.Equal(Bits(expected), Bits(afterStop));
        Assert.Equal(prefix.Length + shorter.Length, cache.Length);
    }

    [Fact]
    public void ALiveTokenThatIsNeverCancelled_LeavesThePrefillBitIdentical()
    {
        using Fixture f = new(0x1D3Au);
        GenericTransformerModel model = new(f.Transformer, f.Backend);
        int[] prompt = f.Prompt(20);
        using CancellationTokenSource live = new();
        using ISequenceState plain = model.CreateSequenceState(new SequenceStateOptions(64));
        using ISequenceState watched = model.CreateSequenceState(new SequenceStateOptions(64));

        using Tensor expected = model.Prefill(new PrefillChunk(prompt, 0), plain);
        using Tensor actual = model.Prefill(new PrefillChunk(prompt, 0), watched, live.Token);

        Assert.Equal(Bits(expected), Bits(actual));
        Assert.Equal(plain.Length, watched.Length);
    }

    private static Task<IReadOnlyList<TextChunk>> Produce(TextGenerationPipeline pipeline, GenerationRequest request,
        RetainedSequence? reuse, CancellationToken token) =>
        Task.Run<IReadOnlyList<TextChunk>>(() =>
        {
            GenerationResult result = pipeline.Generate(request, reuse, onToken: null, token);
            return [new TextChunk { Kind = TextChunkKind.Result, Text = result.Text }];
        }, token);

    private static unsafe uint[] Bits(Tensor t) => new ReadOnlySpan<uint>(t.DataPointer, (int)t.ElementCount).ToArray();

    /// <summary>A tiny random-weight model on the CPU backend, driven through <see cref="RecordingModel"/>.</summary>
    private sealed class Fixture : IDisposable
    {
        private readonly Dictionary<string, Tensor> _weights;
        private readonly StubTokenizer _tokenizer = new();

        public Fixture(uint seed)
        {
            _rng = seed;
            TransformerConfig cfg = Cfg();
            _weights = Weights(cfg);
            Backend = new CpuBackend();
            Transformer = new GenericTransformer(cfg);
            Transformer.LoadWeights(_weights, "model");
            Model = new RecordingModel(new GenericTransformerModel(Transformer, Backend));
            Pipeline = new TextGenerationPipeline(Model, _tokenizer);
        }

        public CpuBackend Backend { get; }

        public GenericTransformer Transformer { get; }

        public RecordingModel Model { get; }

        public TextGenerationPipeline Pipeline { get; }

        public int[] Prompt(int length)
        {
            int[] ids = new int[length];
            for (int i = 0; i < length; i++) ids[i] = (int)(NextRaw() % (uint)(Cfg().VocabSize - 1));
            return ids;
        }

        /// <summary>The tokens a fresh pipeline with a fresh cache generates for <paramref name="request"/>.</summary>
        public string Reference(GenerationRequest request) =>
            string.Join(",", new TextGenerationPipeline(Transformer, _tokenizer, Backend).Generate(request).TokenIds);

        public void Dispose()
        {
            Transformer.Dispose();
            Backend.Dispose();
            foreach (Tensor t in _weights.Values) t.Dispose();
        }
    }

    /// <summary>Delegates to a real <see cref="GenericTransformerModel"/>, handing out <see cref="RecordingKvCache"/>s so a
    /// test sees which layers a prefill reached and can cancel from inside one.</summary>
    private sealed class RecordingModel(GenericTransformerModel inner) : IGenerationModel
    {
        private (int Layer, CancellationTokenSource Source)? _cancelAt;

        public List<RecordingKvCache> Caches { get; } = [];

        /// <summary>Layers that wrote prompt rows (more than one token) since the last <see cref="Reset"/>, in order.</summary>
        public IEnumerable<int> PrefillLayers => Caches.SelectMany(c => c.PrefillLayers);

        public GenerationModelInfo Info => inner.Info;
        public GenerationCapabilities Capabilities => inner.Capabilities;
        public IBackend OutputBackend => inner.OutputBackend;

        public void CancelAt(int layer, CancellationTokenSource source)
        {
            _cancelAt = (layer, source);
            foreach (RecordingKvCache cache in Caches) cache.CancelAt(layer, source);
        }

        public void Reset()
        {
            _cancelAt = null;
            foreach (RecordingKvCache cache in Caches) cache.ClearRecord();
            Caches.RemoveAll(c => c.Disposed);
        }

        public ISequenceState CreateSequenceState(SequenceStateOptions options)
        {
            RecordingKvCache cache = new((IKvCache)inner.CreateSequenceState(options));
            if (_cancelAt is { } at) cache.CancelAt(at.Layer, at.Source);
            Caches.Add(cache);
            return cache;
        }

        public Tensor Prefill(in PrefillChunk chunk, ISequenceState state) => inner.Prefill(chunk, state);
        public Tensor Prefill(in PrefillChunk chunk, ISequenceState state, CancellationToken cancel) => inner.Prefill(chunk, state, cancel);
        public Tensor DecodeBatch(ReadOnlySpan<int> tokenIds, ISequenceState[] states) => inner.DecodeBatch(tokenIds, states);
        public Tensor ProjectLogits(Tensor hidden, int rows) => inner.ProjectLogits(hidden, rows);
        public IEnumerable<Tensor> EnumerateWeights(bool includeRedundantSplits) => inner.EnumerateWeights(includeRedundantSplits);
        public long EstimateSequenceBytes(int contextTokens) => inner.EstimateSequenceBytes(contextTokens);
        public CapacitySnapshot Capacity() => inner.Capacity();
        public void Dispose() => inner.Dispose();
    }

    /// <summary>Wraps a real cache, recording each layer that appends a prompt (more than one row) and cancelling a source
    /// from inside the chosen layer's append, the moment that layer's work is in flight.</summary>
    private sealed class RecordingKvCache(IKvCache inner) : IKvCache
    {
        private (int Layer, CancellationTokenSource Source)? _cancelAt;

        public List<int> PrefillLayers { get; } = [];

        public bool Disposed { get; private set; }

        /// <summary>The committed length when the cache was disposed; -1 while it is live.</summary>
        public int CommittedAtDispose { get; private set; } = -1;

        public int NumLayers => inner.NumLayers;
        public int CurrentLength => inner.CurrentLength;
        public int Length => inner.CurrentLength;
        public int Capacity => ((ISequenceState)inner).Capacity;
        public int MaxRollback => ((ISequenceState)inner).MaxRollback;

        public void CancelAt(int layer, CancellationTokenSource source) => _cancelAt = (layer, source);

        public void ClearRecord()
        {
            _cancelAt = null;
            PrefillLayers.Clear();
        }

        public void AppendStep(IBackend backend, int layer, Tensor newK, Tensor newV)
        {
            if (newK.Shape[2] > 1)
            {
                PrefillLayers.Add(layer);
                if (_cancelAt is { } at && at.Layer == layer) at.Source.Cancel();
            }
            inner.AppendStep(backend, layer, newK, newV);
        }

        public Tensor KeyPrefix(int layer) => inner.KeyPrefix(layer);
        public Tensor ValuePrefix(int layer) => inner.ValuePrefix(layer);
        public void AdvanceLength(int by) => inner.AdvanceLength(by);
        public void Truncate(int newLength) => ((ISequenceState)inner).Truncate(newLength);
        public void Reset() => inner.Reset();

        public void Dispose()
        {
            if (Disposed) return;
            CommittedAtDispose = inner.CurrentLength;
            Disposed = true;
            inner.Dispose();
        }
    }

    /// <summary>Bypasses Encode/chat-template entirely: every test drives prompts via <see cref="GenerationRequest.RawTokenIds"/>.</summary>
    private sealed class StubTokenizer : ILlmTokenizer
    {
        public int[] Encode(string text, bool addSpecial) => throw new NotSupportedException();
        public int[] EncodeOrdinary(string text) => throw new NotSupportedException();
        public string Decode(IReadOnlyList<int> ids) => string.Join(",", ids);
        public int? SpecialId(string token) => null;
        public int? BosId => null;
        public int? EosId => 36;
        public IReadOnlyList<int> StopIds => [36];
        public string? BosToken => null;
        public string? EosToken => null;
    }
}
