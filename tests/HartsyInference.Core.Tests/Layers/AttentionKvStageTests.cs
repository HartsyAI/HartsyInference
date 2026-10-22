using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common.Attention;
using Xunit;

namespace HartsyInference.Core.Tests.Layers;

/// <summary>The key/value attention stages an LLM runs: the op order, which operands are released when, the one Flash call each
/// issues, what they refuse, and that they allocate no more than the inlined block they replaced.</summary>
public sealed class AttentionKvStageTests
{
    private const int Heads = 4, KvHeads = 2, HeadDim = 4, Capacity = 8;

    private static AttentionLayer Layer(float scale = 0.5f) =>
        new(new AttentionLayerConfig { Heads = Heads, KvHeads = KvHeads, HeadDim = HeadDim, ComputeDType = DType.F32, Scale = scale });

    private static Tensor TokenMajor(int heads, int tokens) => new(new TensorShape(1, tokens, heads, HeadDim), DType.F32);

    private static Tensor HeadMajor(int heads, int tokens) => new(new TensorShape(1, heads, tokens, HeadDim), DType.F32);

    private static unsafe bool IsDisposed(Tensor t)
    {
        try
        {
            _ = (nint)t.DataPointer;
            return false;
        }
        catch (ObjectDisposedException)
        {
            return true;
        }
    }

    private static string State(Tensor t) => IsDisposed(t) ? "disposed" : "live";

    /// <summary>A cache stand-in that logs into the backend's op list, so the interleaving of appends, reads and attention is one sequence.</summary>
    private sealed class RecordingSource(List<string> log, Func<string>? watch = null, bool failAppend = false) : IAttentionKvSource
    {
        public Tensor Keys { get; } = HeadMajor(KvHeads, Capacity);

        public Tensor Values { get; } = HeadMajor(KvHeads, Capacity);

        public List<(Tensor K, Tensor V)> Appended { get; } = [];

        public void AppendKv(IBackend backend, int layer, Tensor k, Tensor v)
        {
            log.Add($"AppendKv L{layer} k={State(k)} v={State(v)}{(watch is null ? "" : " " + watch())}");
            Appended.Add((k, v));
            if (failAppend) throw new InvalidOperationException("append failed");
        }

        public Tensor KeyBuffer(int layer)
        {
            log.Add($"KeyBuffer L{layer} appended={(Appended.Count > 0 ? State(Appended[^1].K) : "none")}");
            return Keys;
        }

        public Tensor ValueBuffer(int layer)
        {
            log.Add($"ValueBuffer L{layer} appended={(Appended.Count > 0 ? State(Appended[^1].V) : "none")}");
            return Values;
        }
    }

    private sealed class SilentSource(Tensor keys, Tensor values) : IAttentionKvSource
    {
        public void AppendKv(IBackend backend, int layer, Tensor k, Tensor v) { }

        public Tensor KeyBuffer(int layer) => keys;

        public Tensor ValueBuffer(int layer) => values;
    }

    [Fact]
    public void AttendOverKv_PermutesAppendsReadsKThenVOnceAndIssuesOneFlashCall()
    {
        AttentionTraceBackend backend = new() { FlashAsAttention = true };
        Tensor q = TokenMajor(Heads, 3), k = TokenMajor(KvHeads, 3), v = TokenMajor(KvHeads, 3);
        RecordingSource source = new(backend.Ops, () => $"src={State(q)}/{State(k)}/{State(v)}");

        Tensor attn = Layer().AttendOverKv(backend, q, k, v, source, layer: 1, kvLayer: 1,
            new KvAttentionCall { KvLength = 3, QueryOffset = 0 });

        Assert.Equal(
        [
            "Permute0213 3,4,4", "Permute0213 3,2,4", "Permute0213 3,2,4",
            "AppendKv L1 k=live v=live src=disposed/disposed/disposed",
            "KeyBuffer L1 appended=disposed", "ValueBuffer L1 appended=disposed",
            "Attention Flash heads=4/2 kv=3 q@0 causal window=0 softcap=0",
        ], backend.Ops);
        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Null(call.LegacyGap());
        Assert.Equal(AttentionRoute.Flash, call.Route);
        Assert.Equal(0.5f, call.Scale);
        Assert.Equal(2, call.KvGroup);
        Assert.Same(source.Keys, call.Key.Storage);
        Assert.Same(source.Values, call.Value.Storage);
        Assert.Same(attn, call.Output.Storage);
        Assert.Equal(new TensorShape(1, Heads, 3, HeadDim), attn.Shape);
        Assert.Equal(Capacity, call.KeyTokens);
        foreach (AttentionOperand operand in (AttentionOperand[])[call.Query, call.Key, call.Value, call.Output])
        {
            Assert.Equal(AttentionLayout.HeadMajor, operand.Layout);
            Assert.True(operand.IsPlain);
        }
        Assert.True(IsDisposed(call.Query.Storage));
        Assert.False(IsDisposed(attn));
        Assert.False(IsDisposed(source.Keys));
        Assert.False(IsDisposed(source.Values));
    }

    [Fact]
    public void AttendOverKv_ReadsADonorWithoutAppendingWhenTheKeyIsNull()
    {
        AttentionTraceBackend backend = new() { FlashAsAttention = true };
        Tensor q = TokenMajor(Heads, 1);
        RecordingSource source = new(backend.Ops);

        Tensor attn = Layer().AttendOverKv(backend, q, null, null, source, layer: 3, kvLayer: 1, new KvAttentionCall { KvLength = 5, QueryOffset = 4 });

        Assert.Equal(
        [
            "Permute0213 1,4,4",
            "KeyBuffer L1 appended=none", "ValueBuffer L1 appended=none",
            "Attention Flash heads=4/2 kv=5 q@4 causal window=0 softcap=0",
        ], backend.Ops);
        Assert.Empty(source.Appended);
        Assert.True(IsDisposed(q));
        Assert.False(IsDisposed(attn));
    }

    [Fact]
    public void AttendOverKvHeadMajor_ReleasesOnlyWhatConsumeNames()
    {
        AttentionTraceBackend backend = new() { FlashAsAttention = true };
        RecordingSource source = new(backend.Ops);
        Tensor q = HeadMajor(Heads, 1), k = HeadMajor(KvHeads, 1), v = HeadMajor(KvHeads, 1);

        Tensor attn = Layer().AttendOverKvHeadMajor(backend, q, k, v, source, 0, 0, new KvAttentionCall { KvLength = 1 }, OperandOwnership.Key);

        Assert.True(IsDisposed(k));
        Assert.False(IsDisposed(q));
        Assert.False(IsDisposed(v));
        Assert.Equal("KeyBuffer L0 appended=disposed", backend.Ops[1]);
        Assert.Equal("ValueBuffer L0 appended=live", backend.Ops[2]);
        Assert.False(IsDisposed(attn));
    }

    [Fact]
    public void AttendOverKvHeadMajor_WithoutASourceAttendsOverTheOperandsAndReleasesThemAfterwards()
    {
        AttentionTraceBackend backend = new() { FlashAsAttention = true };
        Tensor q = HeadMajor(Heads, 2), k = HeadMajor(KvHeads, 6), v = HeadMajor(KvHeads, 6);

        Tensor attn = Layer().AttendOverKvHeadMajor(backend, q, k, v, null, 0, 0, new KvAttentionCall { KvLength = 6, Bidirectional = true });

        Assert.Equal(["Attention Flash heads=4/2 kv=6 q@0 full window=0 softcap=0"], backend.Ops);
        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Same(k, call.Key.Storage);
        Assert.Same(v, call.Value.Storage);
        Assert.False(call.Mask.Causal);
        Assert.Null(call.LegacyGap());
        Assert.True(IsDisposed(q) && IsDisposed(k) && IsDisposed(v));
        Assert.False(IsDisposed(attn));
    }

    [Fact]
    public void AttendOverKvHeadMajor_WithoutASourceNeedsKeyAndValue()
    {
        AttentionTraceBackend backend = new() { FlashAsAttention = true };

        Assert.Throws<ArgumentException>(() => Layer().AttendOverKvHeadMajor(backend, HeadMajor(Heads, 1), null, null, null, 0, 0, default));
        Assert.Empty(backend.Calls);
    }

    [Theory]
    [InlineData(0, 0f)]
    [InlineData(1, 0f)]
    [InlineData(3, 50f)]
    [InlineData(100, 20.5f)]
    public void TheCallCarriesWindowSoftcapSinksAndAlibiUnchanged(int window, float softcap)
    {
        AttentionTraceBackend backend = new() { FlashAsAttention = true };
        Tensor sinks = new(new TensorShape(Heads), DType.F32), alibi = new(new TensorShape(Heads), DType.F32);

        Layer().AttendOverKv(backend, TokenMajor(Heads, 2), TokenMajor(KvHeads, 2), TokenMajor(KvHeads, 2), new RecordingSource(backend.Ops),
            0, 0, new KvAttentionCall { KvLength = 7, QueryOffset = 5, SlidingWindow = window, Softcap = softcap, Sinks = sinks, AlibiSlopes = alibi });

        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Equal(window, call.Mask.LegacySlidingWindow);
        Assert.Equal(window > 0 ? window - 1 : null, call.Mask.WindowLeft);
        Assert.Equal(softcap, call.Logits.Softcap);
        Assert.Same(sinks, call.Logits.Sinks);
        Assert.Same(alibi, call.Logits.AlibiSlopes);
        Assert.Equal(7, call.ValidKvLength);
        Assert.Equal(5, call.Positions.QueryOffset);
        Assert.Null(call.LegacyGap());
    }

    [Fact]
    public void AFailingAttentionDisposesTheOutputAndConsumedOperandsButNeverTheCache()
    {
        AttentionTraceBackend backend = new() { FailAttention = true, FlashAsAttention = true };
        RecordingSource source = new(backend.Ops);
        Tensor q = TokenMajor(Heads, 2), k = TokenMajor(KvHeads, 2), v = TokenMajor(KvHeads, 2);

        Assert.Throws<OutOfMemoryException>(() => Layer().AttendOverKv(backend, q, k, v, source, 0, 0, new KvAttentionCall { KvLength = 2 }));

        AttentionCall call = Assert.Single(backend.Calls);
        Assert.True(IsDisposed(call.Output.Storage));
        Assert.True(IsDisposed(call.Query.Storage));
        Assert.True(IsDisposed(q) && IsDisposed(k) && IsDisposed(v));
        Assert.False(IsDisposed(source.Keys));
        Assert.False(IsDisposed(source.Values));
    }

    [Fact]
    public void AFailingAppendDisposesTheConsumedOperandsAndIssuesNoAttention()
    {
        AttentionTraceBackend backend = new() { FlashAsAttention = true };
        RecordingSource source = new(backend.Ops, failAppend: true);
        Tensor q = HeadMajor(Heads, 1), k = HeadMajor(KvHeads, 1), v = HeadMajor(KvHeads, 1);

        Assert.Throws<InvalidOperationException>(() => Layer().AttendOverKvHeadMajor(backend, q, k, v, source, 0, 0, new KvAttentionCall { KvLength = 1 }));

        Assert.Empty(backend.Calls);
        Assert.True(IsDisposed(q) && IsDisposed(k) && IsDisposed(v));
        Assert.DoesNotContain(backend.Ops, op => op.StartsWith("KeyBuffer", StringComparison.Ordinal));
    }

    [Fact]
    public void ABufferWithTheWrongKvHeadCountIsRefusedBeforeAnyAttention()
    {
        AttentionTraceBackend backend = new() { FlashAsAttention = true };
        RecordingSource source = new(backend.Ops);
        AttentionLayer wrong = new(new AttentionLayerConfig { Heads = Heads, KvHeads = 4, HeadDim = HeadDim, ComputeDType = DType.F32 });

        Assert.Throws<ArgumentException>(() => wrong.AttendOverKv(backend, TokenMajor(Heads, 1), TokenMajor(4, 1), TokenMajor(4, 1), source, 0, 0,
            new KvAttentionCall { KvLength = 1 }));

        Assert.Empty(backend.Calls);
    }

    [Fact]
    public void AttendDevice_IssuesOneFlashDeviceCallAndDisposesTheQuery()
    {
        AttentionTraceBackend backend = new() { FlashAsAttention = true };
        Tensor q = HeadMajor(Heads, 1), keys = HeadMajor(KvHeads, Capacity), values = HeadMajor(KvHeads, Capacity);

        Tensor attn = Layer().AttendDevice(backend, q, keys, values, 7, new KvAttentionCall { SlidingWindow = 3, Softcap = 50f });

        Assert.Equal(["Attention FlashDevice heads=4/2 kv=0 q@0 causal window=3 softcap=50 dev=7"], backend.Ops);
        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Null(call.LegacyGap());
        Assert.Equal(7UL, call.Positions.DeviceHandle);
        Assert.Same(keys, call.Key.Storage);
        Assert.Same(values, call.Value.Storage);
        Assert.Same(attn, call.Output.Storage);
        Assert.Equal(new TensorShape(1, Heads, 1, HeadDim), attn.Shape);
        Assert.Equal(0.5f, call.Scale);
        Assert.True(call.Query.IsPlain && call.Key.IsPlain && call.Value.IsPlain && call.Output.IsPlain);
        Assert.True(IsDisposed(q));
        Assert.False(IsDisposed(attn) || IsDisposed(keys) || IsDisposed(values));
    }

    [Fact]
    public void AttendDevice_KeepsTheQueryWhenNotConsumed()
    {
        AttentionTraceBackend backend = new() { FlashAsAttention = true };
        Tensor q = HeadMajor(Heads, 1);

        Layer().AttendDevice(backend, q, HeadMajor(KvHeads, Capacity), HeadMajor(KvHeads, Capacity), 7, default, consumeQuery: false);

        Assert.False(IsDisposed(q));
    }

    [Fact]
    public void AttendDevice_RefusesSinksAndAlibiBeforeAllocatingOrIssuingAnything()
    {
        AttentionTraceBackend backend = new() { FlashAsAttention = true };
        Tensor q = HeadMajor(Heads, 1), keys = HeadMajor(KvHeads, Capacity), values = HeadMajor(KvHeads, Capacity);
        Tensor extra = new(new TensorShape(Heads), DType.F32);

        Assert.Throws<NotSupportedException>(() => Layer().AttendDevice(backend, q, keys, values, 7, new KvAttentionCall { Sinks = extra }));
        Assert.Throws<NotSupportedException>(() => Layer().AttendDevice(backend, q, keys, values, 7, new KvAttentionCall { AlibiSlopes = extra }));

        Assert.Empty(backend.Calls);
        Assert.False(IsDisposed(q));
    }

    [Fact]
    public void AFailingDeviceAttentionDisposesTheOutputAndTheConsumedQuery()
    {
        AttentionTraceBackend backend = new() { FailAttention = true, FlashAsAttention = true };
        Tensor q = HeadMajor(Heads, 1);

        Assert.Throws<OutOfMemoryException>(() => Layer().AttendDevice(backend, q, HeadMajor(KvHeads, Capacity), HeadMajor(KvHeads, Capacity), 7, default));

        Assert.True(IsDisposed(Assert.Single(backend.Calls).Output.Storage));
        Assert.True(IsDisposed(q));
    }

    // ── allocation parity: the stage allocates no more than the block it replaced ───────────────────────────────────

    // The thread's allocation counter advances in allocation-context steps (8 KB), so equal work can read one step apart; a stage
    // allocation of even one object per call is 4096 x 24 B, far past the slack.
    private const int Rounds = 4096, Slack = 2 * 8192;

    /// <summary>The attention block of GenericTransformer.Layer.Forward as it was before the stage, verbatim except for the trace backend.</summary>
    private static Tensor LegacyForward(IBackend backend, Tensor q, Tensor? k, Tensor? v, IAttentionKvSource source, int layer, int kvLayer,
        int t, int posStart, Tensor? sink, int window, float softcap)
    {
        Tensor qMh = new(new TensorShape(1, Heads, t, HeadDim), DType.F32);
        backend.Permute0213(qMh, q, t, Heads, HeadDim);
        q.Dispose();
        Tensor kFull, vFull;
        if (k is not null)
        {
            Tensor kMh = new(new TensorShape(1, KvHeads, t, HeadDim), DType.F32);
            Tensor vMh = new(new TensorShape(1, KvHeads, t, HeadDim), DType.F32);
            backend.Permute0213(kMh, k, t, KvHeads, HeadDim);
            backend.Permute0213(vMh, v!, t, KvHeads, HeadDim);
            k.Dispose(); v!.Dispose();
            source.AppendKv(backend, layer, kMh, vMh);
            kMh.Dispose(); vMh.Dispose();
            kFull = source.KeyBuffer(layer);
            vFull = source.ValueBuffer(layer);
        }
        else
        {
            kFull = source.KeyBuffer(kvLayer);
            vFull = source.ValueBuffer(kvLayer);
        }
        Tensor attn = new(new TensorShape(1, Heads, t, HeadDim), DType.F32);
        backend.FlashAttention(attn, qMh, kFull, vFull, posStart + t, KvHeads == 0 ? 0 : Heads / KvHeads, causal: true, qOffset: posStart, 0.5f, softcap, sink, window, null);
        qMh.Dispose();
        return attn;
    }

    private static long Measure(Action<int> body)
    {
        body(-1);   // warm-up (JIT)
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < Rounds; i++) body(i);
        return GC.GetAllocatedBytesForCurrentThread() - before;
    }

    [Theory]
    [InlineData(1, true)]
    [InlineData(5, true)]
    [InlineData(1, false)]
    public void AttendOverKv_AllocatesNoMoreThanTheInlinedBlock(int t, bool ownKv)
    {
        AttentionTraceBackend backend = new() { Silent = true, FlashAsAttention = true };
        SilentSource source = new(HeadMajor(KvHeads, Capacity), HeadMajor(KvHeads, Capacity));
        AttentionLayer layer = Layer();
        KvAttentionCall call = new() { KvLength = t, QueryOffset = 0, SlidingWindow = 3, Softcap = 50f };
        Tensor[] qs = new Tensor[Rounds + 1], ks = new Tensor[Rounds + 1], vs = new Tensor[Rounds + 1];

        void Fill()
        {
            for (int i = 0; i <= Rounds; i++)
            {
                qs[i] = TokenMajor(Heads, t);
                ks[i] = ownKv ? TokenMajor(KvHeads, t) : null!;
                vs[i] = ownKv ? TokenMajor(KvHeads, t) : null!;
            }
        }

        Fill();
        long legacy = Measure(i => LegacyForward(backend, qs[i + 1], ownKv ? ks[i + 1] : null, ownKv ? vs[i + 1] : null, source, 0, 0, t, 0, null, 3, 50f).Dispose());
        Fill();
        long stage = Measure(i => layer.AttendOverKv(backend, qs[i + 1], ownKv ? ks[i + 1] : null, ownKv ? vs[i + 1] : null, source, 0, 0, in call).Dispose());

        Assert.True(stage <= legacy + Slack, $"The stage allocated {stage} bytes over {Rounds} calls; the inlined block {legacy}.");
    }

    [Fact]
    public void AttendOverKvHeadMajor_AllocatesNoMoreThanTheInlinedAppendAndFlash()
    {
        AttentionTraceBackend backend = new() { Silent = true, FlashAsAttention = true };
        SilentSource source = new(HeadMajor(KvHeads, Capacity), HeadMajor(KvHeads, Capacity));
        AttentionLayer layer = Layer();
        KvAttentionCall call = new() { KvLength = 2, QueryOffset = 1 };
        Tensor[] qs = new Tensor[Rounds + 1], ks = new Tensor[Rounds + 1], vs = new Tensor[Rounds + 1];

        void Fill()
        {
            for (int i = 0; i <= Rounds; i++) (qs[i], ks[i], vs[i]) = (HeadMajor(Heads, 1), HeadMajor(KvHeads, 1), HeadMajor(KvHeads, 1));
        }

        Fill();
        long legacy = Measure(i =>
        {
            source.AppendKv(backend, 0, ks[i + 1], vs[i + 1]);
            ks[i + 1].Dispose(); vs[i + 1].Dispose();
            Tensor attn = new(new TensorShape(1, Heads, 1, HeadDim), DType.F32);
            backend.FlashAttention(attn, qs[i + 1], source.KeyBuffer(0), source.ValueBuffer(0), 2, 2, causal: true, qOffset: 1, 0.5f, 0f, null, 0, null);
            qs[i + 1].Dispose();
            attn.Dispose();
        });
        Fill();
        long stage = Measure(i => layer.AttendOverKvHeadMajor(backend, qs[i + 1], ks[i + 1], vs[i + 1], source, 0, 0, in call).Dispose());

        Assert.True(stage <= legacy + Slack, $"The stage allocated {stage} bytes over {Rounds} calls; the inlined block {legacy}.");
    }

    [Fact]
    public void AttendDevice_AllocatesNoMoreThanTheInlinedDeviceCall()
    {
        AttentionTraceBackend backend = new() { Silent = true, FlashAsAttention = true };
        Tensor keys = HeadMajor(KvHeads, Capacity), values = HeadMajor(KvHeads, Capacity);
        AttentionLayer layer = Layer();
        KvAttentionCall call = new() { SlidingWindow = 3, Softcap = 50f };
        Tensor[] qs = new Tensor[Rounds + 1];

        void Fill()
        {
            for (int i = 0; i <= Rounds; i++) qs[i] = HeadMajor(Heads, 1);
        }

        Fill();
        long legacy = Measure(i =>
        {
            Tensor attn = new(new TensorShape(1, Heads, 1, HeadDim), DType.F32);
            ((IBackend)backend).FlashAttentionDev(attn, qs[i + 1], keys, values, 0, 2, causal: true, 0, 0.5f, 7, 50f, 3);
            qs[i + 1].Dispose();
            attn.Dispose();
        });
        Fill();
        long stage = Measure(i => layer.AttendDevice(backend, qs[i + 1], keys, values, 7, in call).Dispose());

        Assert.True(stage <= legacy + Slack, $"The stage allocated {stage} bytes over {Rounds} calls; the inlined block {legacy}.");
    }
}
