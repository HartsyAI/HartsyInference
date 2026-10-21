using System.Text.RegularExpressions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Looped;
using HartsyInference.LLM.Multimodal;
using HartsyInference.LLM.Ssm;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests.Attention;

/// <summary>The attention call each migrated LLM site (tensor-parallel ranks, looped nGPT, Qwen3.5, Mllama cross-attention) issues through the layer's stage: the permutes, the cache appends and reads, and
/// the one attention call, in the order the code ran before the migration (permute q, k, v; append; read K then V once; attend; permute back), with the head counts, key length, mask, scale and route.</summary>
public sealed unsafe class AttentionKvSiteTraceTests
{
    private const int Vocab = 24, Hidden = 16, Inter = 24;

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
        if (name.EndsWith("norm.weight", StringComparison.Ordinal) || name.EndsWith("layernorm.weight", StringComparison.Ordinal))
            for (long i = 0; i < tensor.ElementCount; i++) data[i] += 1f;
        return tensor;
    }

    private sealed class Weights : Dictionary<string, Tensor>, IDisposable
    {
        public void Add(string name, params long[] dims) => base.Add(name, Seeded(name, dims));

        public void AddScale(string name, long n, float center)
        {
            Tensor t = Seeded(name, n);
            float* p = (float*)t.DataPointer;
            for (long i = 0; i < n; i++) p[i] = center * (1f + 0.3f * p[i]);
            base.Add(name, t);
        }

        public void AddScalar(string name, float value)
        {
            Tensor t = new(new TensorShape(1), DType.F32);
            *(float*)t.DataPointer = value;
            base.Add(name, t);
        }

        public void Dispose()
        {
            foreach (Tensor t in Values) t.Dispose();
        }
    }

    private static bool IsDisposed(Tensor t)
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

    /// <summary>Wraps a cache and logs the appends and the K and V reads (with whether the appended operands were already released) into the backend's trace.</summary>
    private sealed class LoggingKvCache(IKvCache inner, List<string> log) : IKvCache
    {
        private (Tensor K, Tensor V)? _last;

        public int NumLayers => inner.NumLayers;
        public int CurrentLength => inner.CurrentLength;
        public int Length => inner.CurrentLength;
        public int Capacity => ((ISequenceState)inner).Capacity;
        public int MaxRollback => ((ISequenceState)inner).MaxRollback;

        public void AppendStep(IBackend backend, int layer, Tensor newK, Tensor newV)
        {
            log.Add($"Append L{layer} k={State(newK)} v={State(newV)}");
            _last = (newK, newV);
            inner.AppendStep(backend, layer, newK, newV);
        }

        public Tensor KeyPrefix(int layer)
        {
            log.Add($"Key L{layer} appended={(_last is { } last ? State(last.K) : "none")}");
            return inner.KeyPrefix(layer);
        }

        public Tensor ValuePrefix(int layer)
        {
            log.Add($"Value L{layer} appended={(_last is { } last ? State(last.V) : "none")}");
            _last = null;
            return inner.ValuePrefix(layer);
        }

        public void AdvanceLength(int by) => inner.AdvanceLength(by);
        public void Truncate(int newLength) => ((ISequenceState)inner).Truncate(newLength);
        public void Reset() => inner.Reset();
        public void Dispose() => inner.Dispose();

        private static string State(Tensor t) => IsDisposed(t) ? "disposed" : "live";
    }

    /// <summary>The trace without the cache-internal copies, which depend on the cache and not on the attention site.</summary>
    private static List<string> Sans(List<string> trace) => [.. trace.Where(op => !op.StartsWith("KvCacheAppend", StringComparison.Ordinal))];

    private static string Flash(int heads, int kvHeads, int kv, int qOffset, bool causal, int window) =>
        $"Attention Flash heads={heads}/{kvHeads} kv={kv} q@{qOffset} {(causal ? "causal" : "full")} window={window} softcap=0";

    // ── Tensor parallel ────────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void EachTensorParallelRankAttendsItsOwnHeadSliceInTheLegacyOrder()
    {
        TransformerConfig cfg = new()
        {
            HiddenSize = Hidden, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = 8, IntermediateSize = Inter, VocabSize = Vocab,
            MaxPositionEmbeddings = 64, RopeTheta = 10_000f, TieWordEmbeddings = false, QkNorm = true,
        };
        using Weights w = new();
        w.Add("model.embed_tokens.weight", Vocab, Hidden);
        w.Add("lm_head.weight", Vocab, Hidden);
        w.Add("model.norm.weight", Hidden);
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            w.Add($"{p}.input_layernorm.weight", Hidden);
            w.Add($"{p}.post_attention_layernorm.weight", Hidden);
            w.Add($"{p}.self_attn.q_proj.weight", 32, Hidden);
            w.Add($"{p}.self_attn.k_proj.weight", 16, Hidden);
            w.Add($"{p}.self_attn.v_proj.weight", 16, Hidden);
            w.Add($"{p}.self_attn.o_proj.weight", Hidden, 32);
            w.Add($"{p}.self_attn.q_norm.weight", 8);
            w.Add($"{p}.self_attn.k_norm.weight", 8);
            w.Add($"{p}.mlp.gate_proj.weight", Inter, Hidden);
            w.Add($"{p}.mlp.up_proj.weight", Inter, Hidden);
            w.Add($"{p}.mlp.down_proj.weight", Hidden, Inter);
        }
        List<string>[] traces = [[], []];
        using DevicePositionShimBackend rank0 = new() { Trace = traces[0] };
        using DevicePositionShimBackend rank1 = new() { Trace = traces[1] };
        using HostStagedComm comm = new(2);
        using TensorParallelTransformer tp = new(cfg, new TpPlacement([rank0, rank1], comm));
        tp.LoadWeights(w, "model");
        IKvCache[] caches = [.. tp.CreateKvCaches().Select((cache, r) => (IKvCache)new LoggingKvCache(cache, traces[r]))];
        try
        {
            int position = 0;
            foreach (int[] ids in new int[][] { [3, 7, 1, 12, 5], [9], [4] })
            {
                int t = ids.Length;
                foreach (List<string> trace in traces) trace.Clear();
                using Tensor embeds = new(new TensorShape(1, t, Hidden), DType.F32);
                tp.EmbedLookup(embeds, ids);
                using Tensor onRank0 = new(embeds.Shape, DType.F32);
                rank0.CopyTo(onRank0, embeds);
                using Tensor hidden = tp.ForwardEmbedsTp(onRank0, t, position, caches);

                List<string> expected = [];
                for (int l = 0; l < cfg.NumLayers; l++)
                {
                    expected.AddRange([$"Permute0213 {t},2,8", $"Permute0213 {t},1,8", $"Permute0213 {t},1,8",
                        $"Append L{l} k=live v=live", $"Key L{l} appended=disposed", $"Value L{l} appended=disposed",
                        Flash(2, 1, position + t, position, causal: true, window: 0), $"Permute0213 2,{t},8"]);
                }
                DevicePositionShimBackend[] ranks = [rank0, rank1];
                for (int r = 0; r < 2; r++)
                {
                    Assert.Equal(expected, Sans(traces[r]));
                    Assert.All(ranks[r].Calls, call =>
                    {
                        Assert.Equal(cfg.AttnScale, call.Scale);
                        Assert.Null(call.LegacyGap());
                    });
                    ranks[r].Calls.Clear();
                }
                position += t;
            }
        }
        finally
        {
            foreach (IKvCache cache in caches) cache.Dispose();
        }
    }

    // ── Looped nGPT ────────────────────────────────────────────────────────────────────────────────────────────────

    private static LoopedTransformerConfig LoopedConfig() => new()
    {
        HiddenSize = 16, NumHeads = 2, NumKvHeads = 1, HeadDim = 8, IntermediateSize = 24, VocabSize = 20, MaxPositionEmbeddings = 64,
        InputLayers = 2, RecurrentLayers = 2, OutputLayers = 2, Loops = 2,
        LayerKinds = [LoopedLayerKind.SlidingAttention, LoopedLayerKind.Mlp, LoopedLayerKind.FullAttention, LoopedLayerKind.Mlp,
            LoopedLayerKind.SlidingAttention, LoopedLayerKind.Mlp],
        RopeThetas = [10_000f, 0f, 100_000f, 0f, 10_000f, 0f],
        SlidingWindow = 3, KvCacheMode = LoopedKvCacheMode.PerLoop, InitStd = 0.02f, AlphaInit = 0.05f, Architecture = "dory", LowVramQuant = false,
    };

    private static Weights LoopedWeights(LoopedTransformerConfig cfg)
    {
        Weights w = new();
        int h = cfg.HiddenSize, nq = cfg.QDim, nk = cfg.KvDim, ff = cfg.IntermediateSize, v = cfg.VocabSize;
        w.Add("model.embed_tokens.weight", v, h);
        w.Add("lm_head.weight", v, h);
        w.AddScale("lm_head.logit_scale.weight", v, cfg.InitStd * 0.5f);
        for (int p = 0; p < cfg.PhysicalLayers; p++)
        {
            string prefix = $"model.layers.{p}";
            w.AddScale($"{prefix}.residual_alpha.weight", h, cfg.InitStd / cfg.AlphaInit * 0.5f);
            if (cfg.IsAttention(p))
            {
                w.Add($"{prefix}.self_attn.q_proj.weight", nq, h);
                w.Add($"{prefix}.self_attn.k_proj.weight", nk, h);
                w.Add($"{prefix}.self_attn.v_proj.weight", nk, h);
                w.Add($"{prefix}.self_attn.o_proj.weight", h, nq);
                w.Add($"{prefix}.self_attn.gate_proj.weight", nq, h);
                w.AddScale($"{prefix}.self_attn.qk_scale.weight", nq, cfg.InitStd / MathF.Sqrt(cfg.HeadDim));
                w.AddScale($"{prefix}.self_attn.gate_scale.weight", nq, cfg.InitStd / MathF.Sqrt(cfg.EffectiveGateInit));
            }
            else
            {
                w.Add($"{prefix}.mlp.gate_proj.weight", ff, h);
                w.Add($"{prefix}.mlp.up_proj.weight", ff, h);
                w.Add($"{prefix}.mlp.down_proj.weight", h, ff);
                w.AddScale($"{prefix}.mlp.gate_scale.weight", ff, 1f / MathF.Sqrt(h));
                w.AddScale($"{prefix}.mlp.up_scale.weight", ff, 1f / MathF.Sqrt(h));
            }
        }
        return w;
    }

    /// <summary>The attention layers of one forward as (slot, window), in run order.</summary>
    private static List<(int Slot, int Window)> LoopedAttention(LoopedTransformerConfig cfg)
    {
        LoopedSchedule schedule = new(cfg, cfg.Loops);
        List<(int, int)> layers = [];
        for (int v = 0; v < schedule.VirtualLayers; v++)
        {
            int physical = schedule.Physical(v);
            if (!cfg.IsAttention(physical)) continue;
            layers.Add((schedule.CacheSlot(v), cfg.LayerKinds[physical] == LoopedLayerKind.SlidingAttention ? cfg.SlidingWindow : 0));
        }
        return layers;
    }

    private static LoopedSequenceState LoopedState(LoopedTransformerConfig cfg, List<string> trace) =>
        new(new LoggingKvCache(new FixedKvCache(LoopedSchedule.MaxSlotCount(cfg), 1, cfg.NumKvHeads, cfg.HeadDim, 16), trace), cfg.Loops);

    [Fact]
    public void ALoopedLayerPermutesAppendsAndAttendsOverItsSlotInTheLegacyOrder()
    {
        LoopedTransformerConfig cfg = LoopedConfig();
        using Weights w = LoopedWeights(cfg);
        using LoopedTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        List<string> trace = [];
        using DevicePositionShimBackend shim = new() { Trace = trace };
        using LoopedSequenceState state = LoopedState(cfg, trace);
        int position = 0;
        // A prefill chunk keeps all four permutes; a single token skips v's and the output's.
        foreach (int[] ids in new int[][] { [3, 7, 1, 12], [5], [9] })
        {
            int t = ids.Length;
            trace.Clear();
            using Tensor hidden = model.Forward(shim, ids, position, state, cfg.Loops);

            List<string> expected = [];
            foreach ((int slot, int window) in LoopedAttention(cfg))
            {
                expected.Add($"Permute0213 {t},2,8");
                expected.Add($"Permute0213 {t},1,8");
                if (t > 1) expected.Add($"Permute0213 {t},1,8");
                expected.AddRange([$"Append L{slot} k=live v=live", $"Key L{slot} appended=disposed", $"Value L{slot} appended=disposed",
                    Flash(2, 1, position + t, position, causal: true, window)]);
                if (t > 1) expected.Add($"Permute0213 2,{t},8");
            }
            Assert.Equal(expected, Sans(trace));
            Assert.All(shim.Calls, call => Assert.Equal(1f / MathF.Sqrt(8), call.Scale));
            shim.Calls.Clear();
            position += t;
        }
    }

    [Fact]
    public void ABatchedLoopedDecodeAttendsEachSequenceOverItsOwnPositionAfterOneAppend()
    {
        LoopedTransformerConfig cfg = LoopedConfig();
        using Weights w = LoopedWeights(cfg);
        using LoopedTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        List<string> trace = [];
        using DevicePositionShimBackend shim = new() { Trace = trace };
        using LoopedSequenceState a = LoopedState(cfg, trace), b = LoopedState(cfg, trace);
        IKvCache[] states = [a, b];
        int[][] prompts = [[3, 7, 1, 12, 5], [11, 2, 9]];
        for (int s = 0; s < 2; s++)
        {
            using Tensor hidden = model.Forward(shim, prompts[s], 0, states[s], cfg.Loops);
        }
        trace.Clear();
        int[] lengths = [a.CurrentLength, b.CurrentLength];
        using Tensor embeds = new(new TensorShape(1, 2, cfg.HiddenSize), DType.F32);
        model.EmbedLookup(embeds, [4, 6]);
        using Tensor result = model.ForwardBatchDecode(shim, embeds, lengths, states, cfg.Loops);

        List<string> expected = [];
        foreach ((int slot, int window) in LoopedAttention(cfg))
        {
            for (int s = 0; s < 2; s++)
            {
                expected.AddRange([$"Append L{slot} k=live v=live", $"Key L{slot} appended=disposed", $"Value L{slot} appended=disposed",
                    Flash(2, 1, lengths[s] + 1, lengths[s], causal: true, window)]);
            }
        }
        Assert.Equal(expected, Sans(trace));
    }

    [Fact]
    public void ALoopedGraphStepAttendsOnTheDeviceOverTheBuffersItAlreadyHolds()
    {
        LoopedTransformerConfig cfg = LoopedConfig();
        using Weights w = LoopedWeights(cfg);
        using LoopedTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        List<string> trace = [];
        using DevicePositionShimBackend shim = new() { Trace = trace };
        using LoopedSequenceState state = LoopedState(cfg, trace);
        using (Tensor hidden = model.Forward(shim, [3, 7, 1, 12, 5, 9], 0, state, cfg.Loops)) { }
        Tensor embedTable = model.EnsureEmbedResidentForGraphDecode(shim);
        model.EnsureRopeTablesForGraphDecode(shim, 16);
        ulong devicePos = shim.AllocDevicePos(), tokenId = shim.AllocDeviceTokenId();
        int pos = state.CurrentLength;
        shim.WriteDeviceTokenId(tokenId, 4);
        shim.WriteDevicePos(devicePos, pos + 1, pos);
        trace.Clear();
        model.ForwardGraphDecodeStep(shim, embedTable, state, cfg.Loops, devicePos, tokenId);

        List<string> expected = [];
        foreach ((int slot, int window) in LoopedAttention(cfg))
        {
            expected.AddRange([$"Key L{slot} appended=none", $"Value L{slot} appended=none",
                $"Attention FlashDevice heads=2/1 kv=0 q@0 causal window={window} softcap=0 dev={devicePos}"]);
        }
        Assert.Equal(expected, trace);
        shim.FreeDevicePos(devicePos);
        shim.FreeDeviceTokenId(tokenId);
    }

    // ── Qwen3.5 ────────────────────────────────────────────────────────────────────────────────────────────────────

    private const int QHidden = 32, QHeads = 4, QKvHeads = 2, QHeadDim = 8, QInter = 48;

    private static Qwen35HfConfig QwenConfig() => new()
    {
        HiddenSize = QHidden, NumLayers = 3, VocabSize = Vocab, NumHeads = QHeads, NumKvHeads = QKvHeads, HeadDim = QHeadDim, FullAttentionInterval = 0,
        LinearLayers = [false, false, false], LinearConvKernel = 4, LinearKeyHeadDim = 4, LinearValueHeadDim = 4, LinearNumKeyHeads = 2, LinearNumValueHeads = 4,
        PartialRotaryFactor = 0.5f, RopeTheta = 10_000f, TieWordEmbeddings = false, MaxPositionEmbeddings = 64, MropeSection = [2, 1, 1], MropeInterleaved = true,
    };

    private static Weights QwenWeights(Qwen35HfConfig cfg)
    {
        Weights w = new();
        const string root = "model.language_model";
        w.Add($"{root}.embed_tokens.weight", Vocab, QHidden);
        w.Add("lm_head.weight", Vocab, QHidden);
        w.Add($"{root}.norm.weight", QHidden);
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            string p = $"{root}.layers.{i}";
            w.Add($"{p}.input_layernorm.weight", QHidden);
            w.Add($"{p}.post_attention_layernorm.weight", QHidden);
            w.Add($"{p}.mlp.gate_proj.weight", QInter, QHidden);
            w.Add($"{p}.mlp.up_proj.weight", QInter, QHidden);
            w.Add($"{p}.mlp.down_proj.weight", QHidden, QInter);
            w.Add($"{p}.self_attn.q_proj.weight", QHeads * QHeadDim * 2, QHidden);
            w.Add($"{p}.self_attn.k_proj.weight", QKvHeads * QHeadDim, QHidden);
            w.Add($"{p}.self_attn.v_proj.weight", QKvHeads * QHeadDim, QHidden);
            w.Add($"{p}.self_attn.o_proj.weight", QHidden, QHeads * QHeadDim);
            w.Add($"{p}.self_attn.q_norm.weight", QHeadDim);
            w.Add($"{p}.self_attn.k_norm.weight", QHeadDim);
        }
        return w;
    }

    [Fact]
    public void Qwen35PrefillAndAHostDecodePermuteAppendAndAttendInTheLegacyOrder()
    {
        Qwen35HfConfig cfg = QwenConfig();
        using Weights w = QwenWeights(cfg);
        List<string> trace = [];
        using DevicePositionShimBackend shim = new(graphDecodeSupported: false) { Trace = trace };
        using Qwen35Model model = Qwen35Model.FromHuggingFace(w, cfg, maxSequenceLength: 16);
        int position = 0;
        foreach (int[] ids in new int[][] { [3, 7, 1, 12, 5, 9], [4], [8] })
        {
            int t = ids.Length;
            trace.Clear();
            using Tensor embeds = new(new TensorShape(1, t, QHidden), DType.F32);
            model.EmbedLookup(embeds, ids);
            using Tensor hidden = model.ForwardEmbedsHidden(shim, embeds, t);

            List<string> expected = [];
            for (int l = 0; l < cfg.NumLayers; l++)
            {
                expected.AddRange([$"Permute0213 {t},4,8", $"Permute0213 {t},2,8", $"Permute0213 {t},2,8", $"KvCacheAppend @{position}", $"KvCacheAppend @{position}",
                    Flash(4, 2, position + t, position, causal: true, window: 0), $"Permute0213 4,{t},8"]);
            }
            Assert.Equal(expected, trace);
            Assert.All(shim.Calls, call => Assert.Equal(1f / MathF.Sqrt(QHeadDim), call.Scale));
            shim.Calls.Clear();
            position += t;
        }
    }

    [Fact]
    public void Qwen35DeviceStepAppendsAndAttendsWithoutPermutes()
    {
        Qwen35HfConfig cfg = QwenConfig();
        using Weights w = QwenWeights(cfg);
        List<string> trace = [];
        using DevicePositionShimBackend shim = new(graphDecodeSupported: true) { Trace = trace };
        using Qwen35Model model = Qwen35Model.FromHuggingFace(w, cfg, maxSequenceLength: 16);
        int[] prompt = [3, 7, 1, 12, 5, 9];
        using (Tensor embeds = new(new TensorShape(1, prompt.Length, QHidden), DType.F32))
        {
            model.EmbedLookup(embeds, prompt);
            using Tensor hidden = model.ForwardEmbedsHidden(shim, embeds, prompt.Length);
        }
        trace.Clear();
        using (Tensor embeds = new(new TensorShape(1, 1, QHidden), DType.F32))
        {
            model.EmbedLookup(embeds, [4]);
            using Tensor hidden = model.ForwardEmbedsHidden(shim, embeds, 1);
        }

        List<string> expected = [];
        for (int l = 0; l < cfg.NumLayers; l++)
            expected.AddRange(["KvCacheAppend @6", "KvCacheAppend @6", Flash(4, 2, 7, 6, causal: true, window: 0)]);
        Assert.Equal(expected, trace);
    }

    [Fact]
    public void Qwen35GraphStepAttendsOnTheDeviceOverTheCacheBuffers()
    {
        Qwen35HfConfig cfg = QwenConfig();
        using Weights w = QwenWeights(cfg);
        List<string> trace = [];
        using DevicePositionShimBackend shim = new(graphDecodeSupported: true) { Trace = trace };
        using Qwen35Model model = Qwen35Model.FromHuggingFace(w, cfg, maxSequenceLength: 16);
        int[] prompt = [3, 7, 1, 12, 5, 9];
        using (Tensor embeds = new(new TensorShape(1, prompt.Length, QHidden), DType.F32))
        {
            model.EmbedLookup(embeds, prompt);
            using Tensor hidden = model.ForwardEmbedsHidden(shim, embeds, prompt.Length);
        }
        (Tensor cos, Tensor sin) = shim.BuildRopeTableDevice(16, QHeadDim, cfg.RotaryDim, cfg.RopeTheta, RopeScaling.None, splitHalfPartial: true);
        ulong devicePos = shim.AllocDevicePos(), tokenId = shim.AllocDeviceTokenId();
        shim.WriteDeviceTokenId(tokenId, 4);
        shim.WriteDevicePos(devicePos, 7, 6);
        trace.Clear();
        model.ForwardGraphDecodeStep(shim, w["model.language_model.embed_tokens.weight"], cos, sin, devicePos, tokenId);

        List<string> expected = [];
        for (int l = 0; l < cfg.NumLayers; l++)
            expected.Add($"Attention FlashDevice heads=4/2 kv=0 q@0 causal window=0 softcap=0 dev={devicePos}");
        Assert.Equal(expected, trace);
        shim.FreeDevicePos(devicePos);
        shim.FreeDeviceTokenId(tokenId);
        cos.Dispose();
        sin.Dispose();
    }

    // ── Mllama cross-attention ─────────────────────────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(3)]
    [InlineData(1)]
    public void MllamaCrossAttentionAttendsBidirectionallyOverTheVisionKeysWithoutACache(int t)
    {
        const int visionLen = 5;
        using Weights w = new();
        const string p = "blk.0";
        w.Add($"{p}.input_layernorm.weight", Hidden);
        w.Add($"{p}.post_attention_layernorm.weight", Hidden);
        w.Add($"{p}.cross_attn.q_proj.weight", 32, Hidden);
        w.Add($"{p}.cross_attn.k_proj.weight", 16, Hidden);
        w.Add($"{p}.cross_attn.v_proj.weight", 16, Hidden);
        w.Add($"{p}.cross_attn.o_proj.weight", Hidden, 32);
        w.Add($"{p}.cross_attn.q_norm.weight", 8);
        w.Add($"{p}.cross_attn.k_norm.weight", 8);
        w.Add($"{p}.mlp.gate_proj.weight", Inter, Hidden);
        w.Add($"{p}.mlp.up_proj.weight", Inter, Hidden);
        w.Add($"{p}.mlp.down_proj.weight", Hidden, Inter);
        w.AddScalar($"{p}.cross_attn_attn_gate", 0.7f);
        w.AddScalar($"{p}.cross_attn_mlp_gate", -0.4f);
        List<string> trace = [];
        using DevicePositionShimBackend shim = new() { Trace = trace };
        MllamaCrossAttentionLayer layer = new(Hidden, 4, 2, 8, Inter, 1e-6f);
        layer.LoadWeights(w, p);
        using Tensor vision = Seeded("vision", 1, visionLen, Hidden);
        using Tensor text = Seeded("text", 1, t, Hidden);

        using Tensor outp = layer.Forward(shim, text, t, vision, visionLen);

        Assert.Equal([$"Permute0213 {t},4,8", $"Permute0213 {visionLen},2,8", $"Permute0213 {visionLen},2,8",
            Flash(4, 2, visionLen, 0, causal: false, window: 0), $"Permute0213 4,{t},8"], trace);
        AttentionCall call = Assert.Single(shim.Calls);
        Assert.Equal(1f / MathF.Sqrt(8), call.Scale);
        Assert.Null(call.LegacyGap());
    }
}
