using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>End-to-end decode-loop parity for the Vulkan LLM step graph: a small synthetic dense transformer decoded greedily
/// for many tokens through the eager <c>Forward</c> path and through one recorded step replayed once per token must pick the
/// same token every step. This is the gate the <c>GraphDecodeSupported</c> doc comment asks for: per-op tests cannot catch
/// a replay that writes the KV cache at the wrong slot or reads a stale device position.</summary>
[Trait("Category", "GpuIntegration")]
public sealed unsafe class VulkanLlmGraphDecodeTests
{
    private readonly ITestOutputHelper _out;
    public VulkanLlmGraphDecodeTests(ITestOutputHelper output) => _out = output;

    private static bool VulkanAvailable()
    {
        try { using VulkanInstance i = new(); return i.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    private uint _rng = 0xC0FFEEu;
    private float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.4f; }
    private Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private Tensor F1(int a, float center) { Tensor t = F1(a); float* p = (float*)t.DataPointer; for (int i = 0; i < a; i++) p[i] += center; return t; }
    private Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    /// <summary>Q8_0 expert matrix: the device-routed MoE stage runs only quantized experts (F32 experts fall back to host routing).</summary>
    private Tensor Q8(int a, int b) { using Tensor f = F2(a, b); return GgufQuantizer.Quantize(f, DType.Q8_0); }
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }

    [Theory]
    [InlineData(false, false, false)]   // Llama-shaped: no bias, no QK-norm
    [InlineData(true, false, false)]    // Qwen2-shaped: QKV bias
    [InlineData(false, true, false)]    // Qwen3-shaped: per-head QK-norm
    [InlineData(false, true, true)]     // OLMoE-shaped: full-width QK-norm over all heads
    public void GraphDecode_MatchesEagerGreedyLoop(bool bias, bool qkNorm, bool fullNorm)
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        IBackend b = backend;
        Assert.True(b.StepGraphSupported || !b.LlmGraphDecodeSupported);
        if (!b.LlmGraphDecodeSupported) { _out.WriteLine("SKIPPED: push descriptors unavailable"); return; }

        TransformerConfig cfg = new()
        {
            HiddenSize = 64, NumLayers = 3, NumHeads = 4, NumKvHeads = 2, HeadDim = 16,
            IntermediateSize = 96, VocabSize = 80, MaxPositionEmbeddings = 256, AttentionBias = bias, QkNorm = qkNorm, QkNormFullDim = fullNorm,
            TieWordEmbeddings = true,
        };
        Dictionary<string, Tensor> w = Weights(cfg, bias, qkNorm, fullNorm);
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        Assert.True(model.SupportsGraphDecode(b));

        const int maxLen = 96, steps = 40;
        int[] prompt = [3, 17, 5, 41, 9];

        List<int> eager = RunEager(b, model, cfg, prompt, maxLen, steps);
        List<int> graph = RunGraph(b, model, cfg, prompt, maxLen, steps);
        _out.WriteLine("eager: " + string.Join(",", eager));
        _out.WriteLine("graph: " + string.Join(",", graph));
        Assert.Equal(eager, graph);
        foreach (Tensor t in w.Values) t.Dispose();
    }

    /// <summary>An untied quantized embed table never reaches the device under graph decode: the step embeds from the host row
    /// written before each replay. The greedy tokens must match the eager loop, which reads the same host table.</summary>
    [Theory]
    [InlineData("Q8_0")]
    [InlineData("Q5_0")]
    public void GraphDecode_UntiedQuantizedEmbed_StagedRow_MatchesEagerGreedyLoop(string dtypeName)
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        IBackend b = backend;
        if (!b.LlmGraphDecodeSupported) { _out.WriteLine("SKIPPED: push descriptors unavailable"); return; }

        TransformerConfig cfg = new()
        {
            HiddenSize = 64, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = 16,
            IntermediateSize = 96, VocabSize = 80, MaxPositionEmbeddings = 256, TieWordEmbeddings = false,
        };
        Dictionary<string, Tensor> w = Weights(cfg, bias: false, qkNorm: false);
        w["model.embed_tokens.weight"].Dispose();
        w["model.embed_tokens.weight"] = dtypeName == "Q8_0"
            ? GgufQuantizer.Quantize(F2(cfg.VocabSize, cfg.HiddenSize), DType.Q8_0)
            : RandomQ5_0(cfg.VocabSize, cfg.HiddenSize, seed: 7);
        w["lm_head.weight"] = F2(cfg.VocabSize, cfg.HiddenSize);
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model", "lm_head.weight");
        Assert.True(model.SupportsGraphDecode(b));
        Assert.True(model.StagesGraphDecodeEmbedRows(b));

        const int maxLen = 96, steps = 40;
        int[] prompt = [3, 17, 5, 41, 9];
        List<int> eager = RunEager(b, model, cfg, prompt, maxLen, steps);
        List<int> graph = RunGraph(b, model, cfg, prompt, maxLen, steps);
        _out.WriteLine("eager: " + string.Join(",", eager));
        _out.WriteLine("graph: " + string.Join(",", graph));
        Assert.Equal(eager, graph);
        foreach (Tensor t in w.Values) t.Dispose();
    }

    /// <summary>A random Q5_0 table with block scales in [2^-6, 2^-3]: real blocks and real scales, so the rows carry signal.</summary>
    private static unsafe Tensor RandomQ5_0(int rows, int cols, int seed)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.Q5_0);
        long bytes = DType.Q5_0.ComputeByteCount((long)rows * cols);
        byte* p = (byte*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < bytes; i++) p[i] = (byte)rng.Next(256);
        long blocks = bytes / DType.Q5_0.BlockByteSize;
        for (long blk = 0; blk < blocks; blk++)
            VulkanQuantGemvTests.WriteHalf(p + blk * DType.Q5_0.BlockByteSize, (float)Math.Pow(2, -6 + 3 * rng.NextDouble()));
        return t;
    }

    /// <summary>The token-level loop above can settle into a repeated token and hide a wrong layer, so this compares the continuous post-norm hidden of every step (distinct random input embeddings, advancing positions) between the eager forward and the replayed graph.</summary>
    [Theory]
    [InlineData(false, false, false)]
    [InlineData(true, false, false)]
    [InlineData(false, true, false)]
    [InlineData(false, true, true)]
    public void GraphDecodeEmbeds_ReplayedHiddenMatchesEager(bool bias, bool qkNorm, bool fullNorm)
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        IBackend b = backend;
        if (!b.LlmGraphDecodeSupported) { _out.WriteLine("SKIPPED: push descriptors unavailable"); return; }
        TransformerConfig cfg = new()
        {
            HiddenSize = 64, NumLayers = 3, NumHeads = 4, NumKvHeads = 2, HeadDim = 16,
            IntermediateSize = 96, VocabSize = 80, MaxPositionEmbeddings = 256, AttentionBias = bias, QkNorm = qkNorm, QkNormFullDim = fullNorm,
            TieWordEmbeddings = true,
        };
        Dictionary<string, Tensor> w = Weights(cfg, bias, qkNorm, fullNorm);
        float worst = ReplayedHiddenWorstDelta(b, cfg, w, steps: 12);
        _out.WriteLine($"worst |delta| over 12 replays: {worst:E3}");
        foreach (Tensor t in w.Values) t.Dispose();
        Assert.True(worst < 2e-3f, $"replayed graph diverges from eager by {worst:E3}");
    }

    /// <summary>Granite-shaped MoE decoder (every layer routed, top-2 of 4 experts, device-routed indexed stage). The router,
    /// top-k and combine run on the device inside the captured step, so a replay that baked a stale route would show here
    /// and not in the dense cases above. Compared per step against the eager forward on distinct embeddings.</summary>
    [Fact]
    public void GraphDecodeEmbeds_MoeLayerReplayMatchesEager()
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        IBackend b = backend;
        if (!b.LlmGraphDecodeSupported) { _out.WriteLine("SKIPPED: push descriptors unavailable"); return; }
        TransformerConfig cfg = new()
        {
            HiddenSize = 64, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = 16,
            IntermediateSize = 96, VocabSize = 80, MaxPositionEmbeddings = 256, TieWordEmbeddings = true,
            Moe = new MoeConfig { NumExperts = 4, NumExpertsPerTok = 2, MoeIntermediateSize = 32, NormTopKProb = true, Scoring = MoeScoring.Softmax },
        };
        Dictionary<string, Tensor> w = Weights(cfg, bias: false, qkNorm: false);
        float worst = ReplayedHiddenWorstDelta(b, cfg, w, steps: 12);
        _out.WriteLine($"worst |delta| over 12 MoE replays: {worst:E3}");
        foreach (Tensor t in w.Values) t.Dispose();
        // Exact: the replay runs the same kernels on the same inputs as the eager step, so any difference is a bug.
        Assert.True(worst == 0f, $"replayed MoE graph diverges from eager by {worst:E3}");
    }

    /// <summary>Runs <paramref name="steps"/> distinct-embedding decode steps eagerly and as one replayed graph and returns the
    /// largest per-element difference of the post-norm hidden state. The caller owns and disposes <paramref name="w"/>.</summary>
    private float ReplayedHiddenWorstDelta(IBackend b, TransformerConfig cfg, Dictionary<string, Tensor> w, int steps)
    {
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        // Expert banks must be placed as groups before any forward: a group with one member already resident is refused whole,
        // and MoeExpertsResident (the device-routed stage) then stays false.
        b.PreloadWeightGroups(model.EnumerateWeightGroups(includeRedundantSplits: false));
        const int maxLen = 48, prefill = 5;
        using FixedKvCache ce = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, maxLen);
        using FixedKvCache cg = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, maxLen);
        int[] prompt = [3, 17, 5, 41, 9];
        model.Forward(b, prompt, 0, ce).Dispose();
        model.Forward(b, prompt, 0, cg).Dispose();
        // Checked after the preload: the MoE gate asks whether the expert banks are resident on the device.
        Assert.True(model.SupportsGraphDecode(b), "the model must be eligible for the decode graph");
        (Tensor cos, Tensor sin) = model.EnsureRopeTableForGraphDecode(b, maxLen);
        ulong dp = b.AllocDevicePos();
        int h = cfg.HiddenSize;
        using Tensor inE = new(new TensorShape(1, 1, h), DType.F32);
        using Tensor outH = new(new TensorShape(1, 1, h), DType.F32);
        float[] got = new float[h];
        object? graph = null;
        float worst = 0f;
        try
        {
            for (int s = 0; s < steps; s++)
            {
                int pos = prefill + s;
                using Tensor e = Fill(new Tensor(new TensorShape(1, 1, h), DType.F32));
                using Tensor eager = model.ForwardEmbeds(b, e, 1, pos, ce);
                float[] want = new Span<float>((float*)eager.DataPointer, h).ToArray();
                b.CopyInto(inE, e);
                b.WriteDevicePos(dp, pos + 1, pos);
                if (graph is null)
                {
                    graph = b.CaptureGraph(() => model.ForwardGraphDecodeStepEmbeds(b, inE, cg, cos, sin, dp, outH));
                    Assert.NotNull(graph);
                }
                b.LaunchGraph(graph!);
                // Copy to a scratch tensor and read that: reading outH's host side would sync-and-free its device buffer, which the graph has baked in.
                using Tensor scratch = new(new TensorShape(1, 1, h), DType.F32);
                b.CopyTo(scratch, outH);
                new Span<float>((float*)scratch.DataPointer, h).CopyTo(got);
                for (int i = 0; i < h; i++) worst = MathF.Max(worst, MathF.Abs(want[i] - got[i]));
            }
        }
        finally
        {
            if (graph is not null) b.DisposeGraph(graph);
            b.FreeDevicePos(dp);
        }
        return worst;
    }

    [Fact]
    public void KvAppendAndAttention_RefuseUnsupportedShapesWithADevicePosition()
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        IBackend b = backend;
        ulong pos = b.AllocDevicePos();
        try
        {
            // An F16 cache cannot take the device-position append; the eager fallback would write slot 0 every step.
            using Tensor cacheF16 = new(new TensorShape(1, 2, 8, 16), DType.F16);
            using Tensor kv = new(new TensorShape(1, 2, 1, 16), DType.F32);
            Assert.Throws<NotSupportedException>(() => b.KvCacheAppendDev(cacheF16, kv, 0, pos));
            // Sliding-window attention is outside the device-position kernel.
            using Tensor q = new(new TensorShape(1, 4, 1, 16), DType.F32);
            using Tensor k = new(new TensorShape(1, 2, 8, 16), DType.F32);
            using Tensor v = new(new TensorShape(1, 2, 8, 16), DType.F32);
            using Tensor o = new(new TensorShape(1, 4, 1, 16), DType.F32);
            Assert.Throws<NotSupportedException>(() => b.FlashAttentionDev(o, q, k, v, 0, 2, true, 0, 0.25f, pos, 0f, 4));
        }
        finally { b.FreeDevicePos(pos); }
    }

    private static int ArgMaxLast(Tensor logits, int vocab)
    {
        float* p = (float*)logits.DataPointer + (logits.ElementCount - vocab);
        int best = 0;
        for (int i = 1; i < vocab; i++) if (p[i] > p[best]) best = i;
        return best;
    }

    private static List<int> RunEager(IBackend b, GenericTransformer model, TransformerConfig cfg, int[] prompt, int maxLen, int steps)
    {
        using FixedKvCache cache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, maxLen);
        List<int> outTokens = [];
        int next;
        using (Tensor hid = model.Forward(b, prompt, 0, cache))
        using (Tensor lg = model.ProjectLogits(b, hid, prompt.Length)) next = ArgMaxLast(lg, cfg.VocabSize);
        int pos = prompt.Length;
        for (int s = 0; s < steps; s++)
        {
            outTokens.Add(next);
            using Tensor hid = model.Forward(b, [next], pos, cache);
            using Tensor lg = model.ProjectLogits(b, hid, 1);
            next = ArgMaxLast(lg, cfg.VocabSize);
            pos++;
        }
        return outTokens;
    }

    private static List<int> RunGraph(IBackend b, GenericTransformer model, TransformerConfig cfg, int[] prompt, int maxLen, int steps)
    {
        using FixedKvCache cache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, maxLen);
        List<int> outTokens = [];
        int first;
        using (Tensor hid = model.Forward(b, prompt, 0, cache))
        using (Tensor lg = model.ProjectLogits(b, hid, prompt.Length)) first = ArgMaxLast(lg, cfg.VocabSize);

        // Same warm-up the production capture does: one throwaway step so every weight is resident before recording.
        using (FixedKvCache warm = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, 2))
        {
            using Tensor wh = model.Forward(b, [first], 0, warm);
            model.ProjectLogits(b, wh, 1).Dispose();
        }
        b.PreloadWeights(model.EnumerateWeights(includeRedundantSplits: false).ToList());
        Tensor embed = model.EnsureEmbedResidentForGraphDecode(b);
        // An untied quantized table embeds from a host-staged row, written before each replay as the session does.
        bool staged = model.StagesGraphDecodeEmbedRows(b);
        ulong embedRow = staged ? b.AllocDecodeEmbedRow(cfg.HiddenSize) : 0;
        (Tensor cos, Tensor sin) = model.EnsureRopeTableForGraphDecode(b, maxLen);
        ulong devicePos = b.AllocDevicePos();
        ulong tokenId = b.AllocDeviceTokenId();
        ulong history = b.AllocDeviceHistory(maxLen);
        ulong historyCount = b.AllocDeviceCounter();
        object? graph = null;
        try
        {
            int pos = prompt.Length;
            b.WriteDeviceTokenId(tokenId, first);
            b.WriteDevicePos(devicePos, pos + 1, pos);
            b.WriteDeviceCounter(historyCount, 0);
            graph = b.CaptureGraph(() => model.ForwardGraphDecodeStep(b, embed, cache, cos, sin, devicePos, tokenId, history, historyCount, 1.0f,
                embedRow: embedRow));
            Assert.NotNull(graph);
            int next = first;
            for (int s = 0; s < steps; s++)
            {
                outTokens.Add(next);
                if (staged) b.WriteDecodeEmbedRow(embedRow, model.GraphDecodeEmbedRow(next), cfg.HiddenSize);
                b.LaunchGraph(graph!);
                next = b.ReadDeviceTokenId(tokenId);
                pos++;
                b.WriteDevicePos(devicePos, pos + 1, pos);
            }
        }
        finally
        {
            if (graph is not null) b.DisposeGraph(graph);
            b.FreeDevicePos(devicePos);
            b.FreeDeviceTokenId(tokenId);
            b.FreeDeviceHistory(history);
            b.FreeDeviceCounter(historyCount);
            b.FreeDecodeEmbedRow(embedRow);
        }
        return outTokens;
    }

    private Dictionary<string, Tensor> Weights(TransformerConfig c, bool bias, bool qkNorm, bool fullNorm = false)
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
            if (bias)
            {
                w[$"{p}.self_attn.q_proj.bias"] = F1(qDim);
                w[$"{p}.self_attn.k_proj.bias"] = F1(kvDim);
                w[$"{p}.self_attn.v_proj.bias"] = F1(kvDim);
            }
            if (qkNorm)
            {
                w[$"{p}.self_attn.q_norm.weight"] = fullNorm ? F1(qDim, 1f) : Ones(c.HeadDim);
                w[$"{p}.self_attn.k_norm.weight"] = fullNorm ? F1(kvDim, 1f) : Ones(c.HeadDim);
            }
            if (c.Moe is null)
            {
                w[$"{p}.mlp.gate_proj.weight"] = F2(c.IntermediateSize, h);
                w[$"{p}.mlp.up_proj.weight"] = F2(c.IntermediateSize, h);
                w[$"{p}.mlp.down_proj.weight"] = F2(h, c.IntermediateSize);
            }
            else
            {
                MoeConfig m = c.Moe;
                w[$"{p}.mlp.gate.weight"] = F2(m.NumExperts, h);
                for (int x = 0; x < m.NumExperts; x++)
                {
                    w[$"{p}.mlp.experts.{x}.gate_proj.weight"] = Q8(m.MoeIntermediateSize, h);
                    w[$"{p}.mlp.experts.{x}.up_proj.weight"] = Q8(m.MoeIntermediateSize, h);
                    w[$"{p}.mlp.experts.{x}.down_proj.weight"] = Q8(h, m.MoeIntermediateSize);
                }
            }
        }
        return w;
    }
}
