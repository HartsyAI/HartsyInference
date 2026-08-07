using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Layer-split placement contracts: a staged forward over a shared KV cache must reproduce the
/// unstaged forward exactly (same math, split loop, host-staged boundary), and the per-stage weight
/// enumeration must tile to exactly the full weight set — a dropped tensor silently becomes a per-op PCIe
/// re-upload on the stage that needed it. CPU backend, synthetic weights: Unit tier.</summary>
public sealed unsafe class LlmPlacementTests
{
    private static uint _rng = 0x9E377u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }
    private static Tensor Embeds(int t, int h) => Fill(new Tensor(new TensorShape(1, t, h), DType.F32));

    private static TransformerConfig Config() => new()
    {
        HiddenSize = 16, NumLayers = 4, NumHeads = 4, NumKvHeads = 2, HeadDim = 8,
        IntermediateSize = 32, VocabSize = 32, MaxPositionEmbeddings = 64, AttentionBias = false, QkNorm = true,
    };

    private static Dictionary<string, Tensor> Weights(TransformerConfig c)
    {
        int h = c.HiddenSize, qDim = c.QDim, kvDim = c.KvDim, hd = c.HeadDim;
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
            w[$"{p}.self_attn.q_norm.weight"] = Ones(hd);
            w[$"{p}.self_attn.k_norm.weight"] = Ones(hd);
            w[$"{p}.mlp.gate_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.up_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.down_proj.weight"] = F2(h, c.IntermediateSize);
        }
        return w;
    }

    [Fact]
    public void ForwardEmbedsStaged_TwoStages_MatchesUnstaged_AcrossPrefillAndDecode()
    {
        TransformerConfig cfg = Config();
        Dictionary<string, Tensor> w = Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");

        using KvCache unstagedCache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim);
        using KvCache stagedCache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim);
        LlmPlacement placement = new([new LlmStage(backend, 0, 2), new LlmStage(backend, 2, 4)]);

        // Prefill 4 tokens, then two 1-token decode steps — the shapes the real pipeline drives.
        int[] steps = [4, 1, 1];
        int pos = 0;
        foreach (int t in steps)
        {
            using Tensor embeds = Embeds(t, cfg.HiddenSize);
            using Tensor embedsCopy = new(embeds.Shape, DType.F32);
            backend.CopyTo(embedsCopy, embeds);

            using Tensor expected = model.ForwardEmbeds(backend, embeds, t, pos, unstagedCache);
            using Tensor staged = model.ForwardEmbedsStaged(placement, embedsCopy, t, pos, stagedCache);

            Assert.Equal(expected.Shape, staged.Shape);
            float* ep = (float*)expected.DataPointer;
            float* sp = (float*)staged.DataPointer;
            for (long i = 0; i < expected.ElementCount; i++)
            {
                Assert.Equal(ep[i], sp[i], 5);
            }
            pos += t;
            // advanceCache contract: exactly one advance per staged call, matching the unstaged path.
            Assert.Equal(unstagedCache.CurrentLength, stagedCache.CurrentLength);
            Assert.Equal(pos, stagedCache.CurrentLength);
        }

        foreach (Tensor t in w.Values) t.Dispose();
    }

    /// <summary>Regression (Phase 0, fixed 2026-08-06): the embedding preamble — absolute position embeddings
    /// (GPT-2/StarCoder) and the BLOOM word-embedding LayerNorm — used to run unconditionally at the top of
    /// <c>ForwardEmbeds</c>, so a staged run re-applied both once per stage to what was by then a HIDDEN state,
    /// silently corrupting output for those lineages (weight tiling already assumed first-stage-only). Same
    /// staged-vs-unstaged parity harness as the base test, with both preamble features enabled.</summary>
    [Fact]
    public void ForwardEmbedsStaged_PosEmbedAndEmbedNorm_AppliedFirstStageOnly()
    {
        TransformerConfig cfg = Config() with { AbsolutePositionEmbeddings = true, EmbeddingLayerNorm = true };
        Dictionary<string, Tensor> w = Weights(cfg);
        w["model.embed_positions.weight"] = F2(cfg.MaxPositionEmbeddings, cfg.HiddenSize);
        w["model.embed_norm.weight"] = Ones(cfg.HiddenSize);
        w["model.embed_norm.bias"] = Fill(new Tensor(new TensorShape(cfg.HiddenSize), DType.F32));
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");

        using KvCache unstagedCache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim);
        using KvCache stagedCache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim);
        LlmPlacement placement = new([new LlmStage(backend, 0, 2), new LlmStage(backend, 2, 4)]);

        int[] steps = [4, 1, 1];
        int pos = 0;
        foreach (int t in steps)
        {
            using Tensor embeds = Embeds(t, cfg.HiddenSize);
            using Tensor embedsCopy = new(embeds.Shape, DType.F32);
            backend.CopyTo(embedsCopy, embeds);

            using Tensor expected = model.ForwardEmbeds(backend, embeds, t, pos, unstagedCache);
            using Tensor staged = model.ForwardEmbedsStaged(placement, embedsCopy, t, pos, stagedCache);

            Assert.Equal(expected.Shape, staged.Shape);
            float* ep = (float*)expected.DataPointer;
            float* sp = (float*)staged.DataPointer;
            for (long i = 0; i < expected.ElementCount; i++)
            {
                Assert.Equal(ep[i], sp[i], 5);
            }
            pos += t;
        }

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void EnumerateStageWeights_FullTiling_EqualsEnumerateWeights()
    {
        TransformerConfig cfg = Config();
        Dictionary<string, Tensor> w = Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");

        HashSet<Tensor> full = new(model.EnumerateWeights(), ReferenceEqualityComparer.Instance);
        HashSet<Tensor> tiled = new(ReferenceEqualityComparer.Instance);
        foreach (Tensor t in model.EnumerateStageWeights(0, 2, isFirstStage: true, isLastStage: false)) tiled.Add(t);
        foreach (Tensor t in model.EnumerateStageWeights(2, 4, isFirstStage: false, isLastStage: true)) tiled.Add(t);

        Assert.True(full.SetEquals(tiled),
            $"stage tiling yielded {tiled.Count} distinct tensors, full enumeration {full.Count} — a stage dropped or duplicated weights");

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void LlmPlacement_RejectsGapsAndOverlaps()
    {
        using CpuBackend backend = new();
        Assert.Throws<ArgumentException>(() => new LlmPlacement([new LlmStage(backend, 0, 2), new LlmStage(backend, 3, 4)]));
        Assert.Throws<ArgumentException>(() => new LlmPlacement([new LlmStage(backend, 0, 2), new LlmStage(backend, 1, 4)]));
        Assert.Throws<ArgumentException>(() => new LlmPlacement([new LlmStage(backend, 1, 4)]));
        Assert.Throws<ArgumentException>(() => new LlmPlacement([]));
    }

    /// <summary>Adds a gated cross-attention layer's weights (mllama) at <paramref name="layerIndex"/> — the
    /// keys <c>MllamaCrossAttentionLayer.LoadWeights</c> reads, mirroring
    /// <c>MllamaCrossAttentionTests.Weights</c>. The base <see cref="Weights"/> call already populated (unused)
    /// self-attn/mlp keys for every layer index, including this one — <c>Layer.LoadWeights</c> checks
    /// <c>IsCrossAttnLayer</c> first and never reads them for a cross-attention layer, so no cleanup is needed.</summary>
    private static void AddCrossAttnWeights(Dictionary<string, Tensor> w, TransformerConfig c, int layerIndex)
    {
        int h = c.HiddenSize, hd = c.HeadDim, hq = c.NumHeads, hkv = c.NumKvHeads;
        string p = $"model.layers.{layerIndex}";
        w[$"{p}.cross_attn.q_proj.weight"] = F2(hq * hd, h);
        w[$"{p}.cross_attn.k_proj.weight"] = F2(hkv * hd, h);
        w[$"{p}.cross_attn.v_proj.weight"] = F2(hkv * hd, h);
        w[$"{p}.cross_attn.o_proj.weight"] = F2(h, hq * hd);
        w[$"{p}.cross_attn.q_norm.weight"] = Ones(hd);
        w[$"{p}.cross_attn.k_norm.weight"] = Ones(hd);
        w[$"{p}.cross_attn_attn_gate"] = ScalarT(0.3f);
        w[$"{p}.cross_attn_mlp_gate"] = ScalarT(-0.2f);
    }

    private static Tensor ScalarT(float v) { Tensor t = new(new TensorShape(1), DType.F32); *(float*)t.DataPointer = v; return t; }

    /// <summary>Real bug this used to guard (Phase 1, fixed 2026-08-05): <see cref="GenericTransformer.ForwardEmbedsStaged"/>
    /// used to silently take the <c>CopyHidden</c> branch for mllama gated cross-attention layers instead of
    /// throwing when an image was present — wrong output, no signal. That guard threw unconditionally on ANY
    /// staged model with cross-attention layers, blocking the feature outright. Now built out for real (Phase 5,
    /// same day): <see cref="GenericTransformer.ForwardEmbedsStaged"/> takes <c>crossStates</c>/<c>crossLen</c> and
    /// peer-copies the vision features onto every stage that owns a cross-attention layer via
    /// <c>IBackend.CopyFromPeer</c> — so the old negative-path guard test is obsolete; this positive-path
    /// test replaces it, asserting staged bit-parity against the unstaged path across TWO DISTINCT backend
    /// instances (exercising the actual peer-copy branch, not just the same-instance early-out).</summary>
    [Fact]
    public void ForwardEmbedsStaged_MllamaCrossAttention_MatchesUnstaged_AcrossPrefillAndDecode()
    {
        TransformerConfig cfg = Config() with { CrossAttnLayers = new HashSet<int> { 1, 3 } };
        Dictionary<string, Tensor> w = Weights(cfg);
        AddCrossAttnWeights(w, cfg, 1);
        AddCrossAttnWeights(w, cfg, 3);
        using CpuBackend backendA = new();
        using CpuBackend backendB = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");

        const int visLen = 6;
        using Tensor crossStates = Fill(new Tensor(new TensorShape(1, visLen, cfg.HiddenSize), DType.F32));

        using KvCache unstagedCache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim);
        using KvCache stagedCache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim);
        // Layer 1 lives in stage 0 (same backend as crossStates' home — the reuse branch); layer 3 lives in
        // stage 1 on a DIFFERENT backend instance — forces the actual CopyFromPeer path.
        LlmPlacement placement = new([new LlmStage(backendA, 0, 2), new LlmStage(backendB, 2, 4)]);

        int[] steps = [4, 1, 1];
        int pos = 0;
        foreach (int t in steps)
        {
            using Tensor embeds = Embeds(t, cfg.HiddenSize);
            using Tensor embedsCopy = new(embeds.Shape, DType.F32);
            backendA.CopyTo(embedsCopy, embeds);

            using Tensor expected = model.ForwardEmbeds(backendA, embeds, t, pos, unstagedCache, crossStates: crossStates, crossLen: visLen);
            using Tensor staged = model.ForwardEmbedsStaged(placement, embedsCopy, t, pos, stagedCache, crossStates: crossStates, crossLen: visLen);

            Assert.Equal(expected.Shape, staged.Shape);
            float* ep = (float*)expected.DataPointer;
            float* sp = (float*)staged.DataPointer;
            for (long i = 0; i < expected.ElementCount; i++)
            {
                Assert.Equal(ep[i], sp[i], 5);
            }
            pos += t;
            Assert.Equal(unstagedCache.CurrentLength, stagedCache.CurrentLength);
            Assert.Equal(pos, stagedCache.CurrentLength);
        }

        foreach (Tensor t in w.Values) t.Dispose();
    }
}
