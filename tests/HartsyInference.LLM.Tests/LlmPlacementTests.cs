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
}
