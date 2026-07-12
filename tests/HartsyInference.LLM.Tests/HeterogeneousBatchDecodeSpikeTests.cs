using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Phase 0 de-risking spike for the graph-decode-into-scheduler retrofit (see
/// docs/Checklists/LLM_DECODE_PERF_GRIND.md's "NEW PLAN" section): proves
/// <see cref="GenericTransformer.ForwardBatchDecode"/>'s <c>IKvCache[]</c> parameter — implementation-agnostic
/// by inspection (every access goes through the interface, no cast to a concrete type) — is ALSO correct in
/// practice with a genuinely HETEROGENEOUS array (some slots <see cref="FixedKvCache"/>, others
/// <see cref="PagedKvCache"/>) in the SAME call. This has never been exercised before: every existing test/
/// production call site uses a uniform cache type per call (<see cref="ContinuousBatchTests"/> — all
/// <see cref="FixedKvCache"/>; <see cref="DynamicBatchScheduler"/> — all <see cref="PagedKvCache"/>). Gate for
/// the whole retrofit: if this fails, mixing a solo graph-eligible sequence's <see cref="FixedKvCache"/> into
/// an eager batched round alongside other sequences' <see cref="PagedKvCache"/> instances is unsafe and the
/// retrofit needs a different design.</summary>
public sealed unsafe class HeterogeneousBatchDecodeSpikeTests
{
    private static uint _rng = 0x7A3E9C1u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }

    private static TransformerConfig Cfg() => new()
    {
        HiddenSize = 16, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = 4,
        IntermediateSize = 32, VocabSize = 32, MaxPositionEmbeddings = 64, AttentionBias = true, QkNorm = false,
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
    public void HeterogeneousArray_FixedAndPaged_MatchesPerSequenceReference()
    {
        TransformerConfig cfg = Cfg();
        Dictionary<string, Tensor> w = Weights(cfg);
        int h = cfg.HiddenSize;
        // 4 sequences, ragged lengths, cache TYPE deliberately alternates Fixed/Paged/Fixed/Paged so the
        // batched call's IKvCache[] is genuinely mixed within one ForwardBatchDecode invocation, not just
        // across separate calls.
        int[] promptLens = [3, 1, 5, 2];
        int bn = promptLens.Length;
        bool[] isFixed = [true, false, true, false];

        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        using PagedKvPool pool = new(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, pageSize: 4, maxPages: 64);

        // Reference: each sequence decoded ALONE through its own FixedKvCache (the oracle every existing
        // cache-correctness test in this repo already uses — PagedKvCacheTests already proves PagedKvCache
        // matches this same oracle for a single sequence; what's new here is proving MIXING doesn't break it).
        FixedKvCache[] refCaches = new FixedKvCache[bn];
        IKvCache[] batCaches = new IKvCache[bn];
        for (int s = 0; s < bn; s++)
        {
            refCaches[s] = new FixedKvCache(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, 16);
            batCaches[s] = isFixed[s] ? new FixedKvCache(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, 16) : new PagedKvCache(pool);
            using Tensor prompt = Fill(new Tensor(new TensorShape(1, promptLens[s], h), DType.F32));
            model.ForwardEmbeds(backend, prompt, promptLens[s], 0, refCaches[s]).Dispose();
            model.ForwardEmbeds(backend, prompt, promptLens[s], 0, batCaches[s]).Dispose();
        }

        for (int step = 0; step < 3; step++)
        {
            using Tensor batchEmbeds = new(new TensorShape(1, bn, h), DType.F32);
            float* be = (float*)batchEmbeds.DataPointer;
            Tensor[] perSeqEmbeds = new Tensor[bn];
            int[] positions = new int[bn];
            for (int s = 0; s < bn; s++)
            {
                perSeqEmbeds[s] = Fill(new Tensor(new TensorShape(1, 1, h), DType.F32));
                float* pe = (float*)perSeqEmbeds[s].DataPointer;
                for (int j = 0; j < h; j++) be[s * h + j] = pe[j];
                positions[s] = refCaches[s].CurrentLength;
            }

            using Tensor batched = model.ForwardBatchDecode(backend, batchEmbeds, positions, batCaches);
            float* bp = (float*)batched.DataPointer;
            for (int s = 0; s < bn; s++)
            {
                using Tensor refOut = model.ForwardEmbeds(backend, perSeqEmbeds[s], 1, refCaches[s].CurrentLength, refCaches[s]);
                float* rp = (float*)refOut.DataPointer;
                for (int j = 0; j < h; j++)
                    Assert.True(MathF.Abs(rp[j] - bp[s * h + j]) <= 1e-4f,
                        $"step {step} seq {s} ({(isFixed[s] ? "Fixed" : "Paged")}): mismatch at {j} ({rp[j]} vs {bp[s * h + j]})");
                perSeqEmbeds[s].Dispose();
            }
        }

        foreach (FixedKvCache c in refCaches) c.Dispose();
        foreach (IKvCache c in batCaches) c.Dispose();
        foreach (Tensor t in w.Values) t.Dispose();
    }
}
