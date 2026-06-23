using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>End-to-end MoE wiring through <see cref="GenericTransformer"/>: a full stack with MoE FFN layers
/// must run prefill + decode and stay batch-invariant (batched decode == per-sequence decode), proving the MoE
/// block plugs into both forward paths through the shared <c>Mlp()</c> seam.</summary>
public sealed unsafe class MoeIntegrationTests
{
    private static uint _rng = 0xBEEF77u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }

    [Fact]
    public void MoeTransformer_BatchedDecode_MatchesPerSequence()
    {
        MoeConfig moe = new()
        {
            NumExperts = 4, NumExpertsPerTok = 2, MoeIntermediateSize = 20,
            NormTopKProb = true, Scoring = MoeScoring.Softmax, SharedExpertIntermediateSize = 16,
        };
        TransformerConfig cfg = new()
        {
            HiddenSize = 16, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = 4,
            IntermediateSize = 32, VocabSize = 32, MaxPositionEmbeddings = 64,
            AttentionBias = false, QkNorm = false, Moe = moe,
        };
        Dictionary<string, Tensor> w = Weights(cfg);
        int h = cfg.HiddenSize;
        int[] promptLens = [4, 2];
        int bn = promptLens.Length;

        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");

        FixedKvCache[] refCaches = new FixedKvCache[bn];
        FixedKvCache[] batCaches = new FixedKvCache[bn];
        for (int s = 0; s < bn; s++)
        {
            refCaches[s] = new FixedKvCache(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, 16);
            batCaches[s] = new FixedKvCache(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, 16);
            using Tensor prompt = Fill(new Tensor(new TensorShape(1, promptLens[s], h), DType.F32));
            using Tensor a = model.ForwardEmbeds(backend, prompt, promptLens[s], 0, refCaches[s]);
            using Tensor b = model.ForwardEmbeds(backend, prompt, promptLens[s], 0, batCaches[s]);
            // prefill output must be finite (exercises MoE in the Forward path)
            float* pa = (float*)a.DataPointer;
            for (long i = 0; i < a.ElementCount; i++) Assert.True(float.IsFinite(pa[i]), "prefill produced non-finite output");
        }

        using Tensor batchEmbeds = new(new TensorShape(1, bn, h), DType.F32);
        float* be = (float*)batchEmbeds.DataPointer;
        Tensor[] perSeq = new Tensor[bn];
        int[] positions = new int[bn];
        for (int s = 0; s < bn; s++)
        {
            perSeq[s] = Fill(new Tensor(new TensorShape(1, 1, h), DType.F32));
            float* pe = (float*)perSeq[s].DataPointer;
            for (int j = 0; j < h; j++) be[s * h + j] = pe[j];
            positions[s] = refCaches[s].CurrentLength;
        }

        using Tensor batched = model.ForwardBatchDecode(backend, batchEmbeds, positions, batCaches);
        float* bp = (float*)batched.DataPointer;
        for (int s = 0; s < bn; s++)
        {
            using Tensor refOut = model.ForwardEmbeds(backend, perSeq[s], 1, refCaches[s].CurrentLength, refCaches[s]);
            float* rp = (float*)refOut.DataPointer;
            for (int j = 0; j < h; j++)
                Assert.True(MathF.Abs(rp[j] - bp[s * h + j]) <= 1e-4f,
                    $"MoE seq {s}: batched vs per-seq mismatch at {j} ({rp[j]} vs {bp[s * h + j]})");
            perSeq[s].Dispose();
        }

        foreach (FixedKvCache c in refCaches) c.Dispose();
        foreach (FixedKvCache c in batCaches) c.Dispose();
        foreach (Tensor t in w.Values) t.Dispose();
    }

    private static Dictionary<string, Tensor> Weights(TransformerConfig c)
    {
        int h = c.HiddenSize, qDim = c.QDim, kvDim = c.KvDim;
        MoeConfig m = c.Moe!;
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
            w[$"{p}.mlp.gate.weight"] = F2(m.NumExperts, h);
            w[$"{p}.mlp.shared_expert.gate_proj.weight"] = F2(m.SharedExpertIntermediateSize, h);
            w[$"{p}.mlp.shared_expert.up_proj.weight"] = F2(m.SharedExpertIntermediateSize, h);
            w[$"{p}.mlp.shared_expert.down_proj.weight"] = F2(h, m.SharedExpertIntermediateSize);
            w[$"{p}.mlp.shared_expert_gate.weight"] = F2(1, h);
            for (int e = 0; e < m.NumExperts; e++)
            {
                w[$"{p}.mlp.experts.{e}.gate_proj.weight"] = F2(m.MoeIntermediateSize, h);
                w[$"{p}.mlp.experts.{e}.up_proj.weight"] = F2(m.MoeIntermediateSize, h);
                w[$"{p}.mlp.experts.{e}.down_proj.weight"] = F2(h, m.MoeIntermediateSize);
            }
        }
        return w;
    }
}
