using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>The continuous-batching decode path (<see cref="GenericTransformer.ForwardBatchDecode"/>) must be
/// numerically identical to decoding each sequence on its own via <see cref="GenericTransformer.ForwardEmbeds"/>
/// — batching is a throughput change, not a math change. Uses ragged prompt lengths so per-sequence positions,
/// KV lengths, and GQA all differ within the batch.</summary>
public sealed unsafe class ContinuousBatchTests
{
    private static uint _rng = 0x51A7Eu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }

    [Fact]
    public void BatchedDecode_MatchesPerSequenceDecode()
    {
        TransformerConfig cfg = new()
        {
            HiddenSize = 16, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = 4,
            IntermediateSize = 32, VocabSize = 32, MaxPositionEmbeddings = 64, AttentionBias = true, QkNorm = false,
        };
        Dictionary<string, Tensor> w = Weights(cfg);
        int h = cfg.HiddenSize;
        int[] promptLens = [3, 1, 5];   // ragged
        int bn = promptLens.Length;

        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");

        // Two independent cache sets prefilled identically: one for the per-sequence reference, one batched.
        FixedKvCache[] refCaches = new FixedKvCache[bn];
        FixedKvCache[] batCaches = new FixedKvCache[bn];
        for (int s = 0; s < bn; s++)
        {
            refCaches[s] = new FixedKvCache(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, 16);
            batCaches[s] = new FixedKvCache(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, 16);
            using Tensor prompt = Fill(new Tensor(new TensorShape(1, promptLens[s], h), DType.F32));
            model.ForwardEmbeds(backend, prompt, promptLens[s], 0, refCaches[s]).Dispose();
            model.ForwardEmbeds(backend, prompt, promptLens[s], 0, batCaches[s]).Dispose();
        }

        // Two decode steps; fresh random token embeds each step (shared between both paths).
        for (int step = 0; step < 2; step++)
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
                        $"step {step} seq {s}: mismatch at {j} ({rp[j]} vs {bp[s * h + j]})");
                perSeqEmbeds[s].Dispose();
            }
        }

        foreach (FixedKvCache c in refCaches) c.Dispose();
        foreach (FixedKvCache c in batCaches) c.Dispose();
        foreach (Tensor t in w.Values) t.Dispose();
    }

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
}
