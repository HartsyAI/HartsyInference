using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Models.Shared;

/// <summary>Attention over a KV cache whose per-head output is scaled by <c>sigmoid(gate logit)</c> before the output projection (LFM3, Spark).</summary>
internal static class HeadGatedAttention
{
    /// <summary>Attends rotated <paramref name="q"/> <c>[1, seq, heads, headDim]</c>, <paramref name="k"/> and <paramref name="v"/> <c>[1, seq, kvHeads, headDim]</c> over <paramref name="kv"/> from <paramref name="position"/>, gates each head by <paramref name="gateLogits"/> <c>[1, seq, heads]</c>, and projects with <paramref name="outProjection"/>. Disposes q, k, v and the logits; the caller owns the result.</summary>
    /// <param name="slidingWindow">Keys each query sees, itself included; 0 for the whole sequence.</param>
    public static Tensor Attend(IBackend backend, AttentionLayer attention, FixedKvCache kv, int layer, Tensor q, Tensor k, Tensor v,
        Tensor gateLogits, int position, int slidingWindow, Tensor outProjection)
    {
        int seq = (int)q.Shape[1], hq = (int)q.Shape[2], hkv = (int)k.Shape[2], hd = (int)q.Shape[3];
        Tensor qMh = new(new TensorShape(1, hq, seq, hd), DType.F32);
        Tensor kMh = new(new TensorShape(1, hkv, seq, hd), DType.F32);
        Tensor vMh = new(kMh.Shape, DType.F32);
        backend.Permute0213(qMh, q, seq, hq, hd);
        backend.Permute0213(kMh, k, seq, hkv, hd);
        backend.Permute0213(vMh, v, seq, hkv, hd);
        q.Dispose();
        k.Dispose();
        v.Dispose();
        KvAttentionCall call = new() { KvLength = position + seq, QueryOffset = position, SlidingWindow = slidingWindow };
        Tensor attended = attention.AttendOverKvHeadMajor(backend, qMh, kMh, vMh, kv, layer, layer, call);

        Tensor flat = new(new TensorShape(1, seq, hq * hd), DType.F32);
        backend.Permute0213(flat, attended, hq, seq, hd);
        attended.Dispose();
        // The shared head gate multiplies by 2*sigmoid; halving restores sigmoid.
        backend.Ltx2HeadGate(flat, gateLogits, seq, hq, hd);
        gateLogits.Dispose();
        Tensor gated = new(flat.Shape, DType.F32);
        backend.Scale(gated, flat, 0.5f);
        flat.Dispose();
        Tensor output = new(new TensorShape(1, seq, outProjection.Shape[0]), DType.F32);
        backend.Linear(output, gated, outProjection, null);
        gated.Dispose();
        return output;
    }
}
