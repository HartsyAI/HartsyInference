using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Layers.Attention;

public sealed partial class AttentionLayer
{
    /// <summary>Appends head-major K/V to <paramref name="source"/> (a null <paramref name="kMh"/> reads <paramref name="kvLayer"/> as a donor), then
    /// attends <paramref name="qMh"/> over its buffers; without a source it attends over <paramref name="kMh"/>/<paramref name="vMh"/>. Disposes the operands
    /// <paramref name="consume"/> names once read. Returns the head-major output; the caller owns it.</summary>
    public Tensor AttendOverKvHeadMajor(IBackend backend, Tensor qMh, Tensor? kMh, Tensor? vMh, IAttentionKvSource? source,
        int layer, int kvLayer, in KvAttentionCall call, OperandOwnership consume = OperandOwnership.All)
    {
        bool ownQ = (consume & OperandOwnership.Query) != 0;
        bool ownK = kMh is not null && (consume & OperandOwnership.Key) != 0;
        bool ownV = vMh is not null && (consume & OperandOwnership.Value) != 0;
        Tensor? attn = null;
        try
        {
            Tensor keys, values;
            if (source is null)
            {
                if (kMh is null || vMh is null)
                    throw new ArgumentException("Attending without a key/value source needs the key and value operands.");
                keys = kMh;
                values = vMh;
            }
            else
            {
                if (kMh is not null)
                {
                    if (vMh is null) throw new ArgumentException("A key without a value cannot be appended.", nameof(vMh));
                    source.AppendKv(backend, layer, kMh, vMh);
                    if (ownK) { kMh.Dispose(); ownK = false; }
                    if (ownV) { vMh.Dispose(); ownV = false; }
                }
                keys = source.KeyBuffer(kvLayer);
                values = source.ValueBuffer(kvLayer);
            }
            CheckKvOperands(qMh, keys);
            if (call.KvLength > keys.Shape[2] || call.KvLength > values.Shape[2])
                throw new ArgumentException($"{call.KvLength} keys do not fit buffers of {keys.Shape[2]} and {values.Shape[2]} rows.", nameof(call));
            attn = new Tensor(new TensorShape(qMh.Shape[0], _config.Heads, qMh.Shape[2], _config.HeadDim), _config.ComputeDType ?? qMh.DType);
            AttentionCall flash = AttentionCall.FromFlashAttention(attn, qMh, keys, values, call.KvLength, !call.Bidirectional,
                call.QueryOffset, _config.EffectiveScale, call.Softcap, call.Sinks, call.SlidingWindow, call.AlibiSlopes);
            if (call.Bias is not null) flash = flash with { Mask = flash.Mask with { Bias = call.Bias } };
            backend.Attention(in flash);
            if (ownQ) qMh.Dispose();
            if (source is null)
            {
                if (ownK) kMh!.Dispose();
                if (ownV) vMh!.Dispose();
            }
            return attn;
        }
        catch
        {
            attn?.Dispose();
            if (ownQ) qMh.Dispose();
            if (ownK) kMh!.Dispose();
            if (ownV) vMh!.Dispose();
            throw;
        }
    }

    /// <summary><see cref="AttendOverKvHeadMajor"/> over token-major post-rotation <paramref name="q"/> <c>[1, t, heads, D]</c> and
    /// <paramref name="k"/>/<paramref name="v"/> <c>[1, t, kvHeads, D]</c> (null for a KV-sharing layer); all three are permuted head-major and disposed.</summary>
    public Tensor AttendOverKv(IBackend backend, Tensor q, Tensor? k, Tensor? v, IAttentionKvSource source,
        int layer, int kvLayer, in KvAttentionCall call)
    {
        if (q.Shape.Rank != 4 || q.Shape[0] != 1)
            throw new ArgumentException($"AttendOverKv takes one batch item of [1, t, heads, D] queries; got {q.Shape}.", nameof(q));
        if ((k is null) != (v is null))
            throw new ArgumentException("Key and value are both given or both null.", nameof(k));
        int t = (int)q.Shape[1];
        Tensor? qMh = null, kMh = null, vMh = null;
        // The operands are consumed on every path: here until the permutes are done, then by AttendOverKvHeadMajor.
        try
        {
            qMh = new Tensor(new TensorShape(1, _config.Heads, t, _config.HeadDim), _config.ComputeDType ?? q.DType);
            backend.Permute0213(qMh, q, t, _config.Heads, _config.HeadDim);
            q.Dispose();
            if (k is not null)
            {
                kMh = new Tensor(new TensorShape(1, _config.KvHeads, t, _config.HeadDim), _config.ComputeDType ?? k.DType);
                vMh = new Tensor(new TensorShape(1, _config.KvHeads, t, _config.HeadDim), _config.ComputeDType ?? v!.DType);
                backend.Permute0213(kMh, k, t, _config.KvHeads, _config.HeadDim);
                backend.Permute0213(vMh, v!, t, _config.KvHeads, _config.HeadDim);
                k.Dispose();
                v!.Dispose();
            }
        }
        catch
        {
            q.Dispose();
            k?.Dispose();
            v?.Dispose();
            qMh?.Dispose();
            kMh?.Dispose();
            vMh?.Dispose();
            throw;
        }
        return AttendOverKvHeadMajor(backend, qMh, kMh, vMh, source, layer, kvLayer, in call);
    }

    /// <summary>Attends <paramref name="qMh"/> over buffers the model already scattered into, positions read on the device from
    /// <paramref name="devicePositions"/>: one output allocation and one call, so it is capture-safe.</summary>
    public Tensor AttendDevice(IBackend backend, Tensor qMh, Tensor keys, Tensor values, ulong devicePositions,
        in KvAttentionCall call, bool consumeQuery = true)
    {
        if (call.Sinks is not null || call.AlibiSlopes is not null || call.Bias is not null)
            throw new NotSupportedException("The device-position attention carries no sinks, ALiBi or bias.");
        CheckKvOperands(qMh, keys);
        Tensor attn = new(new TensorShape(qMh.Shape[0], _config.Heads, qMh.Shape[2], _config.HeadDim), _config.ComputeDType ?? qMh.DType);
        try
        {
            backend.Attention(AttentionCall.FromFlashAttentionDev(attn, qMh, keys, values, call.KvLength, !call.Bidirectional,
                call.QueryOffset, _config.EffectiveScale, devicePositions, call.Softcap, call.SlidingWindow));
        }
        catch
        {
            attn.Dispose();
            if (consumeQuery) qMh.Dispose();
            throw;
        }
        if (consumeQuery) qMh.Dispose();
        return attn;
    }

    // The Flash route takes the GQA group from the buffers' head counts, so a mis-built cache would change it silently.
    private void CheckKvOperands(Tensor qMh, Tensor keys)
    {
        if (qMh.Shape.Rank != 4 || qMh.Shape[1] != _config.Heads || qMh.Shape[3] != _config.HeadDim)
            throw new ArgumentException($"The attention takes [B, {_config.Heads}, S, {_config.HeadDim}] queries; got {qMh.Shape}.", nameof(qMh));
        if (keys.Shape.Rank != 4 || keys.Shape[1] != _config.KvHeads)
            throw new ArgumentException($"The key buffer must hold {_config.KvHeads} key/value heads; got {keys.Shape}.", nameof(keys));
    }
}
