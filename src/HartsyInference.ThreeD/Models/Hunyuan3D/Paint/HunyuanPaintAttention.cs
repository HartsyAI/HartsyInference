using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>A bias-free multi-head attention with an output projection, as diffusers' <c>Attention</c> builds it for the UNet: K and V read the
/// <paramref name="kv"/> tensor, an optional interleaved-pair rotation turns Q and K by per-token tables. Activations are F32.</summary>
internal sealed class HunyuanPaintAttention
{
    private readonly int _heads, _headDim;
    private readonly AttentionLayer _layer;
    private readonly Tensor _qW, _kW, _vW, _outW, _outB;

    internal HunyuanPaintAttention(int heads, int headDim, Tensor qW, Tensor kW, Tensor vW, Tensor outW, Tensor outB)
    {
        _heads = heads;
        _headDim = headDim;
        _qW = qW;
        _kW = kW;
        _vW = vW;
        _outW = outW;
        _outB = outB;
        _layer = new AttentionLayer(new AttentionLayerConfig { Heads = heads, KvHeads = heads, HeadDim = headDim, ComputeDType = DType.F32, AllowF16 = true });
    }

    /// <summary>The weights this layer holds, for ownership by the loader.</summary>
    internal IEnumerable<Tensor> Weights => [_qW, _kW, _vW, _outW, _outB];

    /// <summary>Attends <paramref name="q"/> <c>[B, Sq, C]</c> over <paramref name="kv"/> <c>[B, Skv, Ckv]</c>; with <paramref name="rotation"/> (cos and sin
    /// <c>[S, headDim]</c> F32, duplicated per pair) the batch must be 1 and both sequences equal. Returns <c>[B, Sq, C]</c>; the inputs are borrowed.</summary>
    internal Tensor Run(IBackend backend, Tensor q, Tensor kv, (Tensor Cos, Tensor Sin)? rotation = null)
    {
        int batch = (int)q.Shape[0], sq = (int)q.Shape[1], skv = (int)kv.Shape[1], width = _heads * _headDim;
        if (kv.Shape[0] != batch)
            throw new ArgumentException($"Query batch {batch} and key/value batch {kv.Shape[0]} differ.", nameof(kv));
        TensorShape qHeads = new(batch, sq, _heads, _headDim), kHeads = new(batch, skv, _heads, _headDim);
        Tensor qp = new(qHeads, DType.F32), kp = new(kHeads, DType.F32), vp = new(kHeads, DType.F32);
        backend.Linear(qp, q, _qW, null);
        backend.Linear(kp, kv, _kW, null);
        backend.Linear(vp, kv, _vW, null);
        if (rotation is { } table)
        {
            if (batch != 1 || sq != skv)
                throw new ArgumentException("A rotated attention runs on one sequence.", nameof(rotation));
            backend.Rotary(RotaryCall.Rows(qp, table.Cos, table.Sin, sq, _heads, _headDim, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPair));
            backend.Rotary(RotaryCall.Rows(kp, table.Cos, table.Sin, skv, _heads, _headDim, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPair));
        }
        Tensor qm = Heads(backend, qp, batch, sq), km = Heads(backend, kp, batch, skv), vm = Heads(backend, vp, batch, skv);
        Tensor attn = _layer.AttendHeadMajorOperands(backend, qm, km, vm, default, OperandOwnership.All);
        Tensor merged = _layer.MergeHeads(backend, attn, new TensorShape(batch, sq, width));
        Tensor output = new(new TensorShape(batch, sq, (int)_outW.Shape[0]), DType.F32);
        backend.Linear(output, merged, _outW, _outB);
        merged.Dispose();
        return output;
    }

    private Tensor Heads(IBackend backend, Tensor tokenMajor, int batch, int tokens)
    {
        Tensor headMajor = new(new TensorShape(batch, _heads, tokens, _headDim), DType.F32);
        backend.Permute0213(headMajor, tokenMajor, tokens, _heads, _headDim);
        tokenMajor.Dispose();
        return headMajor;
    }
}
