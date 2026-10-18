using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>MOSS-Image joint block (<c>_Block</c>) over <c>[image ; text]</c>, modulated by the one
/// <see cref="MossImageModulation"/> every block shares. No biases.
/// <code>
/// y = rms(x)·w1·(1 + scale_a) + shift_a
/// q, k = rope(rms_head(q(y))), rope(rms_head(k(y)))
/// x = x + gate_a · o(attention(q, k, v(y)))
/// y = rms(x)·w2·(1 + scale_f) + shift_f
/// x = x + gate_f · ffn_down(situ(ffn_up_gate(y)))      // ffn_up_gate rows are [hidden | gate]
/// </code>
/// The residual stream stays F32; the branch between the two norms and the two gated adds runs in the caller's
/// activation dtype (F16 on backends with F16 activations), which SiTU's soft caps keep in range.</summary>
public sealed class MossImageBlock(int hidden, int numHeads, int ffnHidden, float eps)
{
    private readonly int _hidden = hidden;
    private readonly int _numHeads = numHeads;
    private readonly int _headDim = hidden / numHeads;
    private readonly int _ffnHidden = ffnHidden;
    private readonly float _eps = eps;
    private readonly QkNorm _normQ = new(hidden / numHeads, eps);
    private readonly QkNorm _normK = new(hidden / numHeads, eps);
    private Tensor? _norm1, _norm2;
    private Tensor? _q, _k, _v, _o, _ffnUpGate, _ffnDown;

    /// <summary>Binds this block's weights under <paramref name="prefix"/> (<c>layers.N</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _norm1 = TensorCasts.LoadF32(weights, $"{prefix}.norm1.weight");
        _norm2 = TensorCasts.LoadF32(weights, $"{prefix}.norm2.weight");
        _normQ.LoadWeights(weights[$"{prefix}.q_norm.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.k_norm.weight"]);
        if (weights.TryGetValue($"{prefix}.proj.q.weight", out Tensor? q))
        {
            _q = q;
            _k = weights[$"{prefix}.proj.k.weight"];
            _v = weights[$"{prefix}.proj.v.weight"];
        }
        else
        {
            string qkvKey = $"{prefix}.proj.qkv.weight";
            Tensor qkv = weights[qkvKey];
            _q = HiddenRows(qkv, 0, _hidden, qkvKey);
            _k = HiddenRows(qkv, 1, _hidden, qkvKey);
            _v = HiddenRows(qkv, 2, _hidden, qkvKey);
        }
        _o = weights[$"{prefix}.proj.o.weight"];
        _ffnUpGate = weights[$"{prefix}.proj.ffn_up_gate.weight"];
        _ffnDown = weights[$"{prefix}.proj.ffn_down.weight"];
    }

    /// <summary>The <paramref name="index"/>th run of <paramref name="hidden"/> rows of a fused projection, as a view that
    /// carries its own slice of the per-row quantization companion.</summary>
    private static Tensor HiddenRows(Tensor fused, int index, int hidden, string key)
    {
        Tensor rows = fused.SliceRows(index * hidden, hidden);
        rows.QuantInfo = fused.QuantInfo?.SliceRows(index * hidden, hidden, key);
        return rows;
    }

    /// <summary>Every weight this block reads.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_norm1, _norm2, _q, _k, _v, _o, _ffnUpGate, _ffnDown];
        foreach (Tensor? w in all)
        {
            if (w is not null) yield return w;
        }
        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
    }

    /// <summary>Runs the block on the F32 residual stream <paramref name="x"/> <c>[1, S, hidden]</c> and returns a new
    /// F32 tensor; <paramref name="cos"/>/<paramref name="sin"/> are the joint sequence's <c>[1, S, headDim]</c> tables
    /// and <paramref name="act"/> the branch dtype.</summary>
    public Tensor Forward(IBackend backend, Tensor x, MossImageModulation modulation, Tensor cos, Tensor sin, DType act)
    {
        int seq = (int)x.Shape[1];
        TensorShape flat = new(1, seq, _hidden);
        TensorShape heads = new(1, seq, _numHeads, _headDim);

        Tensor y = new(flat, act);
        backend.RmsNormModulate(y, x, _norm1!, modulation.ScaleAttn, modulation.ShiftAttn, _eps);
        Tensor q = new(heads, act);
        backend.Linear(q, y, _q!, null);
        Tensor k = new(heads, act);
        backend.Linear(k, y, _k!, null);
        Tensor v = new(heads, act);
        backend.Linear(v, y, _v!, null);
        y.Dispose();
        Tensor qNormed = new(heads, act);
        backend.RmsNorm(qNormed, q, _normQ.Weight, _normQ.Eps);
        q.Dispose();
        Tensor kNormed = new(heads, act);
        backend.RmsNorm(kNormed, k, _normK.Weight, _normK.Eps);
        k.Dispose();
        backend.ApplyRope(qNormed, kNormed, cos, sin);
        Tensor attn = MossImageAttention.Run(backend, qNormed, kNormed, v, seq, _numHeads, _headDim);
        Tensor projected = new(flat, DType.F32);
        backend.Linear(projected, attn, _o!, null);
        attn.Dispose();
        Tensor x1 = new(flat, DType.F32);
        backend.GatedResidualLastDim(x1, x, projected, modulation.GateAttn);
        projected.Dispose();

        Tensor y2 = new(flat, act);
        backend.RmsNormModulate(y2, x1, _norm2!, modulation.ScaleFfn, modulation.ShiftFfn, _eps);
        Tensor pair = new(new TensorShape(1, seq, 2 * _ffnHidden), act);
        backend.Linear(pair, y2, _ffnUpGate!, null);
        y2.Dispose();
        Tensor activated = new(new TensorShape(1, seq, _ffnHidden), act);
        backend.SituGlu(activated, pair);
        pair.Dispose();
        Tensor down = new(flat, DType.F32);
        backend.Linear(down, activated, _ffnDown!, null);
        activated.Dispose();
        Tensor result = new(flat, DType.F32);
        backend.GatedResidualLastDim(result, x1, down, modulation.GateFfn);
        x1.Dispose();
        down.Dispose();
        return result;
    }
}
