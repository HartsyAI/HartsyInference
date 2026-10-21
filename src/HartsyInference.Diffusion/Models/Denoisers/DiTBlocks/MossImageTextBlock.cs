using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>MOSS-Image text-refiner block (<c>_TextBlock</c>): a plain pre-norm transformer layer over the text tokens
/// alone, with biases and no modulation, so its output depends on the prompt only:
/// <code>
/// q, k, v = qkv(rms(x)) ; q, k = per-head rms(q), rms(k) ; q, k = rope(q), rope(k)
/// x = x + proj(attention(q, k, v))
/// x = x + ffn_down(situ(ffn_up(rms(x)), ffn_gate(rms(x))))
/// </code>
/// Runs once per prompt in F32.</summary>
public sealed class MossImageTextBlock(int hidden, int numHeads, int ffnHidden, float eps)
{
    private readonly int _hidden = hidden;
    private readonly int _numHeads = numHeads;
    private readonly int _headDim = hidden / numHeads;
    private readonly int _ffnHidden = ffnHidden;
    private readonly float _eps = eps;
    private readonly QkNorm _normQ = new(hidden / numHeads, eps);
    private readonly QkNorm _normK = new(hidden / numHeads, eps);
    private Tensor? _norm1, _norm2;
    private Tensor? _qkv, _qkvBias, _proj, _projBias;
    private Tensor? _ffnGate, _ffnGateBias, _ffnUp, _ffnUpBias, _ffnDown, _ffnDownBias;
    private AttentionLayer? _attention;

    /// <summary>Binds this block's weights under <paramref name="prefix"/> (<c>text_embedder.blocks.N</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _norm1 = TensorCasts.LoadF32(weights, $"{prefix}.norm1.weight");
        _norm2 = TensorCasts.LoadF32(weights, $"{prefix}.norm2.weight");
        _normQ.LoadWeights(weights[$"{prefix}.q_norm.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.k_norm.weight"]);
        _qkv = weights[$"{prefix}.qkv.weight"];
        _qkvBias = TensorCasts.LoadF32(weights, $"{prefix}.qkv.bias");
        _proj = weights[$"{prefix}.proj.weight"];
        _projBias = TensorCasts.LoadF32(weights, $"{prefix}.proj.bias");
        _ffnGate = weights[$"{prefix}.ffn_gate.weight"];
        _ffnGateBias = TensorCasts.LoadF32(weights, $"{prefix}.ffn_gate.bias");
        _ffnUp = weights[$"{prefix}.ffn_up.weight"];
        _ffnUpBias = TensorCasts.LoadF32(weights, $"{prefix}.ffn_up.bias");
        _ffnDown = weights[$"{prefix}.ffn_down.weight"];
        _ffnDownBias = TensorCasts.LoadF32(weights, $"{prefix}.ffn_down.bias");
        _attention = new AttentionLayer(
            new AttentionLayerConfig
            {
                Heads = _numHeads, KvHeads = _numHeads, HeadDim = _headDim, Projection = AttentionProjection.FusedSplitNorm,
                QkNorm = AttentionQkNorm.PerHead, TokenMajorRoute = AttentionRoute.TokenMajor, ComputeDType = DType.F32, AllowF16 = true,
            },
            new AttentionLayerWeights
            {
                Qkv = _qkv, QkvBias = _qkvBias, Output = _proj, OutputBias = _projBias, QueryNorm = _normQ.Weight,
                KeyNorm = _normK.Weight, QueryNormEps = _eps, KeyNormEps = _eps,
            });
    }

    /// <summary>Every weight this block reads.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_norm1, _norm2, _qkv, _qkvBias, _proj, _projBias, _ffnGate, _ffnGateBias, _ffnUp, _ffnUpBias, _ffnDown, _ffnDownBias];
        foreach (Tensor? w in all)
        {
            if (w is not null) yield return w;
        }
        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
    }

    /// <summary>Runs the block on <paramref name="x"/> <c>[1, L, hidden]</c> (F32) and returns a new tensor;
    /// <paramref name="cos"/>/<paramref name="sin"/> are the text tokens' <c>[1, L, headDim]</c> RoPE tables.</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor cos, Tensor sin)
    {
        int seq = (int)x.Shape[1];
        TensorShape flat = new(1, seq, _hidden);

        Tensor n1 = new(flat, DType.F32);
        backend.RmsNorm(n1, x, _norm1!, _eps);
        QkRotation rope = new(QkRotationKind.RotateHalfPaired, cos, sin);
        Tensor projected = _attention!.Forward(backend, n1, n1,
            new AttentionCallContext { QueryRotation = rope, KeyRotation = rope, ReleaseInputs = true });
        Tensor x1 = new(flat, DType.F32);
        backend.Add(x1, x, projected);
        projected.Dispose();

        Tensor n2 = new(flat, DType.F32);
        backend.RmsNorm(n2, x1, _norm2!, _eps);
        TensorShape ffn = new(1, seq, _ffnHidden);
        Tensor up = new(ffn, DType.F32);
        backend.Linear(up, n2, _ffnUp!, _ffnUpBias);
        Tensor gate = new(ffn, DType.F32);
        backend.Linear(gate, n2, _ffnGate!, _ffnGateBias);
        n2.Dispose();
        Tensor pair = new(new TensorShape(1, seq, 2 * _ffnHidden), DType.F32);
        backend.Concat(pair, [up, gate], 2);
        up.Dispose();
        gate.Dispose();
        Tensor activated = new(ffn, DType.F32);
        backend.SituGlu(activated, pair);
        pair.Dispose();
        Tensor down = new(flat, DType.F32);
        backend.Linear(down, activated, _ffnDown!, _ffnDownBias);
        activated.Dispose();
        Tensor result = new(flat, DType.F32);
        backend.Add(result, x1, down);
        x1.Dispose();
        down.Dispose();
        return result;
    }
}
