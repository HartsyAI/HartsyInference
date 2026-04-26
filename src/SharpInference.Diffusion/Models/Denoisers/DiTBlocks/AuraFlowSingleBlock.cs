using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>AuraFlow single-stream block. Processes image tokens only (after joint attention blocks). Uses AdaLN modulation with self-attention and SwiGLU FFN.</summary>
public sealed class AuraFlowSingleBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _mlpDim;

    // AdaLN modulation → 6 params (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
    private readonly AdaLNModulation _modulation;

    // Self-attention projections
    private Tensor? _toQWeight, _toQBias;
    private Tensor? _toKWeight, _toKBias;
    private Tensor? _toVWeight, _toVBias;
    private Tensor? _toOutWeight, _toOutBias;

    // QK-norm
    private readonly QkNorm? _normQ;
    private readonly QkNorm? _normK;

    // FFN
    private readonly SwiGluFfn _ffn;

    /// <summary>Creates an AuraFlow single-stream block.</summary>
    public AuraFlowSingleBlock(int hiddenSize, int numHeads, int mlpDim, bool useQkNorm, float qkNormEps = 1e-6f)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = hiddenSize / numHeads;
        _mlpDim = mlpDim;

        _modulation = new AdaLNModulation(hiddenSize, 6);

        if (useQkNorm)
        {
            _normQ = new QkNorm(_headDim, qkNormEps);
            _normK = new QkNorm(_headDim, qkNormEps);
        }

        _ffn = new SwiGluFfn(hiddenSize, mlpDim);
    }

    /// <summary>Loads weights from named tensors.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _modulation.LoadWeights(
            weights[$"{prefix}.mod.linear.weight"],
            weights[$"{prefix}.mod.linear.bias"]);

        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toQBias = weights.ContainsKey($"{prefix}.attn.to_q.bias") ? weights[$"{prefix}.attn.to_q.bias"] : null;
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toKBias = weights.ContainsKey($"{prefix}.attn.to_k.bias") ? weights[$"{prefix}.attn.to_k.bias"] : null;
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
        _toVBias = weights.ContainsKey($"{prefix}.attn.to_v.bias") ? weights[$"{prefix}.attn.to_v.bias"] : null;
        _toOutWeight = weights[$"{prefix}.attn.to_out.0.weight"];
        _toOutBias = weights[$"{prefix}.attn.to_out.0.bias"];

        if (_normQ is not null) _normQ.LoadWeights(weights[$"{prefix}.attn.norm_q.weight"]);
        if (_normK is not null) _normK.LoadWeights(weights[$"{prefix}.attn.norm_k.weight"]);

        _ffn.LoadSwiGluWeights(
            weights[$"{prefix}.mlp.w1.weight"], weights[$"{prefix}.mlp.w1.bias"],
            weights[$"{prefix}.mlp.w3.weight"], weights[$"{prefix}.mlp.w3.bias"],
            weights[$"{prefix}.mlp.w2.weight"], weights[$"{prefix}.mlp.w2.bias"]);
    }

    /// <summary>Forward pass: self-attention + FFN with AdaLN modulation and residual connections.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor temb)
    {
        // TODO: Implement full forward pass with:
        // 1. AdaLN modulation → shift/scale/gate for attention and FFN
        // 2. LayerNorm + modulate → Q/K/V projections
        // 3. Optional QK-norm
        // 4. Self-attention (SDPA)
        // 5. Gate + residual
        // 6. LayerNorm + modulate → FFN → gate + residual
        throw new NotImplementedException("AuraFlowSingleBlock.Forward requires backend attention and linear ops");
    }

    /// <summary>Yields all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _modulation.EnumerateWeights()) yield return t;

        if (_toQWeight is not null) yield return _toQWeight;
        if (_toQBias is not null) yield return _toQBias;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toKBias is not null) yield return _toKBias;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_toVBias is not null) yield return _toVBias;
        if (_toOutWeight is not null) yield return _toOutWeight;
        if (_toOutBias is not null) yield return _toOutBias;

        if (_normQ is not null) foreach (Tensor t in _normQ.EnumerateWeights()) yield return t;
        if (_normK is not null) foreach (Tensor t in _normK.EnumerateWeights()) yield return t;

        foreach (Tensor t in _ffn.EnumerateWeights()) yield return t;
    }
}
