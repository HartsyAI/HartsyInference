using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>AuraFlow joint MMDiT block. Dual-stream attention between image and text tokens with AdaLN modulation. Similar to SD3 JointBlock but uses a different modulation scheme (AuraFlow uses timestep-conditioned LayerNorm without gating on some variants).</summary>
public sealed class AuraFlowJointBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _mlpDim;

    // Image stream: AdaLN modulation → 6 params (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
    private readonly AdaLNModulation _imgModulation;

    // Text stream: AdaLN modulation → 6 params
    private readonly AdaLNModulation _txtModulation;

    // Image attention projections
    private Tensor? _imgQWeight, _imgQBias;
    private Tensor? _imgKWeight, _imgKBias;
    private Tensor? _imgVWeight, _imgVBias;
    private Tensor? _imgOutWeight, _imgOutBias;

    // Text attention projections
    private Tensor? _txtQWeight, _txtQBias;
    private Tensor? _txtKWeight, _txtKBias;
    private Tensor? _txtVWeight, _txtVBias;
    private Tensor? _txtOutWeight, _txtOutBias;

    // QK-norm
    private readonly QkNorm? _normQ;
    private readonly QkNorm? _normK;

    // Image MLP
    private readonly SwiGluFfn _imgFfn;

    // Text MLP
    private readonly SwiGluFfn _txtFfn;

    /// <summary>Creates an AuraFlow joint block.</summary>
    public AuraFlowJointBlock(int hiddenSize, int numHeads, int mlpDim, bool useQkNorm, float qkNormEps = 1e-6f)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = hiddenSize / numHeads;
        _mlpDim = mlpDim;

        _imgModulation = new AdaLNModulation(hiddenSize, 6);
        _txtModulation = new AdaLNModulation(hiddenSize, 6);

        if (useQkNorm)
        {
            _normQ = new QkNorm(_headDim, qkNormEps);
            _normK = new QkNorm(_headDim, qkNormEps);
        }

        _imgFfn = new SwiGluFfn(hiddenSize, mlpDim);
        _txtFfn = new SwiGluFfn(hiddenSize, mlpDim);
    }

    /// <summary>Loads weights from named tensors.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _imgModulation.LoadWeights(
            weights[$"{prefix}.img_mod.linear.weight"],
            weights[$"{prefix}.img_mod.linear.bias"]);
        _txtModulation.LoadWeights(
            weights[$"{prefix}.txt_mod.linear.weight"],
            weights[$"{prefix}.txt_mod.linear.bias"]);

        _imgQWeight = weights[$"{prefix}.img_attn.to_q.weight"];
        _imgQBias = weights.ContainsKey($"{prefix}.img_attn.to_q.bias") ? weights[$"{prefix}.img_attn.to_q.bias"] : null;
        _imgKWeight = weights[$"{prefix}.img_attn.to_k.weight"];
        _imgKBias = weights.ContainsKey($"{prefix}.img_attn.to_k.bias") ? weights[$"{prefix}.img_attn.to_k.bias"] : null;
        _imgVWeight = weights[$"{prefix}.img_attn.to_v.weight"];
        _imgVBias = weights.ContainsKey($"{prefix}.img_attn.to_v.bias") ? weights[$"{prefix}.img_attn.to_v.bias"] : null;
        _imgOutWeight = weights[$"{prefix}.img_attn.to_out.0.weight"];
        _imgOutBias = weights[$"{prefix}.img_attn.to_out.0.bias"];

        _txtQWeight = weights[$"{prefix}.txt_attn.to_q.weight"];
        _txtQBias = weights.ContainsKey($"{prefix}.txt_attn.to_q.bias") ? weights[$"{prefix}.txt_attn.to_q.bias"] : null;
        _txtKWeight = weights[$"{prefix}.txt_attn.to_k.weight"];
        _txtKBias = weights.ContainsKey($"{prefix}.txt_attn.to_k.bias") ? weights[$"{prefix}.txt_attn.to_k.bias"] : null;
        _txtVWeight = weights[$"{prefix}.txt_attn.to_v.weight"];
        _txtVBias = weights.ContainsKey($"{prefix}.txt_attn.to_v.bias") ? weights[$"{prefix}.txt_attn.to_v.bias"] : null;
        _txtOutWeight = weights[$"{prefix}.txt_attn.to_out.0.weight"];
        _txtOutBias = weights[$"{prefix}.txt_attn.to_out.0.bias"];

        if (_normQ is not null) _normQ.LoadWeights(weights[$"{prefix}.img_attn.norm_q.weight"]);
        if (_normK is not null) _normK.LoadWeights(weights[$"{prefix}.img_attn.norm_k.weight"]);

        _imgFfn.LoadSwiGluWeights(
            weights[$"{prefix}.img_mlp.w1.weight"], weights[$"{prefix}.img_mlp.w1.bias"],
            weights[$"{prefix}.img_mlp.w3.weight"], weights[$"{prefix}.img_mlp.w3.bias"],
            weights[$"{prefix}.img_mlp.w2.weight"], weights[$"{prefix}.img_mlp.w2.bias"]);
        _txtFfn.LoadSwiGluWeights(
            weights[$"{prefix}.txt_mlp.w1.weight"], weights[$"{prefix}.txt_mlp.w1.bias"],
            weights[$"{prefix}.txt_mlp.w3.weight"], weights[$"{prefix}.txt_mlp.w3.bias"],
            weights[$"{prefix}.txt_mlp.w2.weight"], weights[$"{prefix}.txt_mlp.w2.bias"]);
    }

    /// <summary>Forward pass through joint attention + FFN. Returns updated (image, text) tensors.</summary>
    public (Tensor image, Tensor text) Forward(IBackend backend, Tensor image, Tensor text, Tensor temb)
    {
        // TODO: Implement full forward pass with:
        // 1. AdaLN modulation for both streams
        // 2. Q/K/V projections for both streams
        // 3. Optional QK-norm
        // 4. Joint attention (concat Q/K/V across streams, attend, split back)
        // 5. Gate + residual for attention output
        // 6. AdaLN + FFN + gate + residual for both streams
        // This follows the same pattern as JointBlock.cs and FluxDoubleStreamBlock.cs
        throw new NotImplementedException("AuraFlowJointBlock.Forward requires backend attention and linear ops");
    }

    /// <summary>Yields all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _imgModulation.EnumerateWeights()) yield return t;
        foreach (Tensor t in _txtModulation.EnumerateWeights()) yield return t;

        if (_imgQWeight is not null) yield return _imgQWeight;
        if (_imgQBias is not null) yield return _imgQBias;
        if (_imgKWeight is not null) yield return _imgKWeight;
        if (_imgKBias is not null) yield return _imgKBias;
        if (_imgVWeight is not null) yield return _imgVWeight;
        if (_imgVBias is not null) yield return _imgVBias;
        if (_imgOutWeight is not null) yield return _imgOutWeight;
        if (_imgOutBias is not null) yield return _imgOutBias;

        if (_txtQWeight is not null) yield return _txtQWeight;
        if (_txtQBias is not null) yield return _txtQBias;
        if (_txtKWeight is not null) yield return _txtKWeight;
        if (_txtKBias is not null) yield return _txtKBias;
        if (_txtVWeight is not null) yield return _txtVWeight;
        if (_txtVBias is not null) yield return _txtVBias;
        if (_txtOutWeight is not null) yield return _txtOutWeight;
        if (_txtOutBias is not null) yield return _txtOutBias;

        if (_normQ is not null) foreach (Tensor t in _normQ.EnumerateWeights()) yield return t;
        if (_normK is not null) foreach (Tensor t in _normK.EnumerateWeights()) yield return t;

        foreach (Tensor t in _imgFfn.EnumerateWeights()) yield return t;
        foreach (Tensor t in _txtFfn.EnumerateWeights()) yield return t;
    }
}
