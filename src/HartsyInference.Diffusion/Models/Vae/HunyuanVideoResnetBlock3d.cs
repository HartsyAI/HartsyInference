using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>HunyuanVideo VAE resnet block (<c>HunyuanVideoResnetBlockCausal3D</c>):
/// GroupNorm → SiLU → CausalConv3d(k3) → GroupNorm → SiLU → CausalConv3d(k3), residual via 1×1×1
/// causal conv shortcut when channels change. GroupNorm statistics span the full clip (C/g, T, H, W);
/// all convs use HunyuanVideo replicate padding (temporal first-frame left pad + spatial edge pad).</summary>
public sealed class HunyuanVideoResnetBlock3d(int inChannels, int outChannels, int normGroups = 32, float normEps = 1e-6f)
{
    private readonly int _inChannels = inChannels;
    private readonly int _outChannels = outChannels;
    private readonly int _normGroups = normGroups;
    private readonly float _normEps = normEps;

    private Tensor? _norm1Weight, _norm1Bias;
    private Tensor? _norm2Weight, _norm2Bias;
    private CausalConv3d? _conv1, _conv2, _convShortcut;

    /// <summary>Loads <c>{prefix}.{norm1,conv1,norm2,conv2[,conv_shortcut]}</c> weights (conv key spelling fallback-gated, see <see cref="HunyuanVideoVaeKeys"/>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _norm1Weight = weights[$"{prefix}.norm1.weight"];
        _norm1Bias   = weights[$"{prefix}.norm1.bias"];
        _norm2Weight = weights[$"{prefix}.norm2.weight"];
        _norm2Bias   = weights[$"{prefix}.norm2.bias"];

        _conv1 = HunyuanVideoVaeKeys.Conv(weights, $"{prefix}.conv1", padT: 1, padH: 1, padW: 1);
        _conv2 = HunyuanVideoVaeKeys.Conv(weights, $"{prefix}.conv2", padT: 1, padH: 1, padW: 1);
        if (_inChannels != _outChannels)
            _convShortcut = HunyuanVideoVaeKeys.Conv(weights, $"{prefix}.conv_shortcut", padT: 0, padH: 0, padW: 0);
    }

    /// <summary>Enumerates all weights for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] norms = [_norm1Weight, _norm1Bias, _norm2Weight, _norm2Bias];
        for (int i = 0; i < norms.Length; i++)
            if (norms[i] is Tensor t) yield return t;
        if (_conv1 is not null) foreach (Tensor t in _conv1.EnumerateWeights()) yield return t;
        if (_conv2 is not null) foreach (Tensor t in _conv2.EnumerateWeights()) yield return t;
        if (_convShortcut is not null) foreach (Tensor t in _convShortcut.EnumerateWeights()) yield return t;
    }

    /// <summary>Forward over <c>[B, C_in, T, H, W]</c> → <c>[B, C_out, T, H, W]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor x)
    {
        Tensor h = HunyuanVideoVaeKeys.GroupNormSilu3d(backend, x, _norm1Weight!, _norm1Bias!, _normGroups, _normEps);
        Tensor c1 = _conv1!.Forward(backend, h);
        h.Dispose();

        Tensor h2 = HunyuanVideoVaeKeys.GroupNormSilu3d(backend, c1, _norm2Weight!, _norm2Bias!, _normGroups, _normEps);
        c1.Dispose();
        Tensor c2 = _conv2!.Forward(backend, h2);
        h2.Dispose();

        Tensor residual = _convShortcut is null ? x : _convShortcut.Forward(backend, x);
        Tensor output = new Tensor(c2.Shape, DType.F32);
        backend.Add(output, c2, residual);
        c2.Dispose();
        if (_convShortcut is not null) residual.Dispose();
        return output;
    }
}

/// <summary>Shared helpers for the HunyuanVideo VAE: conv key-spelling fallback and clip-wide GroupNorm+SiLU.</summary>
internal static class HunyuanVideoVaeKeys
{
    /// <summary>Builds a HunyuanVideo causal conv from <c>{baseKey}.conv.weight</c> (diffusers module wrapping) with a <c>{baseKey}.weight</c> fallback. Replicate padding everywhere per <c>HunyuanVideoCausalConv3d</c>.</summary>
    public static CausalConv3d Conv(IReadOnlyDictionary<string, Tensor> weights, string baseKey,
        int padT, int padH, int padW, int strideT = 1, int strideH = 1, int strideW = 1)
    {
        string weightKey = weights.ContainsKey($"{baseKey}.conv.weight") ? $"{baseKey}.conv.weight" : $"{baseKey}.weight";
        string biasKey = weightKey[..^"weight".Length] + "bias";
        Tensor? bias = weights.TryGetValue(biasKey, out Tensor? b) ? b : null;
        return new CausalConv3d(weights[weightKey], bias, strideT, strideH, strideW, padT, padH, padW,
            replicateFirstPad: true, causal: true, spatialReplicatePad: true);
    }

    /// <summary>GroupNorm with clip-wide statistics over a <c>[B, C, T, H, W]</c> tensor, followed by SiLU. Views the clip as <c>[B, C, T·H·W, 1]</c> so the existing 4-D GroupNorm op normalizes over (C/g, T, H, W).</summary>
    public static Tensor GroupNormSilu3d(IBackend backend, Tensor x, Tensor weight, Tensor bias, int groups, float eps)
    {
        // Operate on the native [B,C,T,H,W] shape — CudaBackend.GroupNorm derives spatial from dims 2+, so no
        // reshape is needed. The previous reshape to [B,C,T·H·W,1] made GroupNorm cache its output on the *view*
        // object while Silu read the parent → CUDA activation-cache miss → stale host zeros (the whole norm path
        // silently produced 0; resnets masked it via their residual, but norm_out has none → blank decode). Keeping
        // one object identity (normed) across GroupNorm + Silu fixes it. See cuda-activation-cache-reshape-identity.
        Tensor normed = new Tensor(x.Shape, DType.F32);
        backend.GroupNorm(normed, x, weight, bias, groups, eps);
        backend.Silu(normed, normed);
        return normed;
    }
}
