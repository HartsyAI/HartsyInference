using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>SeedVR2 VAE upsampler (reference MAGViT-style <c>Upsample3D</c>): 1×1×1 <c>upscale_conv</c>
/// (C → C·ratio) → channel-to-space shuffle in <c>(x y z c)</c> order → drop the duplicated half of frame 0
/// when temporal (T→2T−1) → k3 causal conv. Ratio is ×8 (2,2,2) on temporal stages, ×4 (1,2,2) otherwise.</summary>
public sealed class SeedVr2VaeUpsampler3d
{
    private readonly bool _temporal;
    private readonly DType _actDtype;
    private CausalConv3d _upscale = null!;
    private CausalConv3d _conv = null!;

    /// <summary>Creates the upsampler; <paramref name="temporal"/> per <see cref="SeedVr2VaeConfig.StageUpsamplesTime"/>.</summary>
    public SeedVr2VaeUpsampler3d(bool temporal, DType? actDtype = null)
    {
        _temporal = temporal;
        _actDtype = actDtype ?? DType.F32;
    }

    /// <summary>Binds <c>{prefix}.upscale_conv</c> and <c>{prefix}.conv</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _upscale = SeedVr2VaeOps.Conv(weights, $"{prefix}.upscale_conv", computeDtype: _actDtype);
        _conv = SeedVr2VaeOps.Conv(weights, $"{prefix}.conv", computeDtype: _actDtype);
    }

    /// <summary>Forward on <c>[B,C,T,H,W]</c> F32.</summary>
    public Tensor Forward(IBackend backend, Tensor x)
    {
        Tensor expanded = _upscale.Forward(backend, x);
        Tensor shuffled = SeedVr2VaeOps.PixelShuffle(backend, expanded, spatialRatio: 2, temporalRatio: _temporal ? 2 : 1);
        expanded.Dispose();
        Tensor output = _conv.Forward(backend, shuffled);
        shuffled.Dispose();
        return output;
    }

    /// <summary>Weights for backend preload/free.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _upscale.EnumerateWeights()) yield return t;
        foreach (Tensor t in _conv.EnumerateWeights()) yield return t;
    }
}
