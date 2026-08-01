using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>SeedVR2 VAE downsampler (reference <c>Downsample3D</c>): asymmetric zero pad (0,1,0,1) —
/// right/bottom only, the diffusers <c>Downsample2D(padding=0)</c> convention — then a stride-2 conv with
/// NO symmetric spatial padding. Temporal variant (kernel (3,3,3), stride-2 time) pads the head with two
/// frame-0 copies; spatial-only variant is kernel (1,3,3) with zero temporal pad.</summary>
public sealed class SeedVr2VaeDownsample3d
{
    private readonly bool _temporal;
    private CausalConv3d _conv = null!;

    /// <summary>Creates the downsampler; <paramref name="temporal"/> per <see cref="SeedVr2VaeConfig.StageDownsamplesTime"/>.</summary>
    public SeedVr2VaeDownsample3d(bool temporal)
    {
        _temporal = temporal;
    }

    /// <summary>Binds <c>{prefix}.conv</c>; pads derive from the kernel shape (kt=1 → padT 0).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
        => _conv = SeedVr2VaeOps.Conv(weights, $"{prefix}.conv",
            strideT: _temporal ? 2 : 1, strideH: 2, strideW: 2, padSpatial: false);

    /// <summary>Forward on <c>[B,C,T,H,W]</c> F32.</summary>
    public Tensor Forward(IBackend backend, Tensor x)
    {
        Tensor padded = SeedVr2VaeOps.PadBottomRight(backend, x);
        Tensor output = _conv.Forward(backend, padded);
        padded.Dispose();
        return output;
    }

    /// <summary>Weights for backend preload/free.</summary>
    public IEnumerable<Tensor> EnumerateWeights() => _conv.EnumerateWeights();
}
