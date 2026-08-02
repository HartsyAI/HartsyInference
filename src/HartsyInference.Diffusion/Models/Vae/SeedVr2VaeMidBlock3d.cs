using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>SeedVR2 VAE mid block: <c>resnets.0 → attentions.0 → resnets.1</c>. Attention is PER-FRAME
/// spatial self-attention — the reference rearranges <c>b c f h w → (b f) c h w</c> and runs a plain 2D
/// attention per frame (single head, dim_head = C, residual inside), so <see cref="VaeAttention.Forward"/>
/// (the 4-D path) is used on the frame-major layout, never the frame-causal 3-D variant.</summary>
public sealed class SeedVr2VaeMidBlock3d
{
    private readonly SeedVr2VaeResnetBlock3d _resnet0;
    private readonly SeedVr2VaeResnetBlock3d _resnet1;
    private readonly VaeAttention _attention;

    /// <summary>Creates the block for <paramref name="channels"/>-wide activations.</summary>
    public SeedVr2VaeMidBlock3d(int channels, int groups, float eps, DType? actDtype = null)
    {
        _resnet0 = new SeedVr2VaeResnetBlock3d(groups, eps, actDtype);
        _resnet1 = new SeedVr2VaeResnetBlock3d(groups, eps, actDtype);
        _attention = new VaeAttention(channels, groups, eps);
    }

    /// <summary>Binds <c>{prefix}.resnets.0/attentions.0/resnets.1</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _resnet0.LoadWeights(weights, $"{prefix}.resnets.0");
        _resnet1.LoadWeights(weights, $"{prefix}.resnets.1");
        _attention.LoadWeights(weights, $"{prefix}.attentions.0");
    }

    /// <summary>Forward on <c>[B,C,T,H,W]</c> F32.</summary>
    public Tensor Forward(IBackend backend, Tensor x)
    {
        Tensor r0 = _resnet0.Forward(backend, x);
        int b = (int)r0.Shape[0], c = (int)r0.Shape[1], t = (int)r0.Shape[2], hh = (int)r0.Shape[3], ww = (int)r0.Shape[4];
        Tensor frames = Vae3dLayout.ToFrames(backend, r0);
        DType actDtype = r0.DType;
        r0.Dispose();
        // Attention runs F32 regardless of the activation dtype: the SDPA F32 route is the tiled/flash one,
        // and the mid-block tensors are latent-resolution (small) so the upcast is cheap.
        frames = DtypeCastHelper.EnsureF32(backend, frames);
        Tensor attended = _attention.Forward(backend, frames);
        frames.Dispose();
        attended = DtypeCastHelper.EnsureDtype(backend, attended, actDtype);
        Tensor h = Vae3dLayout.FromFrames(backend, attended, b, c, t, hh, ww);
        attended.Dispose();
        Tensor output = _resnet1.Forward(backend, h);
        h.Dispose();
        return output;
    }

    /// <summary>Weights for backend preload/free.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _resnet0.EnumerateWeights()) yield return t;
        foreach (Tensor t in _resnet1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _attention.EnumerateWeights()) yield return t;
    }
}
