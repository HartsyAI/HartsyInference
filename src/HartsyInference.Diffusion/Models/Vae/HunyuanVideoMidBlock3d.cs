using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>HunyuanVideo VAE mid block (<c>HunyuanVideoMidBlock3D</c>, num_layers=1): resnet →
/// frame-causal single-head attention over all T·H·W tokens (reuses <see cref="VaeAttention"/>) → resnet.
/// Shared by the encoder and decoder (identical structure and key layout under <c>{prefix}.mid_block</c>).</summary>
public sealed class HunyuanVideoMidBlock3d
{
    private readonly HunyuanVideoResnetBlock3d _resnet0;
    private readonly HunyuanVideoResnetBlock3d _resnet1;
    private readonly VaeAttention? _attention;

    public HunyuanVideoMidBlock3d(int channels, bool addAttention, int normGroups = 32, float normEps = 1e-6f)
    {
        _resnet0 = new HunyuanVideoResnetBlock3d(channels, channels, normGroups, normEps);
        _resnet1 = new HunyuanVideoResnetBlock3d(channels, channels, normGroups, normEps);
        _attention = addAttention ? new VaeAttention(channels, normGroups, normEps) : null;
    }

    /// <summary>Loads <c>{prefix}.resnets.{0,1}.*</c> and <c>{prefix}.attentions.0.*</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _resnet0.LoadWeights(weights, $"{prefix}.resnets.0");
        _resnet1.LoadWeights(weights, $"{prefix}.resnets.1");
        _attention?.LoadWeights(weights, $"{prefix}.attentions.0");
    }

    /// <summary>Enumerates all weights for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _resnet0.EnumerateWeights()) yield return t;
        foreach (Tensor t in _resnet1.EnumerateWeights()) yield return t;
        if (_attention is not null)
            foreach (Tensor t in _attention.EnumerateWeights()) yield return t;
    }

    /// <summary>Forward over <c>[B, C, T, H, W]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor x)
    {
        Tensor h = _resnet0.Forward(backend, x);
        if (_attention is not null)
        {
            Tensor attn = _attention.Forward3D(backend, h, frameCausal: true);
            h.Dispose();
            h = attn;
        }
        Tensor output = _resnet1.Forward(backend, h);
        h.Dispose();
        return output;
    }
}
