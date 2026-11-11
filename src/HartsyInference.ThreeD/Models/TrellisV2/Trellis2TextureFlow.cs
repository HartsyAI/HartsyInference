using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>
/// TRELLIS.2 texture flow (<c>slat_flow_imgshape2tex</c>): the structured-latent flow of the shape path with a wider input. Each call
/// concatenates the normalized shape latent of the same voxels to the noisy texture latent, so the 64 input channels are
/// <c>[texture sample (32), shape latent (32)]</c> (upstream <c>sparse_cat([x, concat_cond], dim=-1)</c>), and the 32-channel
/// velocity is the texture latent's. Borrows <see cref="Trellis2FlowWeights"/>, which must outlive it.
/// </summary>
public sealed class Trellis2TextureFlow : IDisposable
{
    private readonly Trellis2SlatFlow _flow;

    /// <exception cref="ArgumentException">The weights are not a sparse flow whose input is wider than its output.</exception>
    public Trellis2TextureFlow(Trellis2FlowWeights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        Trellis2FlowConfig config = weights.Config;
        if (config.Dense || config.InChannels <= config.OutChannels)
        {
            throw new ArgumentException(
                $"A texture flow is a sparse flow with extra input channels; got in {config.InChannels}, out {config.OutChannels}.", nameof(weights));
        }

        _flow = new Trellis2SlatFlow(weights);
    }

    public Trellis2FlowConfig Config => _flow.Config;

    /// <summary>Channels of the latent being sampled (the flow's output width).</summary>
    public int LatentChannels => Config.OutChannels;

    /// <summary>Channels of the shape latent concatenated to every input (<c>in_channels - out_channels</c>).</summary>
    public int ConcatChannels => Config.InChannels - Config.OutChannels;

    /// <summary>The tensors the flow reads, for <c>IBackend.PreloadWeights</c> and <c>FreeWeights</c>.</summary>
    public IEnumerable<Tensor> EnumerateWeights() => _flow.EnumerateWeights();

    /// <summary>Velocity of the texture latent <paramref name="x"/> (<c>[1, N, LatentChannels]</c>) at model timestep
    /// <paramref name="tModel"/>, given the normalized shape latent <paramref name="shape"/> (<c>[1, N, ConcatChannels]</c>) of the same
    /// voxels and the image tokens <paramref name="cond"/>. <paramref name="coords"/> keeps its identity across the steps of one sample.</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor shape, int[] coords, float tModel, Tensor cond)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(shape);
        int n = x.Shape.Rank == 3 ? (int)x.Shape[1] : -1;
        if (x.Shape.Rank != 3 || x.Shape[0] != 1 || x.Shape[2] != LatentChannels)
            throw new ArgumentException($"The texture sample must be [1, N, {LatentChannels}]; got {x.Shape}.", nameof(x));
        if (shape.Shape.Rank != 3 || shape.Shape[0] != 1 || shape.Shape[1] != n || shape.Shape[2] != ConcatChannels)
            throw new ArgumentException($"The shape latent must be [1, {n}, {ConcatChannels}]; got {shape.Shape}.", nameof(shape));

        using Tensor input = CfgHelper.ConcatLastDim(x, shape);
        return _flow.Forward(backend, input, coords, tModel, cond);
    }

    public void Dispose() => _flow.Dispose();
}
