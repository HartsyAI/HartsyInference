using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>A Hunyuan3D shape denoiser: predicts the flow velocity of a latent set from DINOv2 tokens.</summary>
internal interface IHunyuan3DDenoiser
{
    /// <summary>Velocity for <paramref name="latent"/> <c>[1,N,C]</c> at <paramref name="timestep"/> in [0,1].</summary>
    Tensor Forward(IBackend backend, Tensor latent, Tensor cond, float timestep, float? guidance = null);

    /// <summary>Every weight tensor, for preloading to and freeing from the device.</summary>
    IEnumerable<Tensor> EnumerateWeights();
}
