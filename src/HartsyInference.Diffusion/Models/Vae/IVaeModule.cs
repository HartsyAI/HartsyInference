using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>A VAE layer that owns its weight tensors. It enumerates them for device placement around a decode, and releases them on Dispose.
/// Every weight a layer holds is an owned copy, so disposing the layer never frees a tensor the checkpoint loader still holds.</summary>
public interface IVaeModule : IDisposable
{
    /// <summary>The weight tensors the layer reads now. Live: a conv's weight form can change after its first calls, so enumerate again rather than
    /// keeping the list.</summary>
    IEnumerable<Tensor> EnumerateWeights();
}
