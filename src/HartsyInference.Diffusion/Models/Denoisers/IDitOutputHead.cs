using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>What a DiT does with its final block tokens in place of its own output projection: a model that predicts pixels
/// rather than latent patches replaces the norm, modulation and linear head with a head of its own. The transformer runs its
/// embedding and blocks as usual and hands the result here.</summary>
public interface IDitOutputHead
{
    /// <summary>Maps the last block's tokens <c>[S, dim]</c> (token order <c>t, h, w</c> over <paramref name="grid"/>) and the
    /// timestep embedding <c>[1, dim]</c> to the model's output tensor. Neither input is disposed; the caller owns the result.</summary>
    Tensor Forward(IBackend backend, Tensor hidden, Tensor timestepEmbedding, (int T, int H, int W) grid);

    /// <summary>The weight tensors the head reads, for preloading and release alongside the transformer's own.</summary>
    IEnumerable<Tensor> EnumerateWeights();
}
