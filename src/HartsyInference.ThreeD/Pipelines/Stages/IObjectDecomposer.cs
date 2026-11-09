using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Splits one scene latent into per-object latents.</summary>
public interface IObjectDecomposer
{
    /// <summary>Returns one single-sample latent per object, each on the scene latent's grid. The caller owns the results.</summary>
    IReadOnlyList<SparseTensor> Decompose(IBackend backend, SparseTensor latent, SparseTensor features);
}
