using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.ThreeD.Models.CoLa3D;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Splits one scene latent into per-object latents.</summary>
public interface IObjectDecomposer
{
    /// <summary>Returns one single-sample latent per object, each on the scene latent's grid. The caller owns the results.
    /// <paramref name="prompts"/> is null for free queries and carries the prompt inputs for a prompt checkpoint.</summary>
    IReadOnlyList<SparseTensor> Decompose(IBackend backend, SparseTensor latent, SparseTensor features, CoLa3DQueryPrompts? prompts = null);
}
