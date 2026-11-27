using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;

namespace HartsyInference.Engine.Services;

/// <summary>Which files a 3D model needs and which are on disk, without loading anything. Fetching goes through <see cref="IModelPrefetchService"/>.</summary>
public interface IMeshAssetService
{
    /// <summary>The file inventory of <paramref name="spec"/>'s model, located where the loaders look.</summary>
    /// <exception cref="NotSupportedException">The spec names a model with no wired 3D pipeline.</exception>
    MeshAssetStatus GetStatus(ModelSpec spec);
}
