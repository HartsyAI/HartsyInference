using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Encodes a dual grid into a sparse latent on its coarser grid.</summary>
public interface ISparseShapeEncoder
{
    /// <summary>Encodes <paramref name="voxels"/> on a grid of <paramref name="resolution"/> voxels per axis. The caller owns the returned latent.</summary>
    SparseTensor Encode(IBackend backend, DualGridVoxels voxels, int resolution);
}
