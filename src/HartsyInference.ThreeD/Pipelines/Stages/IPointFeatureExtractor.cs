using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Per-voxel point features from a dual grid, on a coarser grid (Utonia: the encoder's last stage).</summary>
public interface IPointFeatureExtractor
{
    /// <summary>Extracts features for single-sample voxels <paramref name="coords"/> (rows of <c>(batch, x, y, z)</c>, batch 0)
    /// on a grid of <paramref name="resolution"/> voxels per axis. The caller owns the returned features.</summary>
    SparseTensor Extract(IBackend backend, int[] coords, int resolution);
}
