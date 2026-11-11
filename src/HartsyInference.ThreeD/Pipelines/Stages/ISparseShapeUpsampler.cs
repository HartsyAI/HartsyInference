using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Predicts where a coarse shape latent's surface lives on a finer grid, without decoding it.</summary>
public interface ISparseShapeUpsampler
{
    /// <summary>The voxels <c>(batch, x, y, z)</c> the decoder creates after <paramref name="times"/> subdivision levels. The caller keeps ownership of the latent.</summary>
    int[] Upsample(IBackend backend, SparseTensor latent, int times);
}
