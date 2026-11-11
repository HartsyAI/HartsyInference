using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Stage 2: samples the shape latent over a set of voxels and denormalizes it.</summary>
public interface ITrellis2ShapeFlowStage
{
    /// <summary>The denormalized latent <c>[1, N, 32]</c> over <paramref name="coords"/> (<c>(batch, x, y, z)</c> rows on a
    /// <paramref name="gridResolution"/> grid). The caller owns the result and keeps ownership of the conditionings.</summary>
    SparseTensor Sample(
        IBackend backend,
        Trellis2ShapeFlowKind kind,
        int[] coords,
        int gridResolution,
        Tensor cond,
        Tensor negCond,
        int seed,
        Trellis2SamplerParams parameters,
        Action<int>? onStep);
}
