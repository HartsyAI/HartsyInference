using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.ThreeD.Models.Pixal3D;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Pixal3D stage 4: samples the texture latent over the fine shape latent's voxels, with the image reaching the flow as global tokens plus
/// per-voxel projected features, and denormalizes it.</summary>
public interface IPixal3DTextureFlowStage
{
    /// <summary>The denormalized texture latent <c>[1, N, 32]</c> over the voxels of <paramref name="shape"/>. The caller owns the result
    /// and keeps ownership of the shape latent (denormalized, as the shape decoder takes it) and both conditionings.</summary>
    SparseTensor Sample(
        IBackend backend,
        SparseTensor shape,
        Pixal3DCondition cond,
        Pixal3DCondition negCond,
        int seed,
        Trellis2SamplerParams parameters,
        Action<int>? onStep);
}
