using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Stage 3: samples the texture latent over the shape latent's voxels and denormalizes it.</summary>
public interface ITrellis2TextureFlowStage
{
    /// <summary>Whether the flow for <paramref name="kind"/> is installed, so a request fails before its first stage runs.</summary>
    bool Supports(Trellis2ShapeFlowKind kind);

    /// <summary>The denormalized texture latent <c>[1, N, 32]</c> over the voxels of <paramref name="shape"/>. The caller owns the result
    /// and keeps ownership of the shape latent (denormalized, as the shape decoder takes it) and both conditionings.</summary>
    SparseTensor Sample(
        IBackend backend,
        Trellis2ShapeFlowKind kind,
        SparseTensor shape,
        Tensor cond,
        Tensor negCond,
        int seed,
        Trellis2SamplerParams parameters,
        Action<int>? onStep);
}
