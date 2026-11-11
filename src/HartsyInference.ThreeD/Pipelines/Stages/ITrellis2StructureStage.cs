using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Stage 1: samples the dense structure latent, decodes its occupancy and pools it to the voxels the shape flow runs on.</summary>
public interface ITrellis2StructureStage
{
    /// <summary>Voxels <c>(0, x, y, z)</c> on a <paramref name="outputResolution"/> grid. The caller keeps ownership of both conditionings.</summary>
    /// <param name="onStep">Called after each sampler step; it may throw to cancel.</param>
    int[] Sample(
        IBackend backend, Tensor cond, Tensor negCond, int seed, Trellis2SamplerParams parameters, int outputResolution, Action<int>? onStep);
}
