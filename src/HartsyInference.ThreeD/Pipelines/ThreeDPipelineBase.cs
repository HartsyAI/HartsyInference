using HartsyInference.Core.Backends;
using HartsyInference.Diffusion.Pipelines;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>Shared base for 3D-generation pipelines. Reuses <see cref="DiffusionPipelineBase"/> for the
/// compute <see cref="IBackend"/> handle, idempotent disposal, and the component-ownership convention
/// (pipelines don't own the encoders/transformers/VAEs passed in). A thin seam today; a home for shared
/// 3D pipeline helpers as more models land.</summary>
public abstract class ThreeDPipelineBase : DiffusionPipelineBase
{
    /// <summary>Initializes the base with the compute backend.</summary>
    protected ThreeDPipelineBase(IBackend backend) : base(backend) { }
}
