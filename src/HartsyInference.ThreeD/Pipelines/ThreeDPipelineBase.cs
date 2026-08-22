using HartsyInference.Core.Backends;
using HartsyInference.Diffusion.Pipelines;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>Shared base for 3D-generation pipelines, reusing <see cref="DiffusionPipelineBase"/> for the compute backend handle, idempotent disposal, and the convention that pipelines don't own their injected components.</summary>
public abstract class ThreeDPipelineBase : DiffusionPipelineBase
{
    protected ThreeDPipelineBase(IBackend backend) : base(backend) { }
}
