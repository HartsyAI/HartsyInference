using HartsyInference.Core.Backends;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Stages;

namespace HartsyInference.Engine.Services;

/// <summary>A loaded Pixal3D pipeline.</summary>
internal sealed class Pixal3DRuntime : MatterGatedRuntime<Pixal3DImageTo3DPipeline>
{
    private Pixal3DRuntime(Pixal3DImageTo3DPipeline pipeline) : base(pipeline)
    {
    }

    /// <summary>Loads the pipeline from <paramref name="localPath"/>; <paramref name="matter"/> mattes an image without alpha and stays with its owner.</summary>
    public static Pixal3DRuntime Load(IBackend backend, string localPath, IForegroundMatter matter)
        => new(Pixal3DImageTo3DPipeline.LoadFromPath(backend, localPath, matter));
}
