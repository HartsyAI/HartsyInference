using HartsyInference.Core.Backends;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Stages;

namespace HartsyInference.Engine.Services;

/// <summary>A loaded TRELLIS.2 pipeline.</summary>
internal sealed class Trellis2Runtime : MatterGatedRuntime<Trellis2ImageTo3DPipeline>
{
    private Trellis2Runtime(Trellis2ImageTo3DPipeline pipeline) : base(pipeline)
    {
    }

    /// <summary>Loads the pipeline from <paramref name="localPath"/>; <paramref name="matter"/> mattes an image without alpha and stays with its owner.</summary>
    public static Trellis2Runtime Load(IBackend backend, string localPath, IForegroundMatter matter)
        => new(Trellis2ImageTo3DPipeline.LoadFromPath(backend, localPath, matter));
}
